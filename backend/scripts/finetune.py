"""
VeritasAI – Incremental Fine-Tuning Script
Fine-tunes the active model on user-feedback data with safe versioning.

Usage:
    cd backend
    python scripts/finetune.py
    python scripts/finetune.py --min-samples 50 --epochs 5 --lr 1e-5

Safety:
    - Never overwrites the base model (veritas_model.pth)
    - Creates a new versioned checkpoint (veritas_model_v{N}.pth)
    - Validates against a held-out split; rolls back if accuracy drops
"""

import argparse
import json
import logging
import os
import random
import sys
from pathlib import Path

# Allow running from backend/ or backend/scripts/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

from config import DEVICE, FEEDBACK_IMAGES_DIR, IMAGE_SIZE, MODEL_DIR
from services.model_manager import ModelManager

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ── Feedback Dataset ───────────────────────────────────────────────
class FeedbackDataset(Dataset):
    """Loads images from feedback_data/images/{real,fake}/ directories."""

    EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

    def __init__(self, root: Path, image_size: int = 224, augment: bool = False):
        self.samples: list[tuple[str, int]] = []

        for label_name, label_val in [("real", 0), ("fake", 1)]:
            label_dir = root / label_name
            if not label_dir.exists():
                continue
            for f in sorted(label_dir.rglob("*")):
                if f.suffix.lower() in self.EXTS:
                    self.samples.append((str(f), label_val))

        random.seed(42)
        random.shuffle(self.samples)

        if augment:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.1, contrast=0.1),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            img = Image.new("RGB", (224, 224))
        return self.transform(img), torch.tensor(label, dtype=torch.float32)


# ── Model Builder (must match training architecture) ───────────────
def _build_efficientnet() -> nn.Module:
    model = models.efficientnet_b4(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(p=0.2),
        nn.Linear(256, 1),
    )
    return model


# ── Training ───────────────────────────────────────────────────────
def finetune(
    min_samples: int = 20,
    epochs: int = 5,
    lr: float = 1e-5,
    val_split: float = 0.2,
    batch_size: int = 16,
):
    """Run incremental fine-tuning on feedback data."""
    manager = ModelManager()

    # Check feedback data
    dataset = FeedbackDataset(FEEDBACK_IMAGES_DIR, IMAGE_SIZE, augment=True)
    total = len(dataset)
    logger.info("Feedback dataset: %d images", total)

    if total < min_samples:
        logger.warning(
            "Not enough feedback data (%d < %d). Skipping fine-tuning.",
            total, min_samples,
        )
        return

    # Split into train/val
    val_size = max(1, int(total * val_split))
    train_size = total - val_size
    train_ds, val_ds = torch.utils.data.random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # Load current active model
    active_path = manager.get_active_model_path()
    active_version = manager.get_active_version()
    logger.info("Loading active model v%d from %s", active_version, active_path)

    model = _build_efficientnet()
    model.load_state_dict(torch.load(str(active_path), map_location=DEVICE))
    model.to(DEVICE)

    # Freeze all layers except classifier (safe fine-tuning)
    for param in model.parameters():
        param.requires_grad = False
    for param in model.classifier.parameters():
        param.requires_grad = True

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
    )

    # Baseline accuracy on validation set
    baseline_acc = _evaluate(model, val_loader, criterion)
    logger.info("Baseline validation accuracy: %.4f", baseline_acc)

    # Train
    best_acc = baseline_acc
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for images, labels in train_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE).unsqueeze(1)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * images.size(0)

        avg_loss = total_loss / train_size
        val_acc = _evaluate(model, val_loader, criterion)
        logger.info(
            "Epoch %d/%d | Loss: %.4f | Val Acc: %.4f",
            epoch, epochs, avg_loss, val_acc,
        )

        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # Decision: save or rollback
    if best_state is None or best_acc <= baseline_acc:
        logger.warning(
            "Fine-tuning did NOT improve accuracy (%.4f → %.4f). Discarding.",
            baseline_acc, best_acc,
        )
        return

    # Save new version
    new_version = active_version + 1
    filename = f"veritas_model_v{new_version}.pth"
    save_path = MODEL_DIR / filename
    torch.save(best_state, str(save_path))

    metrics = {
        "baseline_accuracy": round(baseline_acc, 4),
        "fine_tuned_accuracy": round(best_acc, 4),
        "improvement": round(best_acc - baseline_acc, 4),
        "feedback_samples": total,
        "epochs": epochs,
        "learning_rate": lr,
    }

    manager.register_model(filename, metrics=metrics, set_active=True)
    logger.info(
        "✓ New model v%d saved: %s (accuracy: %.4f → %.4f)",
        new_version, filename, baseline_acc, best_acc,
    )
    logger.info("  Run the backend with --reload to pick up the new model.")


@torch.no_grad()
def _evaluate(model, loader, criterion) -> float:
    model.eval()
    correct = 0
    total = 0
    for images, labels in loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE).unsqueeze(1)
        outputs = model(images)
        preds = (torch.sigmoid(outputs) >= 0.5).float()
        correct += (preds == labels).sum().item()
        total += images.size(0)
    return correct / total if total > 0 else 0.0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VeritasAI Incremental Fine-Tuning")
    parser.add_argument("--min-samples", type=int, default=20, help="Min feedback images required")
    parser.add_argument("--epochs", type=int, default=5, help="Training epochs")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--val-split", type=float, default=0.2, help="Validation split ratio")
    args = parser.parse_args()

    finetune(
        min_samples=args.min_samples,
        epochs=args.epochs,
        lr=args.lr,
        val_split=args.val_split,
        batch_size=args.batch_size,
    )
