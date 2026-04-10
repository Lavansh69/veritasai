"""
VeritasAI – V3 Production Retraining Script
============================================
Self-contained, Kaggle-ready training pipeline with:
  ✓ Stratified train/val/test split (no data leakage)
  ✓ Fine-tuning from existing veritas_model.pth
  ✓ Weighted loss for class imbalance
  ✓ Advanced Albumentations augmentations
  ✓ Comprehensive metrics (accuracy, precision, recall, F1, confusion matrix)
  ✓ Versioned model saving (NEVER overwrites the original model)
  ✓ Early stopping + LR scheduling
  ✓ Mixed-precision (AMP) for GPU efficiency
  ✓ Gradient accumulation for effective large batches on limited VRAM

Usage (Kaggle):
    !python train_v3.py \\
        --data /kaggle/input/deepfake-vs-real-classification \\
        --resume /kaggle/input/your-model/veritas_model.pth \\
        --epochs 20 --batch-size 32

Usage (Local – fresh training):
    python train_v3.py --data ./data --epochs 30 --batch-size 16

Usage (Local – fine-tune existing model):
    python train_v3.py --data ./data --resume ./output/veritas_model.pth --epochs 10
"""

import argparse
import json
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend (safe for Kaggle/headless)
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models

# Optional imports (graceful fallback)
try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    _ALBUM_AVAILABLE = True
except ImportError:
    _ALBUM_AVAILABLE = False
    from torchvision import transforms

try:
    from sklearn.metrics import (
        accuracy_score, classification_report, confusion_matrix,
        f1_score, precision_score, recall_score, roc_auc_score,
    )
    from sklearn.model_selection import train_test_split
    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False


# ═══════════════════════════════════════════════════════════════════
#  1. DATASET  (self-contained – no external imports needed)
# ═══════════════════════════════════════════════════════════════════

EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


def discover_images(root_dir: str):
    """Discover images from a Fake/ + Real/ folder structure.

    Returns:
        paths: list[str]  – absolute image paths
        labels: list[int] – 0 = real, 1 = fake
    """
    root = Path(root_dir)
    paths, labels = [], []

    # Try multiple naming conventions
    real_candidates = ["Real", "real", "REAL", "authentic", "Authentic"]
    fake_candidates = ["Fake", "fake", "FAKE", "synthetic", "Synthetic"]

    real_dir = None
    for name in real_candidates:
        candidate = root / name
        if candidate.exists():
            real_dir = candidate
            break

    fake_dir = None
    for name in fake_candidates:
        candidate = root / name
        if candidate.exists():
            fake_dir = candidate
            break

    if real_dir is None and fake_dir is None:
        # Fallback: check if pre-split structure exists (train/real, train/fake)
        children = [d.name for d in root.iterdir()] if root.exists() else []
        raise FileNotFoundError(
            f"Cannot find 'Real/' or 'Fake/' folders in '{root_dir}'.\n"
            f"Found: {children}\n"
            f"Expected: {root}/Real/ + {root}/Fake/"
        )

    if real_dir and real_dir.exists():
        for f in real_dir.rglob("*"):
            if f.suffix.lower() in EXTS:
                paths.append(str(f))
                labels.append(0)

    if fake_dir and fake_dir.exists():
        for f in fake_dir.rglob("*"):
            if f.suffix.lower() in EXTS:
                paths.append(str(f))
                labels.append(1)

    return paths, labels


def stratified_split(paths, labels, train_ratio=0.70, val_ratio=0.15, seed=42):
    """Split data into train/val/test with stratification.

    Maintains the same real:fake ratio across all splits.
    No data leakage – each image appears in exactly one split.
    """
    if not _SKLEARN_AVAILABLE:
        # Manual stratified split fallback
        print("[WARN] scikit-learn not found — using manual split (install sklearn for stratified)")
        combined = list(zip(paths, labels))
        random.seed(seed)
        random.shuffle(combined)
        n = len(combined)
        t = int(n * train_ratio)
        v = int(n * (train_ratio + val_ratio))
        train = combined[:t]
        val = combined[t:v]
        test = combined[v:]
        return (
            ([p for p, _ in train], [l for _, l in train]),
            ([p for p, _ in val], [l for _, l in val]),
            ([p for p, _ in test], [l for _, l in test]),
        )

    # First split: train vs (val+test)
    test_val_ratio = 1.0 - train_ratio
    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        paths, labels,
        test_size=test_val_ratio,
        random_state=seed,
        stratify=labels,
    )

    # Second split: val vs test (from the temp set)
    # val_ratio / (val_ratio + test_ratio)
    test_ratio = 1.0 - train_ratio - val_ratio
    val_fraction = val_ratio / (val_ratio + test_ratio)
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths, temp_labels,
        test_size=(1.0 - val_fraction),
        random_state=seed,
        stratify=temp_labels,
    )

    return (
        (train_paths, train_labels),
        (val_paths, val_labels),
        (test_paths, test_labels),
    )


class DeepfakeDatasetV3(Dataset):
    """PyTorch Dataset with Albumentations augmentations (or torchvision fallback)."""

    def __init__(self, paths, labels, image_size=224, augment=False):
        self.paths = paths
        self.labels = labels
        self.image_size = image_size
        self.augment = augment

        if _ALBUM_AVAILABLE:
            self._setup_albumentations()
        else:
            self._setup_torchvision()

    def _setup_albumentations(self):
        """Build Albumentations transform pipelines."""
        if self.augment:
            self.transform = A.Compose([
                A.Resize(self.image_size, self.image_size),
                A.HorizontalFlip(p=0.5),
                A.Affine(
                    translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)},
                    scale=(0.9, 1.1),
                    rotate=(-15, 15),
                    mode=cv2.BORDER_CONSTANT, p=0.5,
                ),
                # Real-world degradation
                A.OneOf([
                    A.ImageCompression(
                        quality_range=(20, 70),
                        p=1.0,
                    ),
                    A.Downscale(
                        scale_range=(0.25, 0.5),
                        p=1.0,
                    ),
                ], p=0.4),
                # Blur
                A.OneOf([
                    A.GaussianBlur(blur_limit=(3, 7), p=1.0),
                    A.MotionBlur(blur_limit=7, p=1.0),
                ], p=0.25),
                # Noise
                A.OneOf([
                    A.GaussNoise(p=1.0),
                    A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=1.0),
                ], p=0.25),
                # Color / lighting
                A.OneOf([
                    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1.0),
                    A.HueSaturationValue(
                        hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=20, p=1.0,
                    ),
                    A.CLAHE(clip_limit=4.0, p=1.0),
                ], p=0.35),
                # Cutout
                A.CoarseDropout(
                    max_holes=6, max_height=self.image_size // 8,
                    max_width=self.image_size // 8,
                    min_holes=1, min_height=self.image_size // 16,
                    min_width=self.image_size // 16,
                    fill_value=0, p=0.25,
                ),
                # Normalize + convert
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ])
        else:
            self.transform = A.Compose([
                A.Resize(self.image_size, self.image_size),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ])

    def _setup_torchvision(self):
        """Fallback to torchvision transforms if Albumentations is not installed."""
        if self.augment:
            self.transform = transforms.Compose([
                transforms.Resize((self.image_size, self.image_size)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        label = self.labels[idx]

        try:
            if _ALBUM_AVAILABLE:
                img = cv2.imread(path)
                if img is None:
                    raise ValueError(f"Could not read {path}")
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                tensor = self.transform(image=img)["image"]
            else:
                img = Image.open(path).convert("RGB")
                tensor = self.transform(img)
        except Exception:
            # Return a blank image on error (prevents crash)
            if _ALBUM_AVAILABLE:
                blank = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
                tensor = self.transform(image=blank)["image"]
            else:
                blank = Image.new("RGB", (self.image_size, self.image_size))
                tensor = self.transform(blank)

        return tensor, torch.tensor(label, dtype=torch.float32)


# ═══════════════════════════════════════════════════════════════════
#  2. MODEL BUILDER
# ═══════════════════════════════════════════════════════════════════

def build_efficientnet(pretrained=True):
    """EfficientNet-B4 with custom classifier head matching veritas_model.pth."""
    weights = models.EfficientNet_B4_Weights.DEFAULT if pretrained else None
    model = models.efficientnet_b4(weights=weights)

    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(p=0.2),
        nn.Linear(256, 1),
    )
    return model


def prepare_model_for_finetuning(model, unfreeze_last_n=30):
    """Freeze early layers, unfreeze the last N parameters + classifier.

    For fine-tuning: keeps pretrained feature extraction intact while
    allowing the model to adapt to new data through the later layers.
    """
    # First freeze everything
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze the last N parameters (later layers + classifier)
    params = list(model.parameters())
    for param in params[-unfreeze_last_n:]:
        param.requires_grad = True

    # Always unfreeze the full classifier head
    for param in model.classifier.parameters():
        param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"[INFO] Trainable: {trainable:,} / {total:,} parameters "
          f"({100 * trainable / total:.1f}%)")

    return model


# ═══════════════════════════════════════════════════════════════════
#  3. TRAINING & EVALUATION
# ═══════════════════════════════════════════════════════════════════

def train_one_epoch(model, loader, criterion, optimizer, device, scaler=None,
                    grad_accum_steps=1):
    """Train for one epoch with optional mixed-precision and gradient accumulation."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    optimizer.zero_grad()

    for step, (images, labels) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).unsqueeze(1)

        # Mixed precision forward pass
        if scaler is not None:
            with torch.amp.autocast("cuda"):
                outputs = model(images)
                loss = criterion(outputs, labels) / grad_accum_steps
            scaler.scale(loss).backward()

            if (step + 1) % grad_accum_steps == 0 or (step + 1) == len(loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels) / grad_accum_steps
            loss.backward()

            if (step + 1) % grad_accum_steps == 0 or (step + 1) == len(loader):
                optimizer.step()
                optimizer.zero_grad()

        total_loss += loss.item() * grad_accum_steps * images.size(0)
        preds = (torch.sigmoid(outputs) >= 0.5).float()
        correct += (preds == labels).sum().item()
        total += images.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """Evaluate and return loss, accuracy, AUC, plus raw predictions."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_probs = []
    all_labels = []
    all_preds = []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).unsqueeze(1)

        outputs = model(images)
        loss = criterion(outputs, labels)

        total_loss += loss.item() * images.size(0)
        probs = torch.sigmoid(outputs)
        preds = (probs >= 0.5).float()
        correct += (preds == labels).sum().item()
        total += images.size(0)

        all_probs.extend(probs.cpu().numpy().flatten().tolist())
        all_labels.extend(labels.cpu().numpy().flatten().tolist())
        all_preds.extend(preds.cpu().numpy().flatten().tolist())

    accuracy = correct / total
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except Exception:
        auc = 0.0

    return total_loss / total, accuracy, auc, all_labels, all_preds, all_probs


def print_full_metrics(labels, preds, probs, split_name="Test"):
    """Print comprehensive classification metrics."""
    print(f"\n{'='*60}")
    print(f"  {split_name.upper()} SET — FULL EVALUATION REPORT")
    print(f"{'='*60}")

    acc = accuracy_score(labels, preds)
    prec = precision_score(labels, preds, zero_division=0)
    rec = recall_score(labels, preds, zero_division=0)
    f1 = f1_score(labels, preds, zero_division=0)

    try:
        auc = roc_auc_score(labels, probs)
    except Exception:
        auc = 0.0

    print(f"  Accuracy:   {acc:.4f}  ({acc*100:.2f}%)")
    print(f"  Precision:  {prec:.4f}")
    print(f"  Recall:     {rec:.4f}")
    print(f"  F1-Score:   {f1:.4f}")
    print(f"  AUC-ROC:    {auc:.4f}")

    cm = confusion_matrix(labels, preds)
    print(f"\n  Confusion Matrix:")
    print(f"                 Predicted")
    print(f"                 Real    Fake")
    print(f"  Actual Real  [ {cm[0][0]:5d}   {cm[0][1]:5d} ]")
    print(f"  Actual Fake  [ {cm[1][0]:5d}   {cm[1][1]:5d} ]")

    print(f"\n  Classification Report:")
    report = classification_report(
        labels, preds,
        target_names=["Real (0)", "Fake (1)"],
        zero_division=0,
    )
    for line in report.split("\n"):
        print(f"  {line}")

    print(f"{'='*60}\n")
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "auc": auc}


# ═══════════════════════════════════════════════════════════════════
#  4. PLOTTING
# ═══════════════════════════════════════════════════════════════════

def save_plots(history, output_dir, tag="v3"):
    """Save training curves and confusion matrix plot."""
    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Loss
    axes[0].plot(history["train_loss"], label="Train Loss", color="#e74c3c", linewidth=2)
    axes[0].plot(history["val_loss"], label="Val Loss", color="#3498db", linewidth=2)
    axes[0].set_title("Loss", fontsize=14, fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy
    axes[1].plot(history["train_acc"], label="Train Acc", color="#e74c3c", linewidth=2)
    axes[1].plot(history["val_acc"], label="Val Acc", color="#3498db", linewidth=2)
    axes[1].set_title("Accuracy", fontsize=14, fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # AUC
    axes[2].plot(history["val_auc"], label="Val AUC", color="#2ecc71", linewidth=2)
    axes[2].set_title("Validation AUC-ROC", fontsize=14, fontweight="bold")
    axes[2].set_xlabel("Epoch")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, f"training_metrics_{tag}.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[OK] Training plots saved: {plot_path}")


# ═══════════════════════════════════════════════════════════════════
#  5. MODEL SAVING (VERSIONED)
# ═══════════════════════════════════════════════════════════════════

def get_next_version(output_dir):
    """Determine the next version number by scanning existing model files."""
    output = Path(output_dir)
    existing_versions = []

    for f in output.glob("veritas_model_v*.pth"):
        try:
            # Extract version from filename like "veritas_model_v2_20260318.pth"
            name = f.stem  # "veritas_model_v2_20260318"
            parts = name.split("_v")
            if len(parts) >= 2:
                v_part = parts[-1].split("_")[0]  # "2"
                existing_versions.append(int(v_part))
        except (ValueError, IndexError):
            continue

    return max(existing_versions, default=1) + 1


def save_model_versioned(model, optimizer, epoch, metrics, output_dir, version=None):
    """Save model with versioned naming. Never overwrites veritas_model.pth."""
    os.makedirs(output_dir, exist_ok=True)

    if version is None:
        version = get_next_version(output_dir)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"veritas_model_v{version}_{timestamp}.pth"
    filepath = os.path.join(output_dir, filename)

    # Save complete checkpoint (model + optimizer + metadata)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "version": version,
        "metrics": metrics,
        "timestamp": timestamp,
        "architecture": "efficientnet_b4",
    }
    torch.save(checkpoint, filepath)

    # Also save a clean state_dict for direct inference loading
    # (compatible with existing backend inference.py)
    clean_path = os.path.join(output_dir, f"veritas_model_v{version}.pth")
    torch.save(model.state_dict(), clean_path)

    print(f"[SAVE] Full checkpoint: {filepath}")
    print(f"[SAVE] Clean weights:   {clean_path}")
    print(f"[SAVE] Version: v{version}")

    # Save training metadata as JSON
    meta_path = os.path.join(output_dir, f"training_meta_v{version}.json")
    meta = {
        "version": version,
        "timestamp": timestamp,
        "epoch": epoch,
        "architecture": "efficientnet_b4",
        "metrics": {k: round(v, 6) if isinstance(v, float) else v for k, v in metrics.items()},
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    return filepath, version


# ═══════════════════════════════════════════════════════════════════
#  6. MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="VeritasAI V3 – Production Retraining Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Fresh training
  python train_v3.py --data ./data --epochs 30

  # Fine-tune from existing model
  python train_v3.py --data ./data --resume ./output/veritas_model.pth --epochs 10

  # Kaggle (with dataset attached)
  python train_v3.py --data /kaggle/input/deepfake-vs-real-classification \\
                      --resume /kaggle/input/your-model/veritas_model.pth \\
                      --epochs 20 --batch-size 32
        """,
    )
    parser.add_argument("--data", required=True, help="Dataset root (must contain Real/ and Fake/)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to veritas_model.pth to resume/fine-tune from")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=5e-5,
                        help="Learning rate (lower for fine-tuning, default: 5e-5)")
    parser.add_argument("--image-size", type=int, default=224, help="Input image size")
    parser.add_argument("--output", type=str, default="./output", help="Output directory")
    parser.add_argument("--train-ratio", type=float, default=0.70, help="Train split ratio")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Validation split ratio")
    parser.add_argument("--num-workers", type=int, default=2,
                        help="DataLoader workers (reduce to 0 if crashes)")
    parser.add_argument("--patience", type=int, default=7, help="Early stopping patience")
    parser.add_argument("--unfreeze-layers", type=int, default=30,
                        help="Number of last parameters to unfreeze for fine-tuning")
    parser.add_argument("--no-amp", action="store_true", help="Disable mixed precision")
    parser.add_argument("--grad-accum", type=int, default=1,
                        help="Gradient accumulation steps (increase if OOM)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    # ──────────────────────────────────────────────────────────────
    #  Setup
    # ──────────────────────────────────────────────────────────────
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = torch.cuda.is_available() and not args.no_amp
    os.makedirs(args.output, exist_ok=True)

    print("=" * 60)
    print("  VeritasAI V3 – Production Retraining Pipeline")
    print("=" * 60)
    print(f"  Device:          {device}")
    if device.type == "cuda":
        print(f"  GPU:             {torch.cuda.get_device_name(0)}")
        print(f"  GPU Memory:      {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"  Mixed Precision: {use_amp}")
    print(f"  Dataset:         {args.data}")
    print(f"  Resume from:     {args.resume or 'None (fresh training)'}")
    print(f"  Image Size:      {args.image_size}")
    print(f"  Epochs:          {args.epochs}")
    print(f"  Batch Size:      {args.batch_size}")
    print(f"  Learning Rate:   {args.lr}")
    print(f"  Grad Accum:      {args.grad_accum}")
    print(f"  Early Stop:      {args.patience} epochs")
    print(f"  Output:          {args.output}")
    print(f"  Albumentations:  {_ALBUM_AVAILABLE}")
    print(f"  Scikit-learn:    {_SKLEARN_AVAILABLE}")
    print("=" * 60)

    # ──────────────────────────────────────────────────────────────
    #  Discover & Split Data
    # ──────────────────────────────────────────────────────────────
    print("\n[STEP 1/6] Discovering images...")
    all_paths, all_labels = discover_images(args.data)

    n_real = sum(1 for l in all_labels if l == 0)
    n_fake = sum(1 for l in all_labels if l == 1)
    print(f"  Found {len(all_paths):,} images total")
    print(f"  Real: {n_real:,} | Fake: {n_fake:,} | Ratio: 1:{n_fake/max(n_real,1):.1f}")

    if len(all_paths) == 0:
        print("[ERROR] No images found! Check your --data path.")
        sys.exit(1)

    print("\n[STEP 2/6] Stratified splitting...")
    (train_paths, train_labels), (val_paths, val_labels), (test_paths, test_labels) = \
        stratified_split(all_paths, all_labels, args.train_ratio, args.val_ratio, args.seed)

    def count_split(labels, name):
        n_r = sum(1 for l in labels if l == 0)
        n_f = sum(1 for l in labels if l == 1)
        print(f"  {name:8s}: {len(labels):6,} images  (Real: {n_r:,}, Fake: {n_f:,}, "
              f"Ratio: {n_r/(n_r+n_f)*100:.1f}% / {n_f/(n_r+n_f)*100:.1f}%)")

    count_split(train_labels, "Train")
    count_split(val_labels, "Val")
    count_split(test_labels, "Test")

    # ──────────────────────────────────────────────────────────────
    #  Create Datasets & DataLoaders
    # ──────────────────────────────────────────────────────────────
    print("\n[STEP 3/6] Building datasets & dataloaders...")
    train_ds = DeepfakeDatasetV3(train_paths, train_labels, args.image_size, augment=True)
    val_ds = DeepfakeDatasetV3(val_paths, val_labels, args.image_size, augment=False)
    test_ds = DeepfakeDatasetV3(test_paths, test_labels, args.image_size, augment=False)

    # Use persistent_workers only if num_workers > 0
    persistent = args.num_workers > 0

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, persistent_workers=persistent,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True, persistent_workers=persistent,
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True, persistent_workers=persistent,
    )
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches:   {len(val_loader)}")
    print(f"  Test batches:  {len(test_loader)}")

    # ──────────────────────────────────────────────────────────────
    #  Build Model
    # ──────────────────────────────────────────────────────────────
    print("\n[STEP 4/6] Building model...")

    if args.resume:
        # Fine-tuning: load model WITHOUT pretrained weights (we'll load our own)
        print(f"  Loading saved model from: {args.resume}")
        model = build_efficientnet(pretrained=False)

        # Load weights – handle both full checkpoint and state_dict formats
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)

        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            # Full checkpoint format (saved by train_v3)
            model.load_state_dict(checkpoint["model_state_dict"])
            prev_version = checkpoint.get("version", 1)
            prev_epoch = checkpoint.get("epoch", 0)
            print(f"  Loaded checkpoint v{prev_version} (epoch {prev_epoch})")
        else:
            # Plain state_dict format (veritas_model.pth from original training)
            if isinstance(checkpoint, dict) and "model_state_dict" not in checkpoint:
                model.load_state_dict(checkpoint)
            else:
                model.load_state_dict(checkpoint)
            print(f"  Loaded plain state_dict weights")

        # Prepare for fine-tuning (freeze early layers)
        model = prepare_model_for_finetuning(model, args.unfreeze_layers)
        mode_str = "FINE-TUNING"
    else:
        # Fresh training with ImageNet pretrained backbone
        print(f"  Building EfficientNet-B4 with ImageNet pretrained weights")
        model = build_efficientnet(pretrained=True)
        model = prepare_model_for_finetuning(model, args.unfreeze_layers)
        mode_str = "FRESH TRAINING"

    model = model.to(device)
    print(f"  Mode: {mode_str}")

    # ──────────────────────────────────────────────────────────────
    #  Loss, Optimizer, Scheduler
    # ──────────────────────────────────────────────────────────────
    # Weighted loss to handle class imbalance
    n_train_real = sum(1 for l in train_labels if l == 0)
    n_train_fake = sum(1 for l in train_labels if l == 1)
    if n_train_fake > 0 and n_train_real > 0:
        pos_weight = torch.tensor([n_train_real / n_train_fake]).to(device)
        print(f"  Class weight: pos_weight = {pos_weight.item():.4f} "
              f"(compensates for {'more fakes' if n_train_fake > n_train_real else 'more reals'})")
    else:
        pos_weight = None

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=3, factor=0.5, verbose=True,
    )
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    # ──────────────────────────────────────────────────────────────
    #  Training Loop
    # ──────────────────────────────────────────────────────────────
    print(f"\n[STEP 5/6] Training ({args.epochs} epochs)...")
    print("-" * 80)

    history = {
        "train_loss": [], "train_acc": [],
        "val_loss": [], "val_acc": [], "val_auc": [],
    }
    best_auc = 0.0
    best_f1 = 0.0
    patience_counter = 0
    best_model_state = None
    version = get_next_version(args.output)

    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()

        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            scaler=scaler, grad_accum_steps=args.grad_accum,
        )

        # Validate
        val_loss, val_acc, val_auc, val_labels_all, val_preds_all, val_probs_all = evaluate(
            model, val_loader, criterion, device,
        )
        scheduler.step(val_loss)

        # Compute F1 on validation
        val_f1 = f1_score(val_labels_all, val_preds_all, zero_division=0) if _SKLEARN_AVAILABLE else 0.0

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_auc"].append(val_auc)

        epoch_time = time.time() - epoch_start
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} AUC: {val_auc:.4f} F1: {val_f1:.4f} | "
            f"LR: {current_lr:.2e} | {epoch_time:.0f}s"
        )

        # Track best model by AUC
        improved = False
        if val_auc > best_auc:
            best_auc = val_auc
            best_f1 = val_f1
            patience_counter = 0
            improved = True
            # Deep copy the best model state
            best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
            print(f"  * New best model! (AUC: {val_auc:.4f}, F1: {val_f1:.4f})")
        else:
            patience_counter += 1

        # Checkpoint every 5 epochs
        if epoch % 5 == 0:
            ckpt_path = os.path.join(args.output, f"checkpoint_epoch_{epoch}.pth")
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "val_auc": val_auc,
            }, ckpt_path)
            print(f"  [CKPT] Checkpoint saved: {ckpt_path}")

        # Early stopping
        if patience_counter >= args.patience:
            print(f"\n[INFO] Early stopping triggered at epoch {epoch} "
                  f"(no improvement for {args.patience} epochs)")
            break

        # Free GPU memory
        if device.type == "cuda":
            torch.cuda.empty_cache()

    total_time = time.time() - start_time
    print("-" * 80)
    print(f"[INFO] Training complete in {total_time/60:.1f} minutes")

    # ──────────────────────────────────────────────────────────────
    #  Save Best Model (versioned)
    # ──────────────────────────────────────────────────────────────
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"\n[INFO] Loaded best model (AUC: {best_auc:.4f})")

    best_metrics = {
        "best_val_auc": best_auc,
        "best_val_f1": best_f1,
        "final_train_loss": history["train_loss"][-1],
        "final_val_loss": history["val_loss"][-1],
        "epochs_trained": len(history["train_loss"]),
        "image_size": args.image_size,
        "learning_rate": args.lr,
        "batch_size": args.batch_size,
        "resumed_from": args.resume or "none",
        "dataset": args.data,
        "total_images": len(all_paths),
    }

    saved_path, saved_version = save_model_versioned(
        model, optimizer, len(history["train_loss"]),
        best_metrics, args.output, version=version,
    )

    # ──────────────────────────────────────────────────────────────
    #  Final Test Evaluation
    # ──────────────────────────────────────────────────────────────
    print(f"\n[STEP 6/6] Final evaluation on test set...")
    test_loss, test_acc, test_auc, test_labels_all, test_preds_all, test_probs_all = evaluate(
        model, test_loader, criterion, device,
    )

    if _SKLEARN_AVAILABLE:
        test_metrics = print_full_metrics(test_labels_all, test_preds_all, test_probs_all, "Test")
    else:
        print(f"\n[TEST] Loss: {test_loss:.4f} | Acc: {test_acc:.4f} | AUC: {test_auc:.4f}")
        print("[WARN] Install scikit-learn for full metrics (precision, recall, F1, confusion matrix)")
        test_metrics = {"accuracy": test_acc, "auc": test_auc}

    # Save plots
    save_plots(history, args.output, tag=f"v{saved_version}")

    # ──────────────────────────────────────────────────────────────
    #  Summary
    # ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  TRAINING COMPLETE — SUMMARY")
    print("=" * 60)
    print(f"  Model Version:    v{saved_version}")
    print(f"  Best Val AUC:     {best_auc:.4f}")
    print(f"  Test Accuracy:    {test_metrics.get('accuracy', test_acc):.4f}")
    if 'f1' in test_metrics:
        print(f"  Test F1:          {test_metrics['f1']:.4f}")
    print(f"  Saved to:         {saved_path}")
    print(f"  Training Time:    {total_time/60:.1f} minutes")
    print("=" * 60)

    print("\n[NEXT STEPS]")
    print(f"  1. Copy the clean weights to your backend:")
    print(f"     cp {args.output}/veritas_model_v{saved_version}.pth "
          f"<project>/backend/models/")
    print(f"  2. To retrain again later, use --resume:")
    print(f"     python train_v3.py --data <DATA> --resume {saved_path}")
    print(f"  3. The original veritas_model.pth is UNTOUCHED [OK]")


if __name__ == "__main__":
    main()
