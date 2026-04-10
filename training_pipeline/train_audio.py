"""
VeritasAI – Audio Deepfake Training Script
Standalone training script for the audio deepfake classifier.
Can be run locally or on Kaggle with GPU.

Usage:
    python train_audio.py --data_dir /path/to/audio_dataset --epochs 30

Dataset structure:
    data_dir/
    ├── real/    (authentic audio files: .wav, .mp3, .flac, .ogg)
    └── fake/    (deepfake / AI-generated audio files)
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

# Allow imports from parent directory
sys.path.insert(0, str(Path(__file__).resolve().parent))

from audio_dataset import AudioDeepfakeDataset
from models.audio_classifier import AudioClassifier


def parse_args():
    parser = argparse.ArgumentParser(description="Train VeritasAI Audio Deepfake Classifier")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to dataset root with real/ and fake/ subdirs")
    parser.add_argument("--output_dir", type=str, default="./audio_output", help="Directory to save model + logs")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--sample_rate", type=int, default=16000, help="Audio sample rate")
    parser.add_argument("--duration", type=int, default=5, help="Audio duration in seconds")
    parser.add_argument("--val_split", type=float, default=0.2, help="Validation split ratio")
    parser.add_argument("--num_workers", type=int, default=2, help="DataLoader workers")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    for batch_idx, (spectrograms, labels) in enumerate(loader):
        spectrograms = spectrograms.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(spectrograms).squeeze(1)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * spectrograms.size(0)
        preds = torch.sigmoid(outputs).detach().cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

        if (batch_idx + 1) % 10 == 0:
            print(f"  Batch {batch_idx + 1}/{len(loader)} — Loss: {loss.item():.4f}")

    avg_loss = total_loss / len(loader.dataset)
    all_preds_bin = [1 if p >= 0.5 else 0 for p in all_preds]
    acc = accuracy_score(all_labels, all_preds_bin)

    return avg_loss, acc


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    for spectrograms, labels in loader:
        spectrograms = spectrograms.to(device)
        labels = labels.to(device)

        outputs = model(spectrograms).squeeze(1)
        loss = criterion(outputs, labels)

        total_loss += loss.item() * spectrograms.size(0)
        preds = torch.sigmoid(outputs).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    all_preds_bin = [1 if p >= 0.5 else 0 for p in all_preds]
    acc = accuracy_score(all_labels, all_preds_bin)
    f1 = f1_score(all_labels, all_preds_bin)

    try:
        auc = roc_auc_score(all_labels, all_preds)
    except ValueError:
        auc = 0.0

    return avg_loss, acc, f1, auc


def main():
    args = parse_args()
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # ── Dataset ────────────────────────────────────────────────────
    print(f"\nLoading dataset from: {args.data_dir}")
    full_dataset = AudioDeepfakeDataset(
        root_dir=args.data_dir,
        sample_rate=args.sample_rate,
        duration=args.duration,
        augment=False,
    )

    if len(full_dataset) == 0:
        print("ERROR: No audio samples found. Check your dataset directory structure.")
        print("Expected: data_dir/real/*.wav and data_dir/fake/*.wav")
        sys.exit(1)

    # Train/val split
    val_size = int(len(full_dataset) * args.val_split)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # Enable augmentation for training subset
    train_aug_dataset = AudioDeepfakeDataset(
        root_dir=args.data_dir,
        sample_rate=args.sample_rate,
        duration=args.duration,
        augment=True,
    )
    # Use same indices as train split
    train_aug_subset = torch.utils.data.Subset(train_aug_dataset, train_dataset.indices)

    train_loader = DataLoader(
        train_aug_subset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
    )

    print(f"Train samples: {train_size}")
    print(f"Val samples:   {val_size}")

    # ── Model ──────────────────────────────────────────────────────
    model = AudioClassifier().to(device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {param_count:,}")

    # ── Training setup ─────────────────────────────────────────────
    criterion = nn.BCEWithLogitsLoss()
    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_auc = 0.0
    best_epoch = 0
    history = []

    print(f"\n{'='*60}")
    print(f"Starting training — {args.epochs} epochs")
    print(f"{'='*60}\n")

    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()

        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )

        # Validate
        val_loss, val_acc, val_f1, val_auc = validate(
            model, val_loader, criterion, device
        )

        scheduler.step()

        epoch_time = time.time() - epoch_start

        # Logging
        lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch:3d}/{args.epochs} "
            f"| Train Loss: {train_loss:.4f}  Acc: {train_acc:.4f} "
            f"| Val Loss: {val_loss:.4f}  Acc: {val_acc:.4f}  F1: {val_f1:.4f}  AUC: {val_auc:.4f} "
            f"| LR: {lr:.6f} | Time: {epoch_time:.1f}s"
        )

        history.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "train_acc": round(train_acc, 4),
            "val_loss": round(val_loss, 4),
            "val_acc": round(val_acc, 4),
            "val_f1": round(val_f1, 4),
            "val_auc": round(val_auc, 4),
            "lr": lr,
            "time_s": round(epoch_time, 1),
        })

        # Save best model
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_epoch = epoch
            best_model_path = output_dir / "veritas_audio_model.pth"
            torch.save(model.state_dict(), str(best_model_path))
            print(f"  ★ New best model saved (AUC: {val_auc:.4f})")

    total_time = time.time() - start_time

    # ── Summary ────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Training complete in {total_time/60:.1f} minutes")
    print(f"Best model: epoch {best_epoch}, val AUC = {best_val_auc:.4f}")
    print(f"Model saved to: {output_dir / 'veritas_audio_model.pth'}")
    print(f"{'='*60}")

    # Save training history
    history_path = output_dir / "audio_training_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"Training history saved to: {history_path}")

    # After training, copy the best model to the backend models/ directory
    print(f"\nTo deploy: copy {output_dir / 'veritas_audio_model.pth'} to backend/models/")


if __name__ == "__main__":
    main()
