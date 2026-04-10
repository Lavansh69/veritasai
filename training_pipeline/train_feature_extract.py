"""
VeritasAI – Feature Extraction Retraining
Keeps the EfficientNet-B4 backbone FROZEN (from v2), resets the classifier
head, and trains ONLY the head on the user's real/fake dataset.

This avoids the fine-tuning pitfall where the v2 bias corrupts predictions.
"""

import argparse
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
)
from sklearn.model_selection import train_test_split


# ── Dataset ────────────────────────────────────────────────────────

def discover_images(data_dir):
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
    root = Path(data_dir)
    paths, labels = [], []

    for label_name, label_val in [("Real", 0), ("Fake", 1)]:
        folder = root / label_name
        if not folder.exists():
            # Try lowercase
            folder = root / label_name.lower()
        if not folder.exists():
            continue
        for f in sorted(folder.rglob("*")):
            if f.suffix.lower() in exts:
                paths.append(str(f))
                labels.append(label_val)

    return paths, labels


class SimpleDataset(Dataset):
    def __init__(self, paths, labels, image_size=224, augment=False):
        self.paths = paths
        self.labels = labels
        if augment:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
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
        return len(self.paths)

    def __getitem__(self, idx):
        try:
            img = Image.open(self.paths[idx]).convert("RGB")
        except Exception:
            img = Image.new("RGB", (224, 224))
        return self.transform(img), torch.tensor(self.labels[idx], dtype=torch.float32)


# ── Main ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Feature Extraction Retraining")
    parser.add_argument("--data", required=True, help="Dataset root with Real/ and Fake/")
    parser.add_argument("--v2-weights", required=True, help="Path to veritas_model_v2.pth")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate for classifier head")
    parser.add_argument("--output", type=str, default="./output")
    parser.add_argument("--patience", type=int, default=7)
    args = parser.parse_args()

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output, exist_ok=True)

    print("=" * 60)
    print("  VeritasAI - Feature Extraction Retraining")
    print("=" * 60)
    print(f"  Device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    # ── Discover data ──────────────────────────────────────────────
    print("\n[1/5] Loading dataset...")
    paths, labels = discover_images(args.data)
    n_real = sum(1 for l in labels if l == 0)
    n_fake = sum(1 for l in labels if l == 1)
    print(f"  Found {len(paths)} images (Real: {n_real}, Fake: {n_fake})")

    # Stratified split
    train_p, temp_p, train_l, temp_l = train_test_split(
        paths, labels, test_size=0.3, random_state=42, stratify=labels)
    val_p, test_p, val_l, test_l = train_test_split(
        temp_p, temp_l, test_size=0.5, random_state=42, stratify=temp_l)

    print(f"  Train: {len(train_p)}, Val: {len(val_p)}, Test: {len(test_p)}")

    train_ds = SimpleDataset(train_p, train_l, augment=True)
    val_ds = SimpleDataset(val_p, val_l, augment=False)
    test_ds = SimpleDataset(test_p, test_l, augment=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)

    # ── Build model ────────────────────────────────────────────────
    print("\n[2/5] Building model...")

    # Step 1: Build EfficientNet-B4 and load v2 backbone weights
    model = models.efficientnet_b4(weights=None)
    in_features = model.classifier[1].in_features

    # Temporarily set the v2 classifier to load weights
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(p=0.2),
        nn.Linear(256, 1),
    )

    # Load v2 weights (full model including classifier)
    state = torch.load(args.v2_weights, map_location=device, weights_only=True)
    model.load_state_dict(state)
    print(f"  Loaded v2 backbone from: {args.v2_weights}")

    # Step 2: FREEZE the entire backbone
    for param in model.features.parameters():
        param.requires_grad = False
    print("  Backbone: FROZEN (all features layers)")

    # Step 3: RE-INITIALIZE the classifier head (fresh weights!)
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, 512),
        nn.ReLU(),
        nn.BatchNorm1d(512),
        nn.Dropout(p=0.2),
        nn.Linear(512, 128),
        nn.ReLU(),
        nn.BatchNorm1d(128),
        nn.Dropout(p=0.1),
        nn.Linear(128, 1),
    )
    print("  Classifier: FRESH (re-initialized from scratch)")

    # Ensure classifier is trainable
    for param in model.classifier.parameters():
        param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Trainable: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")

    model = model.to(device)

    # ── Loss & Optimizer ───────────────────────────────────────────
    # Class weighting to handle imbalance
    n_tr_real = sum(1 for l in train_l if l == 0)
    n_tr_fake = sum(1 for l in train_l if l == 1)
    pos_weight = torch.tensor([n_tr_real / max(n_tr_fake, 1)]).to(device)
    print(f"  pos_weight: {pos_weight.item():.4f}")

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

    # ── Training ───────────────────────────────────────────────────
    print(f"\n[3/5] Training ({args.epochs} epochs, lr={args.lr})...")
    print("-" * 70)

    best_auc = 0.0
    patience_counter = 0
    best_state = None

    for epoch in range(1, args.epochs + 1):
        # Train
        model.train()
        train_loss, train_correct, train_total = 0, 0, 0
        for images, labels_batch in train_loader:
            images = images.to(device)
            labels_batch = labels_batch.to(device).unsqueeze(1)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            preds = (torch.sigmoid(outputs) >= 0.5).float()
            train_correct += (preds == labels_batch).sum().item()
            train_total += images.size(0)

        train_loss /= train_total
        train_acc = train_correct / train_total

        # Validate
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        val_probs, val_labels_all = [], []
        with torch.no_grad():
            for images, labels_batch in val_loader:
                images = images.to(device)
                labels_batch = labels_batch.to(device).unsqueeze(1)
                outputs = model(images)
                loss = criterion(outputs, labels_batch)
                val_loss += loss.item() * images.size(0)
                probs = torch.sigmoid(outputs)
                preds = (probs >= 0.5).float()
                val_correct += (preds == labels_batch).sum().item()
                val_total += images.size(0)
                val_probs.extend(probs.cpu().numpy().flatten())
                val_labels_all.extend(labels_batch.cpu().numpy().flatten())

        val_loss /= val_total
        val_acc = val_correct / val_total
        try:
            val_auc = roc_auc_score(val_labels_all, val_probs)
        except:
            val_auc = 0.0
        val_f1 = f1_score(val_labels_all, [1 if p >= 0.5 else 0 for p in val_probs], zero_division=0)

        scheduler.step(val_loss)
        lr = optimizer.param_groups[0]["lr"]

        print(f"  Epoch {epoch:2d}/{args.epochs} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} AUC: {val_auc:.4f} F1: {val_f1:.4f} | "
              f"LR: {lr:.1e}")

        if val_auc > best_auc:
            best_auc = val_auc
            patience_counter = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            print(f"    * New best! (AUC: {val_auc:.4f})")
        else:
            patience_counter += 1

        if patience_counter >= args.patience:
            print(f"\n  Early stopping at epoch {epoch}")
            break

    print("-" * 70)

    # ── Load best & save ───────────────────────────────────────────
    if best_state:
        model.load_state_dict(best_state)

    # Save clean state_dict (compatible with inference)
    save_path = os.path.join(args.output, "veritas_model_v4.pth")
    torch.save(model.state_dict(), save_path)
    print(f"\n[4/5] Saved model: {save_path}")

    # ── Test evaluation ────────────────────────────────────────────
    print("\n[5/5] Test set evaluation...")
    model.eval()
    test_probs, test_labels_all = [], []
    with torch.no_grad():
        for images, labels_batch in test_loader:
            images = images.to(device)
            labels_batch = labels_batch.to(device).unsqueeze(1)
            outputs = model(images)
            probs = torch.sigmoid(outputs)
            test_probs.extend(probs.cpu().numpy().flatten())
            test_labels_all.extend(labels_batch.cpu().numpy().flatten())

    test_preds = [1 if p >= 0.5 else 0 for p in test_probs]
    acc = accuracy_score(test_labels_all, test_preds)
    prec = precision_score(test_labels_all, test_preds, zero_division=0)
    rec = recall_score(test_labels_all, test_preds, zero_division=0)
    f1 = f1_score(test_labels_all, test_preds, zero_division=0)
    auc = roc_auc_score(test_labels_all, test_probs)

    cm = confusion_matrix(test_labels_all, test_preds)
    print(f"\n  Accuracy:  {acc:.4f} ({acc*100:.1f}%)")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    print(f"  AUC-ROC:   {auc:.4f}")
    print(f"\n  Confusion Matrix:")
    print(f"                Predicted")
    print(f"                Real    Fake")
    print(f"  Actual Real [ {cm[0][0]:5d}   {cm[0][1]:5d} ]")
    print(f"  Actual Fake [ {cm[1][0]:5d}   {cm[1][1]:5d} ]")
    print(f"\n{classification_report(test_labels_all, test_preds, target_names=['Real','Fake'], zero_division=0)}")

    print("=" * 60)
    print(f"  Model saved to: {save_path}")
    print(f"  Best Val AUC: {best_auc:.4f}")
    print(f"  Test Accuracy: {acc:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
