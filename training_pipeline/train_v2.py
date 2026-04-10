"""
VeritasAI – V2 Training Script (Supports All Architectures)
Backwards-compatible: the original train.py is NOT modified.

Supports:
  - All original architectures (efficientnet, xceptionnet)
  - New architectures (vit, swin, frequency_fusion)
  - Advanced albumentations augmentations (--advanced-augment)
  - Configurable image resolution (--image-size)

Usage:
    python train_v2.py --config config.yaml --model efficientnet
    python train_v2.py --config config.yaml --model vit --image-size 384 --advanced-augment
    python train_v2.py --config config.yaml --model frequency_fusion --image-size 384 --advanced-augment
    python train_v2.py --config config.yaml --model swin --image-size 384 --epochs 20
"""

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, roc_auc_score
import yaml

# Original dataset and models
from dataset import DeepfakeDataset
from models.efficientnet import build_efficientnet
from models.xceptionnet import build_xceptionnet

# New models
from models.frequency_fusion_net import build_frequency_fusion_net
from models.vit_deepfake import build_vit, build_swin

# Advanced dataset (optional)
from advanced_dataset import AdvancedDeepfakeDataset


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def get_model(name: str, pretrained: bool = True, image_size: int = 384) -> nn.Module:
    """Build the selected model architecture."""
    if name == "xceptionnet":
        return build_xceptionnet()
    elif name == "vit":
        return build_vit(pretrained=pretrained, image_size=image_size)
    elif name == "swin":
        return build_swin(pretrained=pretrained, image_size=image_size)
    elif name == "frequency_fusion":
        return build_frequency_fusion_net(pretrained=pretrained, image_size=image_size)
    else:
        # Default: EfficientNet-B4
        return build_efficientnet(pretrained=pretrained)


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device).unsqueeze(1)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds = (torch.sigmoid(outputs) >= 0.5).float()
        correct += (preds == labels).sum().item()
        total += images.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_probs = []
    all_labels = []

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device).unsqueeze(1)

        outputs = model(images)
        loss = criterion(outputs, labels)

        total_loss += loss.item() * images.size(0)
        probs = torch.sigmoid(outputs)
        preds = (probs >= 0.5).float()
        correct += (preds == labels).sum().item()
        total += images.size(0)

        all_probs.extend(probs.cpu().numpy().flatten().tolist())
        all_labels.extend(labels.cpu().numpy().flatten().tolist())

    accuracy = correct / total
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = 0.0

    return total_loss / total, accuracy, auc


def save_plots(history: dict, output_dir: str):
    """Save training/validation loss and accuracy plots."""
    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].plot(history["train_loss"], label="Train Loss", color="#e74c3c")
    axes[0].plot(history["val_loss"], label="Val Loss", color="#3498db")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(history["train_acc"], label="Train Acc", color="#e74c3c")
    axes[1].plot(history["val_acc"], label="Val Acc", color="#3498db")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(history["val_auc"], label="Val AUC", color="#2ecc71")
    axes[2].set_title("Validation AUC-ROC")
    axes[2].set_xlabel("Epoch")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "training_metrics_v2.png"), dpi=150)
    plt.close()
    print(f"[OK] Plots saved to {output_dir}/training_metrics_v2.png")


def main():
    parser = argparse.ArgumentParser(description="VeritasAI V2 Model Training")
    parser.add_argument("--config", type=str, default="config.yaml", help="Config YAML path")
    parser.add_argument(
        "--model", type=str, default="efficientnet",
        choices=["efficientnet", "xceptionnet", "vit", "swin", "frequency_fusion"],
        help="Model architecture to train",
    )
    parser.add_argument("--data", type=str, help="Override dataset root directory")
    parser.add_argument("--epochs", type=int, help="Override number of epochs")
    parser.add_argument("--batch-size", type=int, help="Override batch size")
    parser.add_argument("--lr", type=float, help="Override learning rate")
    parser.add_argument(
        "--image-size", type=int,
        help="Override image size (default: 224 for efficientnet/xception, 384 for vit/swin/fusion)",
    )
    parser.add_argument(
        "--advanced-augment", action="store_true",
        help="Use AdvancedDeepfakeDataset with albumentations augmentations",
    )
    args = parser.parse_args()

    # Load config
    cfg = load_config(args.config)
    ds_cfg = cfg["dataset"]
    pp_cfg = cfg["preprocessing"]
    model_cfg = cfg["model"]
    train_cfg = cfg["training"]
    out_cfg = cfg["output"]

    # CLI overrides
    data_dir = args.data or ds_cfg["root_dir"]
    arch = args.model
    epochs = args.epochs or train_cfg["epochs"]
    batch_size = args.batch_size or train_cfg["batch_size"]
    lr = args.lr or train_cfg["learning_rate"]
    output_dir = out_cfg["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    # Auto-select image size based on architecture
    if args.image_size:
        image_size = args.image_size
    elif arch in ("vit", "swin", "frequency_fusion"):
        # V2 models default to 384
        pp_v2 = cfg.get("preprocessing_v2", {})
        image_size = pp_v2.get("image_size", 384)
    else:
        image_size = pp_cfg["image_size"]

    use_advanced = args.advanced_augment or (
        arch in ("vit", "swin", "frequency_fusion")
        and cfg.get("preprocessing_v2", {}).get("advanced_augment", False)
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")
    print(f"[INFO] Model: {arch}")
    print(f"[INFO] Image Size: {image_size}")
    print(f"[INFO] Dataset: {data_dir}")
    print(f"[INFO] Advanced Augmentations: {use_advanced}")
    print(f"[INFO] Epochs: {epochs}, Batch: {batch_size}, LR: {lr}")

    # ── Datasets ───────────────────────────────────────────────────
    DatasetClass = AdvancedDeepfakeDataset if use_advanced else DeepfakeDataset

    train_ds = DatasetClass(
        data_dir, image_size=image_size, split="train",
        train_ratio=ds_cfg["train_ratio"], val_ratio=ds_cfg["val_ratio"],
    )

    # Validation & test always use the simpler dataset (no heavy augment)
    val_ds = DatasetClass(
        data_dir, image_size=image_size, split="val",
        train_ratio=ds_cfg["train_ratio"], val_ratio=ds_cfg["val_ratio"],
        augment=False,
    )
    test_ds = DatasetClass(
        data_dir, image_size=image_size, split="test",
        train_ratio=ds_cfg["train_ratio"], val_ratio=ds_cfg["val_ratio"],
        augment=False,
    )

    print(f"[INFO] Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")
    print(f"[INFO] Dataset class: {DatasetClass.__name__}")

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True,
    )

    # ── Model ──────────────────────────────────────────────────────
    pretrained = model_cfg.get("pretrained", True)
    model = get_model(arch, pretrained=pretrained, image_size=image_size).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=train_cfg.get("weight_decay", 1e-5),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=3, factor=0.5
    )

    # Count trainable parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[INFO] Total params: {total_params:,}")
    print(f"[INFO] Trainable params: {trainable_params:,}")

    # ── Training loop ──────────────────────────────────────────────
    history = {
        "train_loss": [], "train_acc": [],
        "val_loss": [], "val_acc": [], "val_auc": [],
    }
    best_auc = 0.0
    patience_counter = 0
    early_stop = train_cfg.get("early_stopping", 7)
    ckpt_interval = train_cfg.get("checkpoint_interval", 5)

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc, val_auc = evaluate(
            model, val_loader, criterion, device
        )
        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_auc"].append(val_auc)

        print(
            f"Epoch {epoch:3d}/{epochs} | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} AUC: {val_auc:.4f}"
        )

        # Save best model
        if val_auc > best_auc:
            best_auc = val_auc
            patience_counter = 0
            if out_cfg.get("save_best", True):
                # Use architecture-specific naming to avoid overwriting
                best_name = f"veritas_{arch}_model.pth"
                best_path = os.path.join(output_dir, best_name)
                torch.save(model.state_dict(), best_path)
                print(f"  → Best model saved (AUC: {val_auc:.4f}) → {best_name}")
        else:
            patience_counter += 1

        # Checkpoint
        if epoch % ckpt_interval == 0:
            ckpt_path = os.path.join(output_dir, f"{arch}_epoch_{epoch}.pth")
            torch.save(model.state_dict(), ckpt_path)

        # Early stopping
        if patience_counter >= early_stop:
            print(f"[INFO] Early stopping at epoch {epoch}")
            break

    # ── Final model save ───────────────────────────────────────────
    final_name = f"veritas_{arch}_model_final.pth"
    final_path = os.path.join(output_dir, final_name)
    torch.save(model.state_dict(), final_path)
    print(f"[OK] Final model saved: {final_path}")

    # ── Test set evaluation ────────────────────────────────────────
    test_loss, test_acc, test_auc = evaluate(
        model, test_loader, criterion, device
    )
    print(f"\n[TEST] Loss: {test_loss:.4f} | Acc: {test_acc:.4f} | AUC: {test_auc:.4f}")

    # ── Save plots ─────────────────────────────────────────────────
    if out_cfg.get("save_plots", True):
        save_plots(history, output_dir)

    print("\n[DONE] V2 Training complete!")
    print(f"[INFO] Architecture: {arch}, Image Size: {image_size}")
    print(f"[INFO] Best Validation AUC: {best_auc:.4f}")


if __name__ == "__main__":
    main()
