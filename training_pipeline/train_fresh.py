"""
VeritasAI – Fresh Training from ImageNet
Trains EfficientNet-B4 from ImageNet pretrained weights (NOT v2)
on the user's real/fake dataset. This avoids v2's bias.
"""
import argparse, os, random, time
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix, classification_report)
from sklearn.model_selection import train_test_split


def discover_images(data_dir):
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
    root = Path(data_dir)
    paths, labels = [], []
    for label_name, label_val in [("Real", 0), ("Fake", 1)]:
        folder = root / label_name
        if not folder.exists():
            folder = root / label_name.lower()
        if not folder.exists():
            continue
        for f in sorted(folder.rglob("*")):
            if f.suffix.lower() in exts:
                paths.append(str(f))
                labels.append(label_val)
    return paths, labels


class RobustDataset(Dataset):
    """Dataset with aggressive augmentation to prevent shortcut learning."""
    def __init__(self, paths, labels, image_size=224, augment=False):
        self.paths = paths
        self.labels = labels
        if augment:
            self.transform = transforms.Compose([
                transforms.Resize((image_size + 32, image_size + 32)),
                transforms.RandomCrop(image_size),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
                transforms.RandomGrayscale(p=0.05),
                transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                transforms.RandomErasing(p=0.1),
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
        except:
            img = Image.new("RGB", (224, 224))
        return self.transform(img), torch.tensor(self.labels[idx], dtype=torch.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--output", default="./output")
    parser.add_argument("--patience", type=int, default=7)
    args = parser.parse_args()

    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output, exist_ok=True)

    print("=" * 60)
    print("  VeritasAI - Fresh Training (ImageNet backbone)")
    print("=" * 60)
    print(f"  Device: {device}")

    # Data
    print("\n[1/5] Loading dataset...")
    paths, labels = discover_images(args.data)
    n_real = sum(1 for l in labels if l == 0)
    n_fake = sum(1 for l in labels if l == 1)
    print(f"  Found {len(paths)} images (Real: {n_real}, Fake: {n_fake})")

    train_p, temp_p, train_l, temp_l = train_test_split(
        paths, labels, test_size=0.3, random_state=42, stratify=labels)
    val_p, test_p, val_l, test_l = train_test_split(
        temp_p, temp_l, test_size=0.5, random_state=42, stratify=temp_l)
    print(f"  Train: {len(train_p)}, Val: {len(val_p)}, Test: {len(test_p)}")

    train_loader = DataLoader(RobustDataset(train_p, train_l, augment=True),
                              batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(RobustDataset(val_p, val_l),
                            batch_size=args.batch_size, num_workers=0, pin_memory=True)
    test_loader = DataLoader(RobustDataset(test_p, test_l),
                             batch_size=args.batch_size, num_workers=0, pin_memory=True)

    # Model - fresh from ImageNet!
    print("\n[2/5] Building model (ImageNet pretrained)...")
    model = models.efficientnet_b4(weights=models.EfficientNet_B4_Weights.DEFAULT)
    in_features = model.classifier[1].in_features

    # Custom classifier head
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(p=0.2),
        nn.Linear(256, 1),
    )

    # Progressive unfreezing: freeze early layers, unfreeze last 2 feature blocks + classifier
    # EfficientNet-B4 has 9 feature blocks (0-8)
    for i, block in enumerate(model.features):
        if i < 6:  # Freeze blocks 0-5
            for param in block.parameters():
                param.requires_grad = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Trainable: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")

    model = model.to(device)

    # Loss
    n_tr_r = sum(1 for l in train_l if l == 0)
    n_tr_f = sum(1 for l in train_l if l == 1)
    pos_weight = torch.tensor([n_tr_r / max(n_tr_f, 1)]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=1e-3)  # Strong weight decay for regularization
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    # Training
    print(f"\n[3/5] Training ({args.epochs} epochs)...")
    print("-" * 70)

    best_auc, patience_counter, best_state = 0.0, 0, None

    for epoch in range(1, args.epochs + 1):
        model.train()
        t_loss, t_correct, t_total = 0, 0, 0
        for images, labs in train_loader:
            images, labs = images.to(device), labs.to(device).unsqueeze(1)
            optimizer.zero_grad()
            if scaler:
                with torch.amp.autocast("cuda"):
                    out = model(images)
                    loss = criterion(out, labs)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                out = model(images)
                loss = criterion(out, labs)
                loss.backward()
                optimizer.step()
            t_loss += loss.item() * images.size(0)
            t_correct += ((torch.sigmoid(out) >= 0.5).float() == labs).sum().item()
            t_total += images.size(0)

        scheduler.step()
        t_loss /= t_total; t_acc = t_correct / t_total

        # Validate
        model.eval()
        v_loss, v_correct, v_total = 0, 0, 0
        v_probs, v_labs = [], []
        with torch.no_grad():
            for images, labs in val_loader:
                images, labs = images.to(device), labs.to(device).unsqueeze(1)
                out = model(images)
                loss = criterion(out, labs)
                v_loss += loss.item() * images.size(0)
                probs = torch.sigmoid(out)
                v_correct += ((probs >= 0.5).float() == labs).sum().item()
                v_total += images.size(0)
                v_probs.extend(probs.cpu().numpy().flatten())
                v_labs.extend(labs.cpu().numpy().flatten())

        v_loss /= v_total; v_acc = v_correct / v_total
        try: v_auc = roc_auc_score(v_labs, v_probs)
        except: v_auc = 0.0
        v_f1 = f1_score(v_labs, [1 if p >= 0.5 else 0 for p in v_probs], zero_division=0)

        lr = optimizer.param_groups[0]["lr"]
        print(f"  Epoch {epoch:2d}/{args.epochs} | "
              f"Train: {t_loss:.4f}/{t_acc:.4f} | "
              f"Val: {v_loss:.4f}/{v_acc:.4f} AUC:{v_auc:.4f} F1:{v_f1:.4f} | LR:{lr:.1e}")

        if v_auc > best_auc:
            best_auc = v_auc; patience_counter = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            print(f"    * Best! (AUC: {v_auc:.4f})")
        else:
            patience_counter += 1
        if patience_counter >= args.patience:
            print(f"\n  Early stopping at epoch {epoch}")
            break

    print("-" * 70)

    # Save
    if best_state:
        model.load_state_dict(best_state)
    save_path = os.path.join(args.output, "veritas_model_v5.pth")
    torch.save(model.state_dict(), save_path)
    print(f"\n[4/5] Saved: {save_path}")

    # Test
    print("\n[5/5] Test evaluation...")
    model.eval()
    t_probs, t_labs = [], []
    with torch.no_grad():
        for images, labs in test_loader:
            images, labs = images.to(device), labs.to(device).unsqueeze(1)
            probs = torch.sigmoid(model(images))
            t_probs.extend(probs.cpu().numpy().flatten())
            t_labs.extend(labs.cpu().numpy().flatten())

    t_preds = [1 if p >= 0.5 else 0 for p in t_probs]
    acc = accuracy_score(t_labs, t_preds)
    auc = roc_auc_score(t_labs, t_probs)
    f1 = f1_score(t_labs, t_preds, zero_division=0)
    cm = confusion_matrix(t_labs, t_preds)

    print(f"\n  Accuracy:  {acc:.4f} ({acc*100:.1f}%)")
    print(f"  F1-Score:  {f1:.4f}")
    print(f"  AUC-ROC:   {auc:.4f}")
    print(f"  Confusion Matrix:")
    print(f"                Real    Fake")
    print(f"  Actual Real [ {cm[0][0]:5d}   {cm[0][1]:5d} ]")
    print(f"  Actual Fake [ {cm[1][0]:5d}   {cm[1][1]:5d} ]")
    print(f"\n{classification_report(t_labs, t_preds, target_names=['Real','Fake'], zero_division=0)}")

    # Quick test on key images
    print("\n" + "=" * 60)
    print("  QUICK TEST ON KEY IMAGES")
    print("=" * 60)
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)), transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    key_tests = [
        (r"D:\ffffff\WhatsApp Image 2026-03-13 at 9.22.49 PM.jpeg", "AI fantasy (SHOULD BE FAKE)"),
        (r"D:\ffffff\WhatsApp Image 2026-04-10 at 8.10.56 PM.jpeg", "AI face man (SHOULD BE FAKE)"),
        (r"D:\ffffff\WhatsApp Image 2026-03-17 at 11.41.54 PM.jpeg", "Real WhatsApp (SHOULD BE REAL)"),
    ]
    for path, desc in key_tests:
        if not Path(path).exists():
            print(f"  {desc}: FILE NOT FOUND"); continue
        img = Image.open(path).convert("RGB")
        tensor = test_transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            logit = model(tensor)
        prob = torch.sigmoid(logit).item()
        label = "FAKE" if prob >= 0.5 else "REAL"
        print(f"  {desc}")
        print(f"    Logit: {logit.item():.4f}, Probability: {prob:.4f} -> {label}")


if __name__ == "__main__":
    main()
