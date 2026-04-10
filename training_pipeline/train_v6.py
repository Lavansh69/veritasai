"""
VeritasAI – v6 Training: Maximum diversity, fully unfrozen backbone
Key changes from v5:
  - Unfreeze ALL backbone layers (not just blocks 6-8)
  - Lower learning rate for backbone vs classifier (discriminative LR)
  - More aggressive augmentation to prevent overfitting to specific generators
  - JPEG re-compression augmentation (simulates WhatsApp)
"""
import argparse, os, random, time, io
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score, 
                             confusion_matrix, classification_report)
from sklearn.model_selection import train_test_split


class JPEGCompression:
    """Augmentation: randomly re-compress as JPEG (simulates WhatsApp)."""
    def __init__(self, quality_range=(30, 85)):
        self.quality_range = quality_range
    
    def __call__(self, img):
        if random.random() < 0.5:
            q = random.randint(*self.quality_range)
            buf = io.BytesIO()
            img.save(buf, format='JPEG', quality=q)
            buf.seek(0)
            img = Image.open(buf).convert('RGB')
        return img


class ResizeJitter:
    """Augmentation: resize to random size then back (simulates sharing)."""
    def __init__(self, target_size=224, jitter=(0.5, 1.5)):
        self.target_size = target_size
        self.jitter = jitter
    
    def __call__(self, img):
        if random.random() < 0.3:
            scale = random.uniform(*self.jitter)
            new_size = int(self.target_size * scale)
            img = img.resize((new_size, new_size), Image.BILINEAR)
            img = img.resize((self.target_size, self.target_size), Image.BILINEAR)
        return img


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


class DiverseDataset(Dataset):
    """Dataset with maximum augmentation diversity."""
    def __init__(self, paths, labels, image_size=224, augment=False):
        self.paths = paths
        self.labels = labels
        if augment:
            self.transform = transforms.Compose([
                transforms.Resize((image_size + 48, image_size + 48)),
                JPEGCompression(quality_range=(30, 85)),  # Simulate WhatsApp
                ResizeJitter(target_size=image_size + 48),
                transforms.RandomCrop(image_size),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(20),
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.15),
                transforms.RandomGrayscale(p=0.05),
                transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 3.0)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                transforms.RandomErasing(p=0.15),
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
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--output", default="./output")
    parser.add_argument("--patience", type=int, default=10)
    args = parser.parse_args()

    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output, exist_ok=True)

    print("=" * 60)
    print("  VeritasAI v6 - Full Backbone Training")
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

    train_loader = DataLoader(DiverseDataset(train_p, train_l, augment=True),
                              batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(DiverseDataset(val_p, val_l),
                            batch_size=args.batch_size, num_workers=0, pin_memory=True)
    test_loader = DataLoader(DiverseDataset(test_p, test_l),
                             batch_size=args.batch_size, num_workers=0, pin_memory=True)

    # Model - FULLY unfrozen backbone
    print("\n[2/5] Building model (FULL backbone training)...")
    model = models.efficientnet_b4(weights=models.EfficientNet_B4_Weights.DEFAULT)
    in_features = model.classifier[1].in_features

    model.classifier = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(p=0.3),
        nn.Linear(256, 1),
    )

    # ALL parameters trainable
    for param in model.parameters():
        param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  All parameters trainable: {trainable:,}")

    model = model.to(device)

    # Discriminative learning rates: lower for backbone, higher for classifier
    backbone_params = list(model.features.parameters())
    classifier_params = list(model.classifier.parameters())
    
    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": 1e-5},       # Backbone: very low LR
        {"params": classifier_params, "lr": 5e-4},     # Classifier: higher LR
    ], weight_decay=1e-2)  # Strong weight decay

    # Loss with class balancing
    n_tr_r = sum(1 for l in train_l if l == 0)
    n_tr_f = sum(1 for l in train_l if l == 1)
    pos_weight = torch.tensor([n_tr_r / max(n_tr_f, 1)]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2)
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    # Training
    print(f"\n[3/5] Training ({args.epochs} epochs)...")
    print("-" * 75)

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
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                out = model(images)
                loss = criterion(out, labs)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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

        bb_lr = optimizer.param_groups[0]["lr"]
        cl_lr = optimizer.param_groups[1]["lr"]
        print(f"  E{epoch:2d}/{args.epochs} | "
              f"T:{t_loss:.3f}/{t_acc:.3f} | "
              f"V:{v_loss:.3f}/{v_acc:.3f} AUC:{v_auc:.4f} F1:{v_f1:.4f} | "
              f"BB:{bb_lr:.1e} CL:{cl_lr:.1e}")

        if v_auc > best_auc:
            best_auc = v_auc; patience_counter = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            print(f"    * Best! (AUC: {v_auc:.4f})")
        else:
            patience_counter += 1
        if patience_counter >= args.patience:
            print(f"\n  Early stopping at epoch {epoch}")
            break

    print("-" * 75)

    # Save
    if best_state:
        model.load_state_dict(best_state)
    save_path = os.path.join(args.output, "veritas_model_v6.pth")
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

    # Quick test
    print("\n" + "=" * 60)
    print("  QUICK TEST ON KEY IMAGES")
    print("=" * 60)
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)), transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    key_tests = [
        (r"D:\ffffff\WhatsApp Image 2026-04-10 at 8.10.56 PM.jpeg", "AI face man (SHOULD BE FAKE)"),
        (r"D:\ffffff\WhatsApp Image 2026-03-13 at 9.22.49 PM.jpeg", "AI fantasy (SHOULD BE FAKE)"),
        (r"D:\ffffff\WhatsApp Image 2026-03-17 at 11.41.54 PM.jpeg", "Real WhatsApp (SHOULD BE REAL)"),
        (r"D:\ffffff\WhatsApp Image 2026-04-10 at 9.06.17 PM.jpeg", "D:\\ image check"),
        (r"D:\ffffff\WhatsApp Image 2026-04-10 at 9.07.16 PM.jpeg", "D:\\ image check 2"),
    ]
    for path, desc in key_tests:
        if not Path(path).exists():
            continue
        img = Image.open(path).convert("RGB")
        tensor = test_transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            logit = model(tensor)
        prob = torch.sigmoid(logit).item()
        label = "FAKE" if prob >= 0.5 else "REAL"
        print(f"  {desc}")
        print(f"    Prob: {prob:.4f} -> {label}")


if __name__ == "__main__":
    main()
