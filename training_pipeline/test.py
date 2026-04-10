"""
VeritasAI – Standalone Testing Script
Loads a trained veritas_model.pth, evaluates on test data, and outputs metrics.

Usage:
    python test.py --data ./data --model efficientnet --weights output/veritas_model.pth
"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import DeepfakeDataset
from models.efficientnet import build_efficientnet
from models.xceptionnet import build_xceptionnet
import yaml

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
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

    accuracy = correct / total if total > 0 else 0
    return total_loss / total if total > 0 else 0, accuracy

def main():
    parser = argparse.ArgumentParser(description="Test Trained VeritasAI Model")
    parser.add_argument("--data", required=True, help="Path to test dataset root dir")
    parser.add_argument("--model", default="efficientnet", choices=["efficientnet", "xceptionnet"])
    parser.add_argument("--weights", default="output/veritas_model.pth", help="Path to weights")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using Device: {device}")
    
    # ── Load Model ──────────────────────────────────────────────────
    if args.model == "xceptionnet":
        model = build_xceptionnet()
    else:
        model = build_efficientnet(pretrained=False)
        
    print(f"[INFO] Loading Weights from: {args.weights}")
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.to(device)
    
    # ── Load Data ───────────────────────────────────────────────────
    print(f"[INFO] Loading Test Data from: {args.data}")
    test_ds = DeepfakeDataset(
        args.data, image_size=224, split="test",
        train_ratio=0.7, val_ratio=0.15, augment=False
    )
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)
    
    print(f"[INFO] Test dataset size: {len(test_ds)} images.")
    
    if len(test_ds) == 0:
        print("[FAIL] No images found. Check your dataset layout.")
        return
        
    criterion = nn.BCEWithLogitsLoss()
    
    # ── Evaluate ────────────────────────────────────────────────────
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    
    print("-" * 40)
    print(f"[RESULTS]")
    print(f"Total Test Loss: {test_loss:.4f}")
    print(f"Total Accuracy:  {test_acc * 100:.2f}%")
    print("-" * 40)

if __name__ == "__main__":
    main()
