"""
VeritasAI - Comprehensive Accuracy Test Suite
Tests both image and audio deepfake detection models on held-out data.
Produces detailed metrics, confusion matrices, and per-source breakdowns.
"""

import os
import sys
import time
import json
import random
from pathlib import Path
from collections import defaultdict

os.environ["PYTHONUNBUFFERED"] = "1"

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
)

# ============================================================
# Configuration
# ============================================================
BASE_DIR = Path(__file__).resolve().parent
BACKEND_DIR = BASE_DIR / "backend"
MODEL_DIR = BACKEND_DIR / "models"
DATA_DIR = BASE_DIR / "training_pipeline" / "data"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_SIZE = 224
SEED = 99  # Different from training seed (42) for true held-out test


# ============================================================
# 1. IMAGE MODEL TEST
# ============================================================
def build_efficientnet():
    """Build the same EfficientNet-B4 architecture as training."""
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


def load_image_model(model_path):
    """Load the trained image deepfake detection model."""
    model = build_efficientnet()
    state_dict = torch.load(str(model_path), map_location=DEVICE, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    return model


def get_image_transform():
    """Standard inference transform (matches training validation transform)."""
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def discover_test_images(data_dir, test_ratio=0.15, seed=SEED):
    """Get a held-out test set that was NOT used in training.
    
    Uses a different seed than training (42) to create truly unseen samples.
    """
    real_dir = data_dir / "Real"
    fake_dir = data_dir / "Fake"
    
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    
    real_files = []
    if real_dir.exists():
        for f in sorted(real_dir.rglob("*")):
            if f.suffix.lower() in exts:
                real_files.append((str(f), 0, get_source_tag(f.name)))
    
    fake_files = []
    if fake_dir.exists():
        for f in sorted(fake_dir.rglob("*")):
            if f.suffix.lower() in exts:
                fake_files.append((str(f), 1, get_source_tag(f.name)))
    
    # Use specific seed for reproducible test split
    random.seed(seed)
    random.shuffle(real_files)
    random.shuffle(fake_files)
    
    # Take last 15% as test (these were NOT part of training with seed=42)
    real_test = real_files[int(len(real_files) * (1 - test_ratio)):]
    fake_test = fake_files[int(len(fake_files) * (1 - test_ratio)):]
    
    return real_test + fake_test


def get_source_tag(filename):
    """Extract source tag from filename."""
    fname = filename.lower()
    if fname.startswith("aivr_real"):
        return "HuggingFace-Real"
    elif fname.startswith("aivr_fake"):
        return "HuggingFace-Fake"
    elif fname.startswith("ruser"):
        return "RandomUser-Real"
    elif fname.startswith("tpdne") or fname.startswith("aigen"):
        return "StyleGAN-Fake"
    elif fname.startswith("whatsapp"):
        return "WhatsApp-Real"
    elif fname.startswith("pexels") or fname.startswith("pixabay"):
        return "StockPhoto-Real"
    elif fname.startswith("hf"):
        return "HuggingFace"
    else:
        return "Other"


@torch.no_grad()
def test_image_model(model, test_samples, transform):
    """Run inference on all test images and collect predictions."""
    all_labels = []
    all_probs = []
    all_preds = []
    all_sources = []
    errors = 0
    
    for i, (path, label, source) in enumerate(test_samples):
        try:
            img = Image.open(path).convert("RGB")
            tensor = transform(img).unsqueeze(0).to(DEVICE)
            
            output = model(tensor)
            prob = torch.sigmoid(output).item()
            pred = 1 if prob >= 0.5 else 0
            
            all_labels.append(label)
            all_probs.append(prob)
            all_preds.append(pred)
            all_sources.append(source)
            
        except Exception as e:
            errors += 1
        
        if (i + 1) % 200 == 0:
            print(f"  Processed {i+1}/{len(test_samples)} images...", flush=True)
    
    return all_labels, all_probs, all_preds, all_sources, errors


# ============================================================
# 2. AUDIO MODEL TEST
# ============================================================
def test_audio_model():
    """Test the audio deepfake detection model."""
    audio_model_path = MODEL_DIR / "veritas_audio_model.pth"
    audio_data_dir = DATA_DIR / "audio_dataset"
    
    if not audio_model_path.exists():
        print("  [SKIP] Audio model not found", flush=True)
        return None
    
    if not audio_data_dir.exists():
        print("  [SKIP] Audio test data not found", flush=True)
        return None
    
    try:
        import librosa
        
        # Add training pipeline to path for model import
        sys.path.insert(0, str(BASE_DIR / "training_pipeline"))
        from models.audio_classifier import AudioClassifier
        
        # Load model
        model = AudioClassifier()
        model.load_state_dict(torch.load(str(audio_model_path), map_location=DEVICE, weights_only=True))
        model.to(DEVICE)
        model.eval()
        
        # Collect test audio files
        real_dir = audio_data_dir / "real"
        fake_dir = audio_data_dir / "fake"
        
        audio_exts = {".wav", ".mp3", ".flac", ".ogg"}
        real_files = sorted([f for f in real_dir.iterdir() if f.suffix.lower() in audio_exts]) if real_dir.exists() else []
        fake_files = sorted([f for f in fake_dir.iterdir() if f.suffix.lower() in audio_exts]) if fake_dir.exists() else []
        
        # Use last 15% as test
        random.seed(SEED)
        random.shuffle(real_files)
        random.shuffle(fake_files)
        
        real_test = real_files[int(len(real_files) * 0.85):]
        fake_test = fake_files[int(len(fake_files) * 0.85):]
        
        test_files = [(f, 0) for f in real_test] + [(f, 1) for f in fake_test]
        
        if not test_files:
            print("  [SKIP] No audio test files found", flush=True)
            return None
        
        print(f"  Testing {len(test_files)} audio samples ({len(real_test)} real + {len(fake_test)} fake)...", flush=True)
        
        sample_rate = 16000
        duration = 5
        target_length = sample_rate * duration
        n_mels = 128
        
        all_labels = []
        all_probs = []
        all_preds = []
        errors = 0
        
        with torch.no_grad():
            for i, (fpath, label) in enumerate(test_files):
                try:
                    # Load and process audio
                    y, _ = librosa.load(str(fpath), sr=sample_rate, mono=True)
                    
                    # Pad or truncate
                    if len(y) < target_length:
                        y = np.pad(y, (0, target_length - len(y)), mode="constant")
                    else:
                        y = y[:target_length]
                    
                    # Mel spectrogram
                    mel = librosa.feature.melspectrogram(y=y, sr=sample_rate, n_mels=n_mels, fmax=8000)
                    mel_db = librosa.power_to_db(mel, ref=np.max)
                    mel_db = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-8)
                    
                    # To tensor
                    tensor = torch.from_numpy(mel_db.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(DEVICE)
                    
                    output = model(tensor)
                    prob = torch.sigmoid(output).item()
                    pred = 1 if prob >= 0.5 else 0
                    
                    all_labels.append(label)
                    all_probs.append(prob)
                    all_preds.append(pred)
                    
                except Exception as e:
                    errors += 1
                
                if (i + 1) % 50 == 0:
                    print(f"  Processed {i+1}/{len(test_files)} audio files...", flush=True)
        
        return {
            "labels": all_labels,
            "probs": all_probs,
            "preds": all_preds,
            "errors": errors,
            "total": len(test_files),
        }
        
    except ImportError as e:
        print(f"  [SKIP] Missing dependency: {e}", flush=True)
        return None


# ============================================================
# 3. REPORTING
# ============================================================
def print_metrics(labels, preds, probs, title="Model"):
    """Print comprehensive metrics."""
    acc = accuracy_score(labels, preds)
    prec = precision_score(labels, preds, zero_division=0)
    rec = recall_score(labels, preds, zero_division=0)
    f1 = f1_score(labels, preds, zero_division=0)
    
    try:
        auc = roc_auc_score(labels, probs)
    except:
        auc = 0.0
    
    cm = confusion_matrix(labels, preds)
    
    print(f"\n  {'='*56}", flush=True)
    print(f"  {title} - ACCURACY REPORT", flush=True)
    print(f"  {'='*56}", flush=True)
    print(f"  Accuracy:    {acc:.4f}  ({acc*100:.2f}%)", flush=True)
    print(f"  Precision:   {prec:.4f}", flush=True)
    print(f"  Recall:      {rec:.4f}", flush=True)
    print(f"  F1-Score:    {f1:.4f}", flush=True)
    print(f"  AUC-ROC:     {auc:.4f}", flush=True)
    print(f"", flush=True)
    print(f"  Confusion Matrix:", flush=True)
    print(f"                    Predicted", flush=True)
    print(f"                    Real    Fake", flush=True)
    if len(cm) >= 2:
        print(f"  Actual Real  [  {cm[0][0]:5d}   {cm[0][1]:5d} ]", flush=True)
        print(f"  Actual Fake  [  {cm[1][0]:5d}   {cm[1][1]:5d} ]", flush=True)
    print(f"", flush=True)
    
    report = classification_report(
        labels, preds, target_names=["Real (0)", "Fake (1)"], zero_division=0
    )
    for line in report.split("\n"):
        print(f"  {line}", flush=True)
    
    print(f"  {'='*56}", flush=True)
    
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "auc": auc}


def print_source_breakdown(labels, preds, probs, sources):
    """Print accuracy breakdown by image source."""
    print(f"\n  {'='*56}", flush=True)
    print(f"  PER-SOURCE ACCURACY BREAKDOWN", flush=True)
    print(f"  {'='*56}", flush=True)
    
    source_data = defaultdict(lambda: {"labels": [], "preds": [], "probs": []})
    
    for label, pred, prob, source in zip(labels, preds, probs, sources):
        source_data[source]["labels"].append(label)
        source_data[source]["preds"].append(pred)
        source_data[source]["probs"].append(prob)
    
    print(f"  {'Source':<25} {'Count':>6} {'Accuracy':>10} {'F1':>8} {'AUC':>8}", flush=True)
    print(f"  {'-'*25} {'-'*6} {'-'*10} {'-'*8} {'-'*8}", flush=True)
    
    for source in sorted(source_data.keys()):
        data = source_data[source]
        n = len(data["labels"])
        acc = accuracy_score(data["labels"], data["preds"])
        f1 = f1_score(data["labels"], data["preds"], zero_division=0)
        try:
            auc = roc_auc_score(data["labels"], data["probs"])
        except:
            auc = 0.0
        
        print(f"  {source:<25} {n:>6} {acc:>10.4f} {f1:>8.4f} {auc:>8.4f}", flush=True)
    
    print(f"  {'='*56}", flush=True)


# ============================================================
# 4. MAIN
# ============================================================
def main():
    print("=" * 60, flush=True)
    print("  VeritasAI - Comprehensive Accuracy Test", flush=True)
    print("=" * 60, flush=True)
    print(f"  Device: {DEVICE}", flush=True)
    if DEVICE == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}", flush=True)
    print(f"  Test seed: {SEED} (different from training seed 42)", flush=True)
    
    results = {}
    
    # ── Test 1: Image Model ────────────────────────────────────
    print(f"\n{'='*60}", flush=True)
    print(f"  TEST 1: IMAGE DEEPFAKE DETECTION MODEL", flush=True)
    print(f"{'='*60}", flush=True)
    
    # Find the best model
    model_candidates = [
        MODEL_DIR / "veritas_model_v7.pth",
        MODEL_DIR / "veritas_model_v6.pth",
        MODEL_DIR / "veritas_model_v5.pth",
        MODEL_DIR / "veritas_model.pth",
    ]
    
    model_path = None
    for p in model_candidates:
        if p.exists():
            model_path = p
            break
    
    if model_path is None:
        print("  [ERROR] No image model found!", flush=True)
    else:
        print(f"  Model: {model_path.name}", flush=True)
        print(f"  Size: {model_path.stat().st_size / 1024 / 1024:.1f} MB", flush=True)
        
        # Load model
        print(f"  Loading model...", flush=True)
        model = load_image_model(model_path)
        transform = get_image_transform()
        
        # Discover test images
        print(f"  Discovering test images...", flush=True)
        test_samples = discover_test_images(DATA_DIR)
        
        n_real = sum(1 for _, l, _ in test_samples if l == 0)
        n_fake = sum(1 for _, l, _ in test_samples if l == 1)
        print(f"  Test set: {len(test_samples)} images ({n_real} real + {n_fake} fake)", flush=True)
        
        # Run inference
        print(f"  Running inference...", flush=True)
        start = time.time()
        labels, probs, preds, sources, errors = test_image_model(model, test_samples, transform)
        elapsed = time.time() - start
        
        print(f"  Inference time: {elapsed:.1f}s ({elapsed/len(test_samples)*1000:.1f}ms/image)", flush=True)
        if errors > 0:
            print(f"  Errors: {errors}", flush=True)
        
        # Print results
        img_metrics = print_metrics(labels, preds, probs, "IMAGE MODEL")
        print_source_breakdown(labels, preds, probs, sources)
        results["image"] = img_metrics
        
        # Cleanup GPU memory
        del model
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
    
    # ── Test 2: Audio Model ────────────────────────────────────
    print(f"\n{'='*60}", flush=True)
    print(f"  TEST 2: AUDIO DEEPFAKE DETECTION MODEL", flush=True)
    print(f"{'='*60}", flush=True)
    
    audio_results = test_audio_model()
    if audio_results:
        audio_metrics = print_metrics(
            audio_results["labels"],
            audio_results["preds"],
            audio_results["probs"],
            "AUDIO MODEL"
        )
        if audio_results["errors"] > 0:
            print(f"  Errors: {audio_results['errors']}", flush=True)
        results["audio"] = audio_metrics
    
    # ── Final Summary ──────────────────────────────────────────
    print(f"\n{'='*60}", flush=True)
    print(f"  VERITASAI - OVERALL ACCURACY SUMMARY", flush=True)
    print(f"{'='*60}", flush=True)
    
    print(f"\n  {'Model':<20} {'Accuracy':>10} {'F1':>8} {'AUC':>8} {'Precision':>10} {'Recall':>8}", flush=True)
    print(f"  {'-'*20} {'-'*10} {'-'*8} {'-'*8} {'-'*10} {'-'*8}", flush=True)
    
    if "image" in results:
        m = results["image"]
        print(f"  {'Image (v7)':<20} {m['accuracy']:>10.4f} {m['f1']:>8.4f} {m['auc']:>8.4f} {m['precision']:>10.4f} {m['recall']:>8.4f}", flush=True)
    
    if "audio" in results:
        m = results["audio"]
        print(f"  {'Audio':<20} {m['accuracy']:>10.4f} {m['f1']:>8.4f} {m['auc']:>8.4f} {m['precision']:>10.4f} {m['recall']:>8.4f}", flush=True)
    
    print(f"\n  {'='*60}", flush=True)
    
    # Grade the system
    if "image" in results:
        img_acc = results["image"]["accuracy"]
        if img_acc >= 0.95:
            grade = "A+ (Excellent)"
        elif img_acc >= 0.90:
            grade = "A  (Very Good)"
        elif img_acc >= 0.85:
            grade = "B+ (Good)"
        elif img_acc >= 0.80:
            grade = "B  (Acceptable)"
        elif img_acc >= 0.70:
            grade = "C  (Needs Improvement)"
        else:
            grade = "D  (Poor)"
        
        print(f"  Image Model Grade:  {grade}", flush=True)
    
    if "audio" in results:
        aud_acc = results["audio"]["accuracy"]
        if aud_acc >= 0.95:
            grade = "A+ (Excellent)"
        elif aud_acc >= 0.90:
            grade = "A  (Very Good)"
        elif aud_acc >= 0.85:
            grade = "B+ (Good)"
        elif aud_acc >= 0.80:
            grade = "B  (Acceptable)"
        else:
            grade = "C  (Needs Improvement)"
        
        print(f"  Audio Model Grade:  {grade}", flush=True)
    
    print(f"\n  {'='*60}", flush=True)
    
    # Save results to JSON
    results_path = BASE_DIR / "accuracy_report.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Results saved to: {results_path}", flush=True)


if __name__ == "__main__":
    main()
