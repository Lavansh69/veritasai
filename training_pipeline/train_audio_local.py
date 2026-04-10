"""
VeritasAI – Local Audio Training Launcher
Prepares the dataset from separate real/fake directories and trains the model.

This script:
  1. Creates a temporary merged dataset directory with symlinks
  2. Launches training with optimized settings for local GPU (RTX 4060)

Usage:
    python train_audio_local.py
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

# ── Configuration ─────────────────────────────────────────────────
REAL_DIR = Path(r"D:\realwor")
FAKE_DIR = Path(r"D:\fakewor")
MERGED_DIR = Path(__file__).resolve().parent / "data" / "audio_dataset"
OUTPUT_DIR = Path(__file__).resolve().parent / "audio_output"

# Training hyperparameters (tuned for RTX 4060 8GB VRAM)
EPOCHS = 30
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
SAMPLE_RATE = 16000
DURATION = 5  # seconds
VAL_SPLIT = 0.2
NUM_WORKERS = 4


def prepare_dataset():
    """Create merged dataset directory with real/ and fake/ subdirectories."""
    real_out = MERGED_DIR / "real"
    fake_out = MERGED_DIR / "fake"

    # Create output directories
    real_out.mkdir(parents=True, exist_ok=True)
    fake_out.mkdir(parents=True, exist_ok=True)

    # Count existing files
    existing_real = len(list(real_out.iterdir())) if real_out.exists() else 0
    existing_fake = len(list(fake_out.iterdir())) if fake_out.exists() else 0

    real_files = list(REAL_DIR.glob("*.mp3")) + list(REAL_DIR.glob("*.wav")) + list(REAL_DIR.glob("*.flac"))
    fake_files = list(FAKE_DIR.glob("*.mp3")) + list(FAKE_DIR.glob("*.wav")) + list(FAKE_DIR.glob("*.flac"))

    print(f"Found {len(real_files)} real audio files in {REAL_DIR}")
    print(f"Found {len(fake_files)} fake audio files in {FAKE_DIR}")

    if existing_real >= len(real_files) and existing_fake >= len(fake_files):
        print("Dataset already prepared — skipping copy.")
        return

    # Copy real audio files
    print(f"\nCopying real audio files to {real_out}...")
    copied = 0
    for f in real_files:
        dest = real_out / f.name
        if not dest.exists():
            shutil.copy2(str(f), str(dest))
            copied += 1
    print(f"  Copied {copied} new real files (total: {len(list(real_out.iterdir()))})")

    # Copy fake audio files
    print(f"Copying fake audio files to {fake_out}...")
    copied = 0
    for f in fake_files:
        dest = fake_out / f.name
        if not dest.exists():
            shutil.copy2(str(f), str(dest))
            copied += 1
    print(f"  Copied {copied} new fake files (total: {len(list(fake_out.iterdir()))})")

    print(f"\n[OK] Dataset prepared at: {MERGED_DIR}")
    print(f"  Real: {len(list(real_out.iterdir()))} files")
    print(f"  Fake: {len(list(fake_out.iterdir()))} files")


def run_training():
    """Launch the training script with GPU-optimized settings."""
    train_script = Path(__file__).resolve().parent / "train_audio.py"

    cmd = [
        sys.executable,
        str(train_script),
        "--data_dir", str(MERGED_DIR),
        "--output_dir", str(OUTPUT_DIR),
        "--epochs", str(EPOCHS),
        "--batch_size", str(BATCH_SIZE),
        "--lr", str(LEARNING_RATE),
        "--sample_rate", str(SAMPLE_RATE),
        "--duration", str(DURATION),
        "--val_split", str(VAL_SPLIT),
        "--num_workers", str(NUM_WORKERS),
    ]

    print(f"\n{'='*60}")
    print(f"Starting Audio Deepfake Training")
    print(f"{'='*60}")
    print(f"  Dataset:      {MERGED_DIR}")
    print(f"  Output:       {OUTPUT_DIR}")
    print(f"  Epochs:       {EPOCHS}")
    print(f"  Batch Size:   {BATCH_SIZE}")
    print(f"  LR:           {LEARNING_RATE}")
    print(f"  Sample Rate:  {SAMPLE_RATE} Hz")
    print(f"  Duration:     {DURATION}s")
    print(f"  Val Split:    {VAL_SPLIT}")
    print(f"{'='*60}\n")

    # Run training (inherit stdout/stderr for live output)
    # Set PYTHONIOENCODING to avoid Windows cp1252 Unicode errors
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    result = subprocess.run(cmd, env=env)
    return result.returncode


def deploy_model():
    """Copy the trained model to the backend models/ directory."""
    model_path = OUTPUT_DIR / "veritas_audio_model.pth"
    if not model_path.exists():
        print("[WARN] No model file found - training may have failed.")
        return

    backend_models = Path(__file__).resolve().parent.parent / "backend" / "models"
    backend_models.mkdir(parents=True, exist_ok=True)

    dest = backend_models / "veritas_audio_model.pth"
    shutil.copy2(str(model_path), str(dest))
    print(f"\n[OK] Model deployed to: {dest}")
    print(f"  Size: {model_path.stat().st_size / 1024 / 1024:.2f} MB")


if __name__ == "__main__":
    print("=" * 60)
    print("  VeritasAI Audio Deepfake Model — Local Training")
    print("=" * 60)

    # Step 1: Prepare dataset
    print("\n[1/3] Preparing dataset...")
    prepare_dataset()

    # Step 2: Train
    print("\n[2/3] Training model...")
    exit_code = run_training()

    if exit_code == 0:
        # Step 3: Deploy
        print("\n[3/3] Deploying model...")
        deploy_model()
        print("\n[SUCCESS] Training complete! Model is ready for inference.")
    else:
        print(f"\n[FAILED] Training failed with exit code {exit_code}")
        sys.exit(exit_code)
