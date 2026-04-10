"""
VeritasAI – Add Real Photos & Fine-Tune Script
================================================
This script helps you integrate your 250 real photos from D:\realphoto
into the training pipeline and fine-tune the existing model.

IMPORTANT: You need BOTH real AND fake images to train a deepfake detector.
This script handles the setup and gives you options for obtaining fake data.

Usage:
  Step 1: Setup the dataset folder structure
    python add_real_photos.py --setup --real-dir "D:\realphoto"

  Step 2: Fine-tune (after adding fake images too)
    python add_real_photos.py --train --resume ../backend/models/veritas_model_v2.pth

  Step 3: Deploy the new model to the backend
    python add_real_photos.py --deploy
"""

import argparse
import json
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path


# ═══════════════════════════════════════════════════════════════════
#  1. DATASET SETUP
# ═══════════════════════════════════════════════════════════════════

def setup_dataset(real_photos_dir: str, fake_photos_dir: str = None, data_dir: str = "./data"):
    """
    Copy real and fake photos into the training data structure.
    
    Expected final structure:
        data/
            Real/   ← your real photos
            Fake/   ← your fake/deepfake images
    """
    real_src = Path(real_photos_dir)
    data_root = Path(data_dir)
    real_dest = data_root / "Real"
    fake_dest = data_root / "Fake"

    # Validate source
    if not real_src.exists():
        print(f"[ERROR] Source directory not found: {real_src}")
        sys.exit(1)

    # Count source images
    EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
    src_images = [f for f in real_src.rglob("*") if f.suffix.lower() in EXTS]
    print(f"\n[INFO] Found {len(src_images)} images in {real_src}")

    if len(src_images) == 0:
        print("[ERROR] No image files found in the source directory!")
        sys.exit(1)

    # Create directories
    real_dest.mkdir(parents=True, exist_ok=True)
    fake_dest.mkdir(parents=True, exist_ok=True)

    # Copy real photos
    copied = 0
    skipped = 0
    for img in src_images:
        dest_file = real_dest / img.name
        # Handle duplicate filenames
        if dest_file.exists():
            stem = img.stem
            suffix = img.suffix
            counter = 1
            while dest_file.exists():
                dest_file = real_dest / f"{stem}_{counter}{suffix}"
                counter += 1
        
        shutil.copy2(str(img), str(dest_file))
        copied += 1

    print(f"\n{'='*60}")
    print(f"  DATASET SETUP COMPLETE")
    print(f"{'='*60}")
    print(f"  ✓ Copied {copied} real photos to: {real_dest}")
    print(f"  ✓ Skipped {skipped} duplicates")

    # Copy fake photos if provided
    if fake_photos_dir:
        fake_src = Path(fake_photos_dir)
        if fake_src.exists():
            fake_src_images = [f for f in fake_src.rglob("*") if f.suffix.lower() in EXTS]
            print(f"\n[INFO] Found {len(fake_src_images)} images in {fake_src}")
            fake_copied = 0
            for img in fake_src_images:
                dest_file = fake_dest / img.name
                if dest_file.exists():
                    stem = img.stem
                    suffix = img.suffix
                    counter = 1
                    while dest_file.exists():
                        dest_file = fake_dest / f"{stem}_{counter}{suffix}"
                        counter += 1
                shutil.copy2(str(img), str(dest_file))
                fake_copied += 1
            print(f"  ✓ Copied {fake_copied} fake photos to: {fake_dest}")
        else:
            print(f"[WARNING] Fake directory not found: {fake_src}")
    
    # Check for fake images
    fake_images = [f for f in fake_dest.rglob("*") if f.suffix.lower() in EXTS]
    real_count = len([f for f in real_dest.rglob("*") if f.suffix.lower() in EXTS])
    
    print(f"\n  Current dataset status:")
    print(f"    Real images: {real_count}")
    print(f"    Fake images: {len(fake_images)}")
    
    if len(fake_images) == 0:
        print(f"\n{'='*60}")
        print(f"  ⚠️  WARNING: No fake images found!")
        print(f"{'='*60}")
        print(f"""
  You MUST add fake/deepfake images to: {fake_dest}
  
  Without fake images, the model cannot learn what deepfakes look like.
  
  OPTIONS TO GET FAKE IMAGES:
  ─────────────────────────────────────────────────
  
  Option A: Download from Kaggle (RECOMMENDED)
    1. Go to: https://www.kaggle.com/datasets/manjilkarki/deepfake-and-real-images
       OR:    https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces
    2. Download and extract the 'Fake' folder
    3. Copy fake images to: {fake_dest}
  
  Option B: Use your existing Kaggle dataset
    If you previously trained on 'deepfake-vs-real-60k', download it again
    and copy the Fake folder contents here.
  
  Option C: Generate fake images
    Use tools like FaceSwap, DeepFaceLab, or This-Person-Does-Not-Exist
    to create synthetic faces, then save them to: {fake_dest}
  
  IMPORTANT: For best results, aim for a roughly balanced dataset:
    - If you have 250 real images, get ~250-500 fake images
    - The training script handles mild class imbalance automatically
    
  After adding fake images, run:
    python add_real_photos.py --train --resume ../backend/models/veritas_model_v2.pth
""")
    else:
        ratio = real_count / max(len(fake_images), 1)
        print(f"    Real:Fake ratio: 1:{1/ratio:.1f}")
        
        if ratio > 3 or ratio < 0.33:
            print(f"\n  ⚠️  WARNING: Dataset is highly imbalanced (ratio: {ratio:.2f})")
            print(f"     Recommended range: 1:0.5 to 1:2.0")
            print(f"     The weighted loss function will compensate, but extreme")
            print(f"     imbalance can still hurt performance.")
        
        print(f"\n  ✓ Dataset is ready for training!")
        print(f"  Run: python add_real_photos.py --train --resume ../backend/models/veritas_model_v2.pth")

    return real_count, len(fake_images)


def verify_dataset(data_dir: str = "./data"):
    """Verify the dataset is properly structured and report statistics."""
    data_root = Path(data_dir)
    EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

    print(f"\n{'='*60}")
    print(f"  DATASET VERIFICATION: {data_root.absolute()}")
    print(f"{'='*60}")

    if not data_root.exists():
        print(f"  [ERROR] Data directory does not exist: {data_root}")
        return False

    # Check Real folder
    real_dirs = ["Real", "real", "REAL"]
    real_dir = None
    for name in real_dirs:
        candidate = data_root / name
        if candidate.exists():
            real_dir = candidate
            break

    # Check Fake folder  
    fake_dirs = ["Fake", "fake", "FAKE"]
    fake_dir = None
    for name in fake_dirs:
        candidate = data_root / name
        if candidate.exists():
            fake_dir = candidate
            break

    if real_dir is None:
        print(f"  [ERROR] No 'Real/' folder found in {data_root}")
        return False
    if fake_dir is None:
        print(f"  [ERROR] No 'Fake/' folder found in {data_root}")
        return False

    real_images = [f for f in real_dir.rglob("*") if f.suffix.lower() in EXTS]
    fake_images = [f for f in fake_dir.rglob("*") if f.suffix.lower() in EXTS]

    print(f"  Real folder: {real_dir} ({len(real_images)} images)")
    print(f"  Fake folder: {fake_dir} ({len(fake_images)} images)")
    print(f"  Total:       {len(real_images) + len(fake_images)} images")

    if len(real_images) == 0 or len(fake_images) == 0:
        print(f"\n  [ERROR] Both Real/ and Fake/ must contain images!")
        return False

    # Sample file sizes to detect corrupted images
    import random
    sample = random.sample(real_images + fake_images, min(10, len(real_images) + len(fake_images)))
    issues = []
    for f in sample:
        if f.stat().st_size < 1000:  # Less than 1KB is suspicious
            issues.append(f"    Suspiciously small: {f.name} ({f.stat().st_size} bytes)")

    if issues:
        print(f"\n  ⚠️  Potential issues:")
        for issue in issues:
            print(issue)

    print(f"\n  ✓ Dataset structure is valid!")
    return True


# ═══════════════════════════════════════════════════════════════════
#  2. FINE-TUNING LAUNCHER
# ═══════════════════════════════════════════════════════════════════

def run_training(resume_path: str, data_dir: str = "./data", 
                 epochs: int = 15, batch_size: int = 16, lr: float = 3e-5):
    """Launch train_v3.py with optimal settings for fine-tuning on new real photos."""
    
    # Verify dataset first
    if not verify_dataset(data_dir):
        print("\n[ERROR] Fix the dataset issues above before training.")
        sys.exit(1)

    # Verify model exists
    if resume_path and not Path(resume_path).exists():
        print(f"[ERROR] Model file not found: {resume_path}")
        print(f"  Available models in backend:")
        models_dir = Path(__file__).parent.parent / "backend" / "models"
        if models_dir.exists():
            for f in models_dir.glob("*.pth"):
                print(f"    - {f}")
        sys.exit(1)

    # Build command
    cmd = [
        sys.executable, "train_v3.py",
        "--data", str(data_dir),
        "--epochs", str(epochs),
        "--batch-size", str(batch_size),
        "--lr", str(lr),
        "--image-size", "224",
        "--patience", "7",
        "--unfreeze-layers", "30",
        "--grad-accum", "2",  # Helps with small datasets
        "--num-workers", "0",  # Safe for Windows
    ]

    if resume_path:
        cmd.extend(["--resume", str(resume_path)])

    print(f"\n{'='*60}")
    print(f"  LAUNCHING FINE-TUNING")
    print(f"{'='*60}")
    print(f"  Command: {' '.join(cmd)}")
    print(f"  Resume from: {resume_path or 'Fresh training'}")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {lr}")
    print(f"  Gradient accumulation: 2 steps")
    print(f"{'='*60}\n")

    import subprocess
    result = subprocess.run(cmd, cwd=str(Path(__file__).parent))
    
    if result.returncode == 0:
        print(f"\n✓ Training completed successfully!")
        print(f"  Check ./output/ for the new model version")
    else:
        print(f"\n✗ Training failed with exit code {result.returncode}")
    
    return result.returncode


# ═══════════════════════════════════════════════════════════════════
#  3. DEPLOYMENT 
# ═══════════════════════════════════════════════════════════════════

def deploy_model(output_dir: str = "./output"):
    """Copy the latest trained model to the backend and update the registry."""
    output = Path(output_dir)
    backend_models = Path(__file__).parent.parent / "backend" / "models"
    registry_path = backend_models / "model_registry.json"

    # Find the latest versioned model
    model_files = sorted(output.glob("veritas_model_v*.pth"), key=lambda f: f.stat().st_mtime)
    # Filter out checkpoint files (keep only clean state_dict files)
    model_files = [f for f in model_files if "_2026" not in f.stem]  # Skip timestamped checkpoints

    if not model_files:
        print("[ERROR] No trained model found in output directory!")
        print(f"  Searched: {output}")
        sys.exit(1)

    latest_model = model_files[-1]
    print(f"\n[INFO] Latest model: {latest_model.name}")

    # Find corresponding metadata
    version_str = latest_model.stem.split("_v")[-1]
    meta_files = list(output.glob(f"training_meta_v{version_str}.json"))
    
    metrics = {}
    if meta_files:
        with open(meta_files[0]) as f:
            meta = json.load(f)
            metrics = meta.get("metrics", {})
            print(f"[INFO] Metrics: {json.dumps(metrics, indent=2)}")

    # Copy to backend
    dest_path = backend_models / latest_model.name
    shutil.copy2(str(latest_model), str(dest_path))
    print(f"[OK] Copied model to: {dest_path}")

    # Update registry
    if registry_path.exists():
        with open(registry_path) as f:
            registry = json.load(f)
    else:
        registry = {"models": [], "active_version": 1}

    version = int(version_str)
    
    # Deactivate all existing models
    for m in registry["models"]:
        m["active"] = False

    # Add new model entry
    registry["models"].append({
        "version": version,
        "filename": latest_model.name,
        "path": str(dest_path.absolute()),
        "created_at": datetime.now().isoformat(),
        "metrics": metrics,
        "active": True,
    })
    registry["active_version"] = version

    with open(registry_path, "w") as f:
        json.dump(registry, f, indent=2)

    print(f"[OK] Updated model registry (active version: v{version})")
    print(f"\n{'='*60}")
    print(f"  DEPLOYMENT COMPLETE")
    print(f"{'='*60}")
    print(f"  ✓ Model v{version} is now the active model")
    print(f"  ✓ Restart the backend to load the new model:")
    print(f"     cd ../backend && python main.py")
    print(f"{'='*60}")


# ═══════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="VeritasAI – Add Real Photos & Fine-Tune",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Step 1: Setup dataset with your real photos
  python add_real_photos.py --setup --real-dir "D:\\realphoto"
  
  # Step 1b: Verify dataset (after adding fake images too)
  python add_real_photos.py --verify
  
  # Step 2: Fine-tune the model
  python add_real_photos.py --train --resume ../backend/models/veritas_model_v2.pth
  
  # Step 3: Deploy to backend
  python add_real_photos.py --deploy
        """,
    )
    
    # Actions
    parser.add_argument("--setup", action="store_true", 
                        help="Setup dataset folder with real photos")
    parser.add_argument("--verify", action="store_true",
                        help="Verify dataset structure")
    parser.add_argument("--train", action="store_true",
                        help="Launch fine-tuning")
    parser.add_argument("--deploy", action="store_true",
                        help="Deploy trained model to backend")
    
    # Options
    parser.add_argument("--real-dir", type=str, default=r"D:\realphoto",
                        help="Path to your real photos (default: D:\\realphoto)")
    parser.add_argument("--fake-dir", type=str, default=None,
                        help="Optional: Path to your fake photos")
    parser.add_argument("--data-dir", type=str, default="./data",
                        help="Training data directory (default: ./data)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to model weights to fine-tune from")
    parser.add_argument("--epochs", type=int, default=15,
                        help="Training epochs (default: 15)")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Batch size (default: 16)")
    parser.add_argument("--lr", type=float, default=3e-5,
                        help="Learning rate (default: 3e-5)")

    args = parser.parse_args()

    if not any([args.setup, args.verify, args.train, args.deploy]):
        parser.print_help()
        print("\n[ERROR] Specify at least one action: --setup, --verify, --train, or --deploy")
        sys.exit(1)

    if args.setup:
        setup_dataset(args.real_dir, args.fake_dir, args.data_dir)

    if args.verify:
        verify_dataset(args.data_dir)

    if args.train:
        run_training(args.resume, args.data_dir, args.epochs, args.batch_size, args.lr)

    if args.deploy:
        deploy_model()


if __name__ == "__main__":
    main()
