"""
reorganize_dataset.py — Reorganize multi-source deepfake dataset into real/ + fake/

Your dataset has 10 folders organized by source (Real, Midjourney, DALL-E, etc.)
but VeritasAI needs just two folders: real/ and fake/.

This script copies images into the correct layout WITHOUT modifying your original data.

Usage:
    python reorganize_dataset.py --input "C:\path\to\your\dataset" --output "./data"
"""

import argparse
import shutil
from pathlib import Path


# ─── Image extensions to include ──────────────────────────────────
EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


def reorganize(input_dir: str, output_dir: str):
    root = Path(input_dir)
    out = Path(output_dir)

    if not root.exists():
        print(f"[ERROR] Input directory does not exist: {root}")
        return

    # Create output directories
    (out / "real").mkdir(parents=True, exist_ok=True)
    (out / "fake").mkdir(parents=True, exist_ok=True)

    # ─── Auto-detect folders ──────────────────────────────────────
    # List all subfolders in the dataset
    subfolders = [d for d in root.iterdir() if d.is_dir()]

    if not subfolders:
        print(f"[ERROR] No subfolders found in {root}")
        return

    print(f"[INFO] Found {len(subfolders)} folders in dataset:")
    for sf in subfolders:
        img_count = sum(1 for f in sf.rglob("*") if f.suffix.lower() in EXTS)
        print(f"       → {sf.name:30s}  ({img_count} images)")

    # ─── Classify each folder as real or fake ─────────────────────
    # Folders with these keywords are treated as REAL
    REAL_KEYWORDS = {"real", "authentic", "original", "genuine", "natural"}

    stats = {"real": 0, "fake": 0}

    for folder in subfolders:
        # Determine label: check if folder name contains a "real" keyword
        folder_lower = folder.name.lower().strip()
        is_real = any(kw in folder_lower for kw in REAL_KEYWORDS)
        label = "real" if is_real else "fake"

        # Prefix filenames with source name to avoid collisions
        safe_source = folder.name.replace(" ", "_").replace("-", "_").lower()

        for img_file in folder.rglob("*"):
            if img_file.suffix.lower() not in EXTS:
                continue

            new_name = f"{safe_source}_{img_file.name}"
            dst = out / label / new_name

            # Handle duplicate filenames
            counter = 1
            while dst.exists():
                stem = img_file.stem
                new_name = f"{safe_source}_{stem}_{counter}{img_file.suffix}"
                dst = out / label / new_name
                counter += 1

            shutil.copy2(str(img_file), str(dst))
            stats[label] += 1

    # ─── Summary ──────────────────────────────────────────────────
    total = stats["real"] + stats["fake"]
    print(f"\n{'='*50}")
    print(f"  REORGANIZATION COMPLETE")
    print(f"{'='*50}")
    print(f"  Real images:  {stats['real']:,}")
    print(f"  Fake images:  {stats['fake']:,}")
    print(f"  Total:        {total:,}")
    print(f"  Output:       {out.resolve()}")
    print(f"{'='*50}")

    if stats["real"] == 0:
        print("\n[WARNING] No real images found! Make sure your 'Real' folder")
        print("          name matches. Check the folder names above.")

    if stats["fake"] == 0:
        print("\n[WARNING] No fake images found! Check your folder structure.")

    # Show class balance info
    if stats["real"] > 0 and stats["fake"] > 0:
        ratio = stats["fake"] / stats["real"]
        print(f"\n[INFO] Class ratio: 1 real : {ratio:.1f} fake")
        if ratio > 3:
            print("[WARNING] Heavy class imbalance detected!")
            print("          Consider using weighted loss during training.")
            print("          (Add --weighted-loss flag or see dataset_preparation_guide)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Reorganize deepfake dataset into real/ + fake/ layout"
    )
    parser.add_argument(
        "--input", required=True,
        help="Path to the downloaded dataset (with source-based subfolders)"
    )
    parser.add_argument(
        "--output", default="./data",
        help="Output directory (default: ./data)"
    )
    args = parser.parse_args()

    reorganize(args.input, args.output)
