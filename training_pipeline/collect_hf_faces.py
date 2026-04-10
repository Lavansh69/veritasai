"""
VeritasAI - HuggingFace Face Dataset Downloader (Fast, Bulk)
Downloads real + fake face images from HuggingFace datasets.

This is the fastest way to build a large training dataset.
Uses streaming to avoid downloading the entire dataset at once.
"""

import os
import sys
import time
from pathlib import Path

# Force unbuffered output on Windows
os.environ["PYTHONUNBUFFERED"] = "1"

from datasets import load_dataset
from PIL import Image

# ============================================================
# Configuration 
# ============================================================
BASE_DIR = Path(__file__).resolve().parent / "data"
REAL_DIR = BASE_DIR / "Real"
FAKE_DIR = BASE_DIR / "Fake"
MIN_SIZE = 224

REAL_DIR.mkdir(parents=True, exist_ok=True)
FAKE_DIR.mkdir(parents=True, exist_ok=True)


def collect_from_dataset(dataset_id, max_real=1000, max_fake=1000, label_key="label", real_val=0, fake_val=1):
    """Download images from a HuggingFace dataset."""
    print(f"\n{'='*60}", flush=True)
    print(f"  Dataset: {dataset_id}", flush=True)
    print(f"  Target: {max_real} real + {max_fake} fake", flush=True)
    print(f"{'='*60}", flush=True)
    
    real_count = 0
    fake_count = 0
    skipped = 0
    errors = 0
    
    try:
        print(f"  Loading dataset (streaming mode)...", flush=True)
        ds = load_dataset(dataset_id, split="train", streaming=True)
        print(f"  Dataset loaded. Starting download...", flush=True)
        
        for i, example in enumerate(ds):
            if real_count >= max_real and fake_count >= max_fake:
                break
            
            try:
                label = example.get(label_key)
                image = example.get("image")
                
                if image is None or label is None:
                    skipped += 1
                    continue
                
                # Ensure PIL Image
                if not isinstance(image, Image.Image):
                    skipped += 1
                    continue
                
                w, h = image.size
                if w < MIN_SIZE or h < MIN_SIZE:
                    skipped += 1
                    continue
                
                # Convert to RGB if needed
                if image.mode != "RGB":
                    image = image.convert("RGB")
                
                if label == real_val and real_count < max_real:
                    filename = f"hf_real_{real_count:05d}.jpg"
                    filepath = REAL_DIR / filename
                    if not filepath.exists():
                        image.save(str(filepath), "JPEG", quality=92)
                    real_count += 1
                    
                    if real_count % 100 == 0:
                        print(f"  REAL: {real_count}/{max_real} | FAKE: {fake_count}/{max_fake} (processed {i+1})", flush=True)
                
                elif label == fake_val and fake_count < max_fake:
                    filename = f"hf_fake_{fake_count:05d}.jpg"
                    filepath = FAKE_DIR / filename
                    if not filepath.exists():
                        image.save(str(filepath), "JPEG", quality=92)
                    fake_count += 1
                    
                    if fake_count % 100 == 0:
                        print(f"  REAL: {real_count}/{max_real} | FAKE: {fake_count}/{max_fake} (processed {i+1})", flush=True)
                
            except Exception as e:
                errors += 1
                if errors % 50 == 0:
                    print(f"  ({errors} errors so far: {e})", flush=True)
                continue
        
    except Exception as e:
        print(f"  ERROR: {e}", flush=True)
    
    print(f"\n  Results: {real_count} real + {fake_count} fake collected", flush=True)
    print(f"  Skipped: {skipped} | Errors: {errors}", flush=True)
    return real_count, fake_count


def collect_from_second_dataset(max_real=500, max_fake=500):
    """Download from deepfake face classification dataset (40 techniques)."""
    dataset_id = "afatwapas/deepfake_face_classification"
    
    print(f"\n{'='*60}", flush=True)
    print(f"  Dataset: {dataset_id}", flush=True)
    print(f"  (40 different deepfake techniques)", flush=True)
    print(f"  Target: {max_real} real + {max_fake} fake", flush=True)
    print(f"{'='*60}", flush=True)
    
    real_count = 0
    fake_count = 0
    errors = 0
    
    try:
        print(f"  Loading dataset (streaming mode)...", flush=True)
        ds = load_dataset(dataset_id, split="train", streaming=True)
        print(f"  Dataset loaded. Starting download...", flush=True)
        
        # Use offset to not overwrite files from first dataset
        real_offset = len(list(REAL_DIR.glob("hf_real_*.jpg")))
        fake_offset = len(list(FAKE_DIR.glob("hf_fake_*.jpg")))
        
        for i, example in enumerate(ds):
            if real_count >= max_real and fake_count >= max_fake:
                break
            
            try:
                label = example.get("label")
                image = example.get("image")
                
                if image is None or label is None:
                    continue
                
                if not isinstance(image, Image.Image):
                    continue
                
                w, h = image.size
                if w < MIN_SIZE or h < MIN_SIZE:
                    continue
                
                if image.mode != "RGB":
                    image = image.convert("RGB")
                
                # label 0 = real, 1 = fake
                if label == 0 and real_count < max_real:
                    idx = real_offset + real_count
                    filename = f"hf2_real_{idx:05d}.jpg"
                    filepath = REAL_DIR / filename
                    if not filepath.exists():
                        image.save(str(filepath), "JPEG", quality=92)
                    real_count += 1
                    if real_count % 100 == 0:
                        print(f"  REAL: {real_count}/{max_real} | FAKE: {fake_count}/{max_fake}", flush=True)
                
                elif label == 1 and fake_count < max_fake:
                    idx = fake_offset + fake_count
                    filename = f"hf2_fake_{idx:05d}.jpg"
                    filepath = FAKE_DIR / filename
                    if not filepath.exists():
                        image.save(str(filepath), "JPEG", quality=92)
                    fake_count += 1
                    if fake_count % 100 == 0:
                        print(f"  REAL: {real_count}/{max_real} | FAKE: {fake_count}/{max_fake}", flush=True)
                
            except Exception as e:
                errors += 1
                continue
        
    except Exception as e:
        print(f"  ERROR: {e}", flush=True)
    
    print(f"\n  Results: {real_count} real + {fake_count} fake collected", flush=True)
    return real_count, fake_count


def collect_randomuser_faces(count=500):
    """Download real face photos from RandomUser.me API (fast, reliable)."""
    import requests
    
    print(f"\n{'='*60}", flush=True)
    print(f"  Source: RandomUser.me API (Real face portraits)", flush=True)
    print(f"  Target: {count} images", flush=True)
    print(f"{'='*60}", flush=True)
    
    collected = 0
    offset = len(list(REAL_DIR.glob("ruser_*.jpg")))
    
    session = requests.Session()
    
    for batch in range(count // 50 + 2):
        if collected >= count:
            break
        
        try:
            url = f"https://randomuser.me/api/?results=50&inc=picture&noinfo&seed=veritasai{batch}"
            response = session.get(url, timeout=15)
            
            if response.status_code != 200:
                continue
            
            users = response.json().get("results", [])
            
            for user in users:
                if collected >= count:
                    break
                
                photo_url = user.get("picture", {}).get("large", "")
                if not photo_url:
                    continue
                
                try:
                    img_response = session.get(photo_url, timeout=10)
                    if img_response.status_code == 200 and len(img_response.content) > 1000:
                        idx = offset + collected
                        filepath = REAL_DIR / f"ruser_real_{idx:05d}.jpg"
                        if not filepath.exists():
                            with open(filepath, "wb") as f:
                                f.write(img_response.content)
                        collected += 1
                        
                        if collected % 50 == 0:
                            print(f"  Downloaded {collected}/{count}", flush=True)
                except Exception:
                    continue
            
            time.sleep(0.2)
            
        except Exception as e:
            print(f"  Batch error: {e}", flush=True)
            time.sleep(1)
    
    print(f"  [OK] Collected {collected} RandomUser portraits", flush=True)
    return collected


def collect_pixabay_faces(count=300):
    """Download real face photos from Pixabay API."""
    import requests
    
    print(f"\n{'='*60}", flush=True)
    print(f"  Source: Pixabay API (Real face photos)", flush=True)
    print(f"  Target: {count} images", flush=True)
    print(f"{'='*60}", flush=True)
    
    API_KEY = "46498122-33f629ba07698b2c26d39e6da"
    
    queries = [
        "face+portrait", "headshot", "person+face", "selfie",
        "woman+face", "man+face", "female+portrait", "male+portrait",
        "young+face", "face+closeup", "smiling+face", "serious+face",
        "people+face", "diverse+portrait", "face+natural", "outdoor+portrait",
        "studio+portrait", "face+glasses", "face+beard", "face+makeup",
    ]
    
    collected = 0
    offset = len(list(REAL_DIR.glob("pixabay_*.jpg")))
    session = requests.Session()
    
    for query in queries:
        if collected >= count:
            break
        
        for page in range(1, 8):
            if collected >= count:
                break
            
            try:
                url = (
                    f"https://pixabay.com/api/?key={API_KEY}"
                    f"&q={query}&image_type=photo&orientation=vertical"
                    f"&min_width=300&min_height=300&per_page=100&page={page}"
                    f"&safesearch=true"
                )
                
                response = session.get(url, timeout=15)
                if response.status_code != 200:
                    continue
                
                hits = response.json().get("hits", [])
                if not hits:
                    break  # No more results for this query
                
                for hit in hits:
                    if collected >= count:
                        break
                    
                    img_url = hit.get("webformatURL", "")
                    if not img_url:
                        continue
                    
                    try:
                        img_response = session.get(img_url, timeout=10)
                        if img_response.status_code == 200 and len(img_response.content) > 3000:
                            idx = offset + collected
                            filepath = REAL_DIR / f"pixabay_real_{idx:05d}.jpg"
                            if not filepath.exists():
                                with open(filepath, "wb") as f:
                                    f.write(img_response.content)
                            collected += 1
                            
                            if collected % 50 == 0:
                                print(f"  Downloaded {collected}/{count}", flush=True)
                    except Exception:
                        continue
                
                time.sleep(0.3)
                
            except Exception:
                time.sleep(1)
    
    print(f"  [OK] Collected {collected} Pixabay faces", flush=True)
    return collected


def main():
    print("=" * 60, flush=True)
    print("  VeritasAI - Bulk Face Image Collection (V2)", flush=True)
    print("=" * 60, flush=True)
    
    existing_real = len(list(REAL_DIR.glob("*")))
    existing_fake = len(list(FAKE_DIR.glob("*")))
    print(f"\n  Existing data: {existing_real} real, {existing_fake} fake", flush=True)
    
    start_time = time.time()
    
    # ── Phase 1: RandomUser.me (fast real faces) ───────────────
    r1 = collect_randomuser_faces(count=500)
    
    # ── Phase 2: Pixabay (high-quality real faces) ─────────────
    r2 = collect_pixabay_faces(count=300)
    
    # ── Phase 3: HuggingFace Dataset 1 (Deepfake-vs-Real-60K) ─
    hr1, hf1 = collect_from_dataset(
        "prithivMLmods/Deepfake-vs-Real-60K",
        max_real=1000,
        max_fake=1000,
    )
    
    # ── Phase 4: HuggingFace Dataset 2 (40 techniques) ────────
    hr2, hf2 = collect_from_second_dataset(max_real=500, max_fake=500)
    
    # ── Summary ────────────────────────────────────────────────
    elapsed = time.time() - start_time
    final_real = len(list(REAL_DIR.glob("*")))
    final_fake = len(list(FAKE_DIR.glob("*")))
    
    print(f"\n{'='*60}", flush=True)
    print(f"  COLLECTION COMPLETE", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"  Time:          {elapsed/60:.1f} minutes", flush=True)
    print(f"  RandomUser:    {r1} real faces", flush=True)
    print(f"  Pixabay:       {r2} real faces", flush=True)
    print(f"  HF Dataset 1:  {hr1} real + {hf1} fake", flush=True)
    print(f"  HF Dataset 2:  {hr2} real + {hf2} fake", flush=True)
    print(f"  ────────────────────────────────────", flush=True)
    print(f"  Total REAL:    {final_real}", flush=True)
    print(f"  Total FAKE:    {final_fake}", flush=True)
    print(f"  Grand Total:   {final_real + final_fake}", flush=True)
    print(f"{'='*60}", flush=True)


if __name__ == "__main__":
    main()
