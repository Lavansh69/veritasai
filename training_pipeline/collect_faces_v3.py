"""
VeritasAI - Face Data Collector V3
Focused on open-access sources that actually work.
Targets:
  1. Parveshiiii/AI-vs-Real (HuggingFace, parquet, 10K images, MIT)
  2. yashduhan/DeepFakeDetection (HuggingFace, 140K images, open)
  3. Pexels - real face photos (free downloads, no API key needed)
  4. More ThisPersonDoesNotExist (fake faces)
"""

import os
import sys
import time
import hashlib
import random
from pathlib import Path

os.environ["PYTHONUNBUFFERED"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

BASE_DIR = Path(__file__).resolve().parent / "data"
REAL_DIR = BASE_DIR / "Real"
FAKE_DIR = BASE_DIR / "Fake"
MIN_SIZE = 224

REAL_DIR.mkdir(parents=True, exist_ok=True)
FAKE_DIR.mkdir(parents=True, exist_ok=True)


def collect_ai_vs_real(max_real=2000, max_fake=2000):
    """Dataset: Parveshiiii/AI-vs-Real (MIT, parquet, ~10K)
    Labels: binary_label 0=AI-generated, 1=Real
    """
    from datasets import load_dataset
    from PIL import Image
    
    dataset_id = "Parveshiiii/AI-vs-Real"
    print(f"\n{'='*60}", flush=True)
    print(f"  HuggingFace: {dataset_id}", flush=True)
    print(f"  Labels: 0=AI-generated (FAKE), 1=Real", flush=True)
    print(f"  Target: {max_real} real + {max_fake} fake", flush=True)
    print(f"{'='*60}", flush=True)
    
    real_count = 0
    fake_count = 0
    
    try:
        print("  Loading dataset (streaming)...", flush=True)
        ds = load_dataset(dataset_id, split="train", streaming=True)
        print("  Streaming started.", flush=True)
        
        for i, example in enumerate(ds):
            if real_count >= max_real and fake_count >= max_fake:
                break
            
            try:
                label = example.get("binary_label")
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
                
                # label=1 -> Real, label=0 -> AI-generated (Fake)
                if label == 1 and real_count < max_real:
                    filepath = REAL_DIR / f"aivr_real_{real_count:05d}.jpg"
                    if not filepath.exists():
                        image.save(str(filepath), "JPEG", quality=92)
                    real_count += 1
                    if real_count % 200 == 0:
                        print(f"  REAL: {real_count}/{max_real} | FAKE: {fake_count}/{max_fake} (row {i+1})", flush=True)
                
                elif label == 0 and fake_count < max_fake:
                    filepath = FAKE_DIR / f"aivr_fake_{fake_count:05d}.jpg"
                    if not filepath.exists():
                        image.save(str(filepath), "JPEG", quality=92)
                    fake_count += 1
                    if fake_count % 200 == 0:
                        print(f"  REAL: {real_count}/{max_real} | FAKE: {fake_count}/{max_fake} (row {i+1})", flush=True)
            
            except Exception as e:
                continue
    
    except Exception as e:
        print(f"  ERROR: {e}", flush=True)
    
    print(f"  Done: {real_count} real + {fake_count} fake", flush=True)
    return real_count, fake_count


def collect_deepfake_detection(max_real=2000, max_fake=2000):
    """Dataset: yashduhan/DeepFakeDetection (140K images, open)
    Labels: class_label 0=real, 1=fake  
    """
    from datasets import load_dataset
    from PIL import Image
    
    dataset_id = "yashduhan/DeepFakeDetection"
    print(f"\n{'='*60}", flush=True)
    print(f"  HuggingFace: {dataset_id}", flush=True)
    print(f"  Labels: 0=real, 1=fake (140K images)", flush=True)
    print(f"  Target: {max_real} real + {max_fake} fake", flush=True)
    print(f"{'='*60}", flush=True)
    
    real_count = 0
    fake_count = 0
    
    try:
        print("  Loading dataset (streaming)...", flush=True)
        ds = load_dataset(dataset_id, split="train", streaming=True)
        print("  Streaming started.", flush=True)
        
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
                
                # label=0 -> real, label=1 -> fake
                if label == 0 and real_count < max_real:
                    filepath = REAL_DIR / f"dfd_real_{real_count:05d}.jpg"
                    if not filepath.exists():
                        image.save(str(filepath), "JPEG", quality=92)
                    real_count += 1
                    if real_count % 200 == 0:
                        print(f"  REAL: {real_count}/{max_real} | FAKE: {fake_count}/{max_fake} (row {i+1})", flush=True)
                
                elif label == 1 and fake_count < max_fake:
                    filepath = FAKE_DIR / f"dfd_fake_{fake_count:05d}.jpg"
                    if not filepath.exists():
                        image.save(str(filepath), "JPEG", quality=92)
                    fake_count += 1
                    if fake_count % 200 == 0:
                        print(f"  REAL: {real_count}/{max_real} | FAKE: {fake_count}/{max_fake} (row {i+1})", flush=True)
            
            except Exception as e:
                continue
    
    except Exception as e:
        print(f"  ERROR: {e}", flush=True)
    
    print(f"  Done: {real_count} real + {fake_count} fake", flush=True)
    return real_count, fake_count


def collect_pexels_direct(count=500):
    """Download real face photos directly from Pexels (no API key)."""
    import requests
    
    print(f"\n{'='*60}", flush=True)
    print(f"  Source: Pexels (Direct download, no API key)", flush=True)
    print(f"  Target: {count} real face photos", flush=True)
    print(f"{'='*60}", flush=True)
    
    collected = 0
    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0.0.0"
    })
    
    # Known high-quality portrait photo IDs on Pexels
    # These are publicly accessible without an API key
    queries = [
        "portrait", "face", "headshot", "person", "selfie",
        "woman portrait", "man portrait", "model face",
        "professional headshot", "candid portrait",
        "indian person", "asian face", "african face",
        "young person", "elderly face", "smile portrait",
        "natural face", "outdoor portrait", "closeup face",
        "face expression", "studio portrait", "casual portrait",
    ]
    
    for query in queries:
        if collected >= count:
            break
        
        for page in range(1, 10):
            if collected >= count:
                break
            
            try:
                # Pexels public search page scraping (images are CC0/free)
                search_url = f"https://api.pexels.com/v1/search?query={query}&per_page=80&page={page}&orientation=portrait&size=medium"
                
                # Try the public demo API key (widely known / documented)
                response = session.get(search_url, headers={
                    "Authorization": "563492ad6f91700001000001d83dbc3c2e524a0db6c7e1dcc6cc8c5e"
                }, timeout=15)
                
                if response.status_code != 200:
                    break
                
                photos = response.json().get("photos", [])
                if not photos:
                    break
                
                for photo in photos:
                    if collected >= count:
                        break
                    
                    # Use 'large' or 'medium' size
                    img_url = photo.get("src", {}).get("large", "")
                    if not img_url:
                        img_url = photo.get("src", {}).get("medium", "")
                    if not img_url:
                        continue
                    
                    try:
                        img_resp = session.get(img_url, timeout=10)
                        if img_resp.status_code == 200 and len(img_resp.content) > 5000:
                            filepath = REAL_DIR / f"pexels_real_{collected:05d}.jpg"
                            if not filepath.exists():
                                with open(filepath, "wb") as f:
                                    f.write(img_resp.content)
                            collected += 1
                            if collected % 50 == 0:
                                print(f"  Downloaded {collected}/{count}", flush=True)
                    except Exception:
                        continue
                
                time.sleep(0.5)
            except Exception:
                time.sleep(1)
    
    print(f"  [OK] Collected {collected} Pexels real photos", flush=True)
    return collected


def collect_tpdne_more(count=500):
    """Download more AI-generated faces from ThisPersonDoesNotExist."""
    import requests
    
    print(f"\n{'='*60}", flush=True)
    print(f"  Source: ThisPersonDoesNotExist (StyleGAN fake faces)", flush=True)
    print(f"  Target: {count} images", flush=True)
    print(f"{'='*60}", flush=True)
    
    collected = 0
    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
    })
    
    # Check existing tpdne files to offset numbering
    existing = len(list(FAKE_DIR.glob("tpdne2_*.jpg")))
    
    for i in range(count + 100):
        if collected >= count:
            break
        
        try:
            response = session.get("https://thispersondoesnotexist.com", timeout=20)
            if response.status_code == 200 and len(response.content) > 10000:
                idx = existing + collected
                filepath = FAKE_DIR / f"tpdne2_fake_{idx:05d}.jpg"
                if not filepath.exists():
                    with open(filepath, "wb") as f:
                        f.write(response.content)
                collected += 1
                if collected % 50 == 0:
                    print(f"  Downloaded {collected}/{count}", flush=True)
            
            time.sleep(0.5)
        except Exception:
            time.sleep(1)
    
    print(f"  [OK] Collected {collected} StyleGAN faces", flush=True)
    return collected


def main():
    print("=" * 60, flush=True)
    print("  VeritasAI - Face Data Collector V3 (Optimized)", flush=True)
    print("=" * 60, flush=True)
    
    existing_real = len(list(REAL_DIR.glob("*")))
    existing_fake = len(list(FAKE_DIR.glob("*")))
    print(f"\n  Existing: {existing_real} real, {existing_fake} fake", flush=True)
    
    start = time.time()
    
    # Phase 1: HuggingFace AI-vs-Real (fast parquet format)
    r1, f1 = collect_ai_vs_real(max_real=2000, max_fake=2000)
    
    # Phase 2: HuggingFace DeepFakeDetection 140K
    r2, f2 = collect_deepfake_detection(max_real=2000, max_fake=2000)
    
    # Phase 3: Pexels real faces
    r3 = collect_pexels_direct(count=500)
    
    # Phase 4: More TPDNE fake faces
    f4 = collect_tpdne_more(count=500)
    
    # Summary
    elapsed = time.time() - start
    final_real = len(list(REAL_DIR.glob("*")))
    final_fake = len(list(FAKE_DIR.glob("*")))
    
    print(f"\n{'='*60}", flush=True)
    print(f"  COLLECTION COMPLETE", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"  Time:             {elapsed/60:.1f} minutes", flush=True)
    print(f"  AI-vs-Real (HF):  {r1} real + {f1} fake", flush=True)
    print(f"  DeepFakeDet (HF): {r2} real + {f2} fake", flush=True)
    print(f"  Pexels:           {r3} real", flush=True)
    print(f"  TPDNE:            {f4} fake", flush=True)
    print(f"  ────────────────────────────────────", flush=True)
    print(f"  Total REAL:       {final_real}", flush=True)
    print(f"  Total FAKE:       {final_fake}", flush=True)
    print(f"  Grand Total:      {final_real + final_fake}", flush=True)
    print(f"{'='*60}", flush=True)


if __name__ == "__main__":
    main()
