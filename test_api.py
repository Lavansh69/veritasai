"""Test all images through the actual running backend API."""
import requests
from pathlib import Path

BASE = "http://localhost:8000"

tests = [
    (r"D:\ffffff\WhatsApp Image 2026-03-13 at 9.22.49 PM.jpeg", "AI-generated fantasy (FAKE)"),
    (r"D:\ffffff\WhatsApp Image 2026-03-17 at 11.41.54 PM.jpeg", "Real WhatsApp photo (REAL)"),
]

# Add training samples
for f in sorted(Path(r"training_pipeline\data\Fake").glob("*.jpeg"))[:3]:
    tests.append((str(f), f"Train FAKE: {f.name[:40]}"))
for f in sorted(Path(r"training_pipeline\data\Real").glob("*.jpeg"))[:3]:
    tests.append((str(f), f"Train REAL: {f.name[:40]}"))

print(f"Testing via API at {BASE}/api/predict")
print(f"{'Description':<50s} {'Prob':>6s} {'Verdict':>20s}")
print("=" * 80)

for path, desc in tests:
    if not Path(path).exists():
        print(f"{desc}: FILE NOT FOUND")
        continue
    
    with open(path, 'rb') as f:
        r = requests.post(f"{BASE}/api/predict", files={'file': f})
    
    if r.status_code == 200:
        data = r.json()
        print(f"{desc:<50s} {data['deepfake_probability']:>6.4f} {data['verdict']:>20s}")
    else:
        print(f"{desc:<50s} ERROR: {r.status_code}")
