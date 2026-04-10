"""Test forensic analysis on key images."""
import sys
sys.path.insert(0, "backend")
from services.forensic_analysis import compute_forensic_score, analyze_ela, analyze_frequency
from pathlib import Path

tests = [
    (r"D:\ffffff\WhatsApp Image 2026-03-13 at 9.22.49 PM.jpeg", "AI-generated fantasy (FAKE)"),
    (r"D:\ffffff\WhatsApp Image 2026-03-17 at 11.41.54 PM.jpeg", "Real WhatsApp (REAL)"),
]

# Add training samples
real_dir = Path(r"training_pipeline\data\Real")
fake_dir = Path(r"training_pipeline\data\Fake")
for f in sorted(real_dir.glob("*.jpeg"))[:3]:
    tests.append((str(f), f"Train Real: {f.name[:35]}"))
for f in sorted(fake_dir.glob("*.jpeg"))[:3]:
    tests.append((str(f), f"Train Fake: {f.name[:35]}"))

print(f"{'Description':<45s} {'ELA':>5s} {'Freq':>5s} {'Total':>6s} {'Verdict'}")
print("-" * 80)

for path, desc in tests:
    if not Path(path).exists():
        continue
    result = compute_forensic_score(path)
    ela_s = result['ela']['ela_score']
    freq_s = result['frequency']['frequency_score']
    total = result['forensic_score']
    verdict = "SUSPICIOUS" if total >= 40 else "LIKELY OK"
    print(f"{desc:<45s} {ela_s:>5.1f} {freq_s:>5.1f} {total:>6.1f} {verdict}")
    
print("\n\nDetailed analysis for AI-generated image:")
r = compute_forensic_score(r"D:\ffffff\WhatsApp Image 2026-03-13 at 9.22.49 PM.jpeg")
print(f"  ELA: {r['ela']}")
print(f"  Freq: {r['frequency']}")

print("\nDetailed analysis for real WhatsApp:")
r = compute_forensic_score(r"D:\ffffff\WhatsApp Image 2026-03-17 at 11.41.54 PM.jpeg")
print(f"  ELA: {r['ela']}")
print(f"  Freq: {r['frequency']}")
