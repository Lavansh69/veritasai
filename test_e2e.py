"""End-to-end test: simulate full analysis pipeline."""
import sys
sys.path.insert(0, "backend")

from services.inference import predict_deepfake
from services.metadata_analysis import analyze_metadata
from services.forensic_analysis import compute_forensic_score
from services.scorecard import compute_scorecard
from pathlib import Path

tests = [
    (r"D:\ffffff\WhatsApp Image 2026-03-13 at 9.22.49 PM.jpeg", "AI-generated fantasy (SHOULD BE FAKE)"),
    (r"D:\ffffff\WhatsApp Image 2026-03-17 at 11.41.54 PM.jpeg", "Real WhatsApp (SHOULD BE REAL)"),
]

for f in sorted(Path(r"training_pipeline\data\Fake").glob("*.jpeg"))[:2]:
    tests.append((str(f), f"Train Fake: {f.name[:35]}"))
for f in sorted(Path(r"training_pipeline\data\Real").glob("*.jpeg"))[:2]:
    tests.append((str(f), f"Train Real: {f.name[:35]}"))

print(f"{'Description':<45s} {'M.Prob':>6s} {'Foren':>5s} {'Meta':>5s} {'Score':>6s} {'Verdict':>15s}")
print("=" * 90)

for path, desc in tests:
    if not Path(path).exists():
        print(f"{desc}: FILE NOT FOUND")
        continue
    
    # Model prediction
    model_result = predict_deepfake([path])
    dp = model_result["probability"]
    
    # Metadata
    meta = analyze_metadata(path)
    mi = meta["integrity_score"]
    
    # Forensic
    forensic = compute_forensic_score(path)
    fs = forensic["forensic_score"]
    
    # Scorecard
    sc = compute_scorecard(
        deepfake_prob=dp,
        face_consistency=50.0,  # no reference
        metadata_integrity=mi,
        forensic_score=fs,
    )
    
    print(f"{desc:<45s} {dp:>6.3f} {fs:>5.1f} {mi:>5.1f} {sc['overall_score']:>6.1f} {sc['verdict']:>15s}")
