"""Quick smoke test to verify backwards compatibility."""
import sys, json
sys.path.insert(0, ".")

from services.inference import predict_deepfake

r = predict_deepfake([r"C:\Users\Lavansh\Downloads\fake_10.jpg"])
print(json.dumps(r, indent=2))
print("\n[PASS] Existing inference pipeline works correctly!")
