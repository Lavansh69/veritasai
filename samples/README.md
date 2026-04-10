# 🧪 Test Samples for VeritasAI

Use these files to quickly test the deepfake detection system without needing to find your own images.

---

## 📁 Files Included

| File | Label | Expected Result |
|---|---|---|
| `sample_REAL_face.png` | ✅ REAL | AI should say **"Likely Authentic"** (green) |
| `sample_FAKE_face.png` | ❌ FAKE | AI should say **"Likely Deepfake"** (red) |

---

## 🚀 How to Test

1. Start the app (follow README steps)
2. Open → `http://localhost:3000`
3. Click **Upload**
4. Drag and drop any file from this folder
5. Click **Analyze Media**
6. View the verdict, confidence score, and Grad-CAM heatmap

---

## 🌐 More Public Test Images

Don't have your own deepfake images? Use these free public sources:

### Real Faces
- [This Person Does Not Exist](https://thispersondoesnotexist.com) — click refresh for new AI faces (label: FAKE)
- Wikimedia Commons photos of real people (label: REAL)

### Fake / Deepfake Images
- [FaceForensics++ samples](https://github.com/ondyari/FaceForensics) — public deepfake dataset
- [DFDC Preview Dataset](https://ai.facebook.com/datasets/dfdc/) — Facebook's deepfake dataset

### Audio Testing
- Record yourself speaking (label: REAL)
- Generate AI voice at [ElevenLabs](https://elevenlabs.io) free tier (label: FAKE)
- Upload `.mp3` or `.wav` files to the Audio section on the Upload page

---

## ⚠️ Note

These sample images are provided solely for testing and demonstration purposes.
