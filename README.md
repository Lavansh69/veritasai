<div align="center">

# 🛡️ VeritasAI
## AI-Powered Deepfake Detection & Forensic Analysis Platform

*Built for the Hackathon by **The Verifiers**

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)
![Next.js](https://img.shields.io/badge/Next.js-14-black?style=flat-square&logo=next.js)
![PyTorch](https://img.shields.io/badge/PyTorch-EfficientNet--B4-orange?style=flat-square&logo=pytorch)
![Accuracy](https://img.shields.io/badge/Accuracy-96.38%25-brightgreen?style=flat-square)
![AUC](https://img.shields.io/badge/AUC-99.66%25-brightgreen?style=flat-square)

</div>

---

## ⚡ For Judges / Evaluators — Run in 5 Minutes

> **Prerequisites:** Python 3.10+, Node.js 18+, Git

### Step 1 — Clone the repo
```bash
git clone https://github.com/Lavansh69/veritasai.git
cd veritasai
```

### Step 2 — Download the AI Model (69 MB)
1. Go to → [Releases Page](https://github.com/Lavansh69/veritasai/releases/tag/v1.0)
2. Download **`veritas_model_v7.pth`**
3. Place it inside → `backend/models/veritas_model_v7.pth`

### Step 3 — Start the Backend
```bash
cd backend
pip install -r requirements.txt
python -m uvicorn main:app --host 0.0.0.0 --port 8000
```
✅ Backend running at → `http://localhost:8000`
✅ API docs → `http://localhost:8000/docs`

### Step 4 — Start the Frontend (new terminal)
```bash
cd frontend
npm install
npm run dev
```
✅ App running at → `http://localhost:3000`

### Step 5 — Test It
1. Open `http://localhost:3000`
2. Click **Upload** → drop any image or video
3. Click **Analyze Media**
4. See the AI verdict, Grad-CAM heatmap, and scorecard
5. Click **Download PDF Report**

> 🎵 **Audio detection:** Go to the Upload page → scroll down → upload any `.mp3` or `.wav` file

---

## 🎯 What is VeritasAI?

VeritasAI is a production-grade deepfake detection platform that analyses images, videos, and audio for AI-generated manipulation. Upload any suspicious media and get a detailed forensic report in the seconds — including Grad-CAM heatmaps showing exactly where the AI spotted manipulation.

**Built to detect deepfakes shared on WhatsApp, Instagram, and Telegram.**

---

## 🏆 Model Performance

| Model | Accuracy | AUC | Trained On |
|---|---|---|---|
| EfficientNet-B4 (Image/Video) | **96.38%** | **99.66%** | 6,801 real & fake images |
| Custom CNN (Audio) | **95.33%** | — | WaveFake + LJ Speech datasets |

---

## ✨ Features

| Feature | How it works |
|---|---|
| 🔍 **AI Deepfake Detection** | EfficientNet-B4 fine-tuned on 6,801 images. Detects GAN, Diffusion, FaceSwap fakes |
| 🔥 **Grad-CAM Heatmap** | Shows exactly which facial regions triggered the AI's decision |
| 🎵 **Audio Analysis** | Mel-spectrogram CNN detects AI voice cloning & synthetic speech |
| 👤 **Face Comparison** | Compares uploaded face against a reference to detect identity swaps |
| 📊 **Authenticity Scorecard** | Multi-factor 0–100 score combining AI + metadata + identity signals |
| 🗂️ **Metadata Forensics** | EXIF analysis — detects editing software, GPS inconsistencies |
| 📄 **PDF Evidence Report** | Downloadable forensic report with full analysis breakdown |
| 🌗 **Dark / Light Mode** | Glassmorphism UI with smooth animations |

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- Node.js 18+

### 1. Download Model Weights

Download the trained model weights from the [Releases page](../../releases/tag/v1.0) and place them in `backend/models/`:

```
backend/models/veritas_model_v7.pth       ← Image/Video model (69 MB)
```

The audio model is included in the repo at:
```
training_pipeline/audio_output/veritas_audio_model.pth
```

### 2. Start the Backend

```bash
cd backend
pip install -r requirements.txt
python -m uvicorn main:app --host 0.0.0.0 --port 8000
```

API docs available at → `http://localhost:8000/docs`

### 3. Start the Frontend

```bash
cd frontend
npm install
npm run dev
```

App available at → `http://localhost:3000`

---

## 🏗️ Architecture

```
Browser (Next.js 14 + Framer Motion)
         │
         ▼
FastAPI Backend (Python 3.10, Uvicorn)
         │
   ┌─────┴───────┬──────────────┬─────────────────┐
   ▼             ▼              ▼                  ▼
EfficientNet  Audio CNN     Grad-CAM           Metadata
   B4          (Librosa    (PyTorch hooks)    Forensics
(Image/Video)  Mel-spec)                      (PIL EXIF)
   │             │              │                  │
   └─────────────┴──────────────┴──────────────────┘
                          │
                          ▼
               Authenticity Scorecard (0–100)
                          │
                          ▼
               PDF Evidence Report (ReportLab)
```

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/analyze` | Upload image/video for deepfake analysis |
| `POST` | `/api/audio/analyze` | Upload audio for voice clone detection |
| `GET` | `/api/report/{id}` | Download PDF forensic report |
| `GET` | `/api/health` | Health check |

---

## 🗂️ Project Structure

```
veritasai/
├── backend/
│   ├── main.py                  # FastAPI app entry point
│   ├── config.py                # Paths, device, settings
│   ├── requirements.txt
│   ├── routers/
│   │   ├── upload.py            # Image/video analysis endpoint
│   │   ├── audio.py             # Audio analysis endpoint
│   │   └── feedback.py         # User feedback collection
│   ├── services/
│   │   ├── inference.py         # EfficientNet-B4 inference
│   │   ├── live_inference.py    # Real-time detection engine
│   │   ├── explainability.py    # Grad-CAM heatmap generation
│   │   ├── face_analysis.py     # Face detection & comparison
│   │   ├── metadata_analysis.py # EXIF forensics
│   │   ├── scorecard.py         # Multi-factor scoring
│   │   ├── report_generator.py  # PDF report (ReportLab)
│   │   ├── model_manager.py     # Model versioning & registry
│   │   └── media_processing.py  # Frame extraction & preprocessing
│   └── models/                  # Model weights (see Releases)
│
├── training_pipeline/
│   ├── train_v3.py              # Main training script (EfficientNet-B4)
│   ├── train_audio_local.py     # Audio CNN training
│   ├── dataset.py               # PyTorch Dataset with augmentations
│   ├── config.yaml              # Training configuration
│   └── audio_output/
│       └── veritas_audio_model.pth
│
├── frontend/
│   └── src/
│       ├── app/
│       │   ├── page.tsx         # Landing page
│       │   ├── upload/          # Media upload page
│       │   ├── analysis/[id]/   # Results & heatmap page
│       │   ├── report/[id]/     # PDF report page
│       │   └── statistics/      # Threat analytics dashboard
│       └── components/
│           ├── FileUploader.tsx
│           ├── AudioUploader.tsx
│           ├── GaugeChart.tsx
│           ├── RadarScores.tsx
│           ├── HeatmapViewer.tsx
│           ├── ScoreCard.tsx
│           └── ScanningOverlay.tsx
│
├── docker-compose.yml
└── README.md
```

---

## 🔒 Security

- File type & size validation
- Temp files auto-deleted after 30 minutes
- API rate limiting (10 req/min per IP)
- CORS restricted origins
- SHA-256 file hashing for evidence integrity

---

## 📦 Supported Media

- **Images**: JPG, PNG (up to 100 MB)
- **Video**: MP4, MOV (up to 100 MB)
- **Audio**: MP3, WAV, M4A, OGG (up to 100 MB)

---

## 📚 Training Data Sources

- Custom scraped dataset: 6,801 images (real + fake faces)
- [WaveFake](https://github.com/RUB-SysSec/WaveFake) — audio deepfakes
- [LJ Speech](https://keithito.com/LJ-Speech-Dataset/) — real speech baseline
- Stock photos (system) — real face baseline

---

## 📄 License

MIT — Free to use, modify, and distribute.
