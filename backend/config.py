"""
VeritasAI - Configuration
"""

import os
from pathlib import Path

import torch

# ── Paths ──────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "temp_uploads"
REPORT_DIR = BASE_DIR / "temp_reports"
MODEL_DIR = BASE_DIR / "models"
HEATMAP_DIR = BASE_DIR / "temp_heatmaps"

UPLOAD_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)
HEATMAP_DIR.mkdir(exist_ok=True)

# ── Upload Limits ──────────────────────────────────────────────────
MAX_FILE_SIZE_MB = 100
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024
ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".mov"}
ALLOWED_EXTENSIONS = ALLOWED_IMAGE_EXTENSIONS | ALLOWED_VIDEO_EXTENSIONS

# ── Model Config ───────────────────────────────────────────────────
MODEL_NAME = os.getenv("VERITAS_MODEL", "efficientnet")  # efficientnet | xceptionnet
MODEL_WEIGHTS_PATH = MODEL_DIR / "veritas_model.pth"
IMAGE_SIZE = 224  # Input size for the deepfake model (default, unchanged)

# V2 models (ViT, FrequencyFusion) can use higher resolution
IMAGE_SIZE_V2 = int(os.getenv("VERITAS_IMAGE_SIZE_V2", "384"))

# Ensemble mode: set VERITAS_ENSEMBLE=1 to average EfficientNet + ViT predictions
ENSEMBLE_MODE = os.getenv("VERITAS_ENSEMBLE", "0") == "1"
VIT_WEIGHTS_PATH = MODEL_DIR / "veritas_vit_model.pth"

# Auto-detect GPU; override with VERITAS_GPU=1 (force GPU) or VERITAS_GPU=0 (force CPU)
_gpu_env = os.getenv("VERITAS_GPU")
if _gpu_env == "1":
    DEVICE = "cuda"
elif _gpu_env == "0":
    DEVICE = "cpu"
else:
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ── API ────────────────────────────────────────────────────────────
RATE_LIMIT = "10/minute"
CORS_ORIGINS = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "*",  # Allow mobile app connections from any origin
]

# ── Cleanup ────────────────────────────────────────────────────────
FILE_TTL_SECONDS = 3600  # delete temp files older than 1 hour

# ── Feedback System ────────────────────────────────────────────────
FEEDBACK_DIR = BASE_DIR / "feedback_data"
FEEDBACK_IMAGES_DIR = FEEDBACK_DIR / "images"
FEEDBACK_LOG_PATH = FEEDBACK_DIR / "feedback_log.jsonl"
MODEL_REGISTRY_PATH = MODEL_DIR / "model_registry.json"

FEEDBACK_DIR.mkdir(exist_ok=True)
FEEDBACK_IMAGES_DIR.mkdir(exist_ok=True)
(FEEDBACK_IMAGES_DIR / "real").mkdir(exist_ok=True)
(FEEDBACK_IMAGES_DIR / "fake").mkdir(exist_ok=True)

# ── Audio Deepfake Detection ───────────────────────────────────────
ALLOWED_AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg"}
AUDIO_MODEL_WEIGHTS_PATH = MODEL_DIR / "veritas_audio_model.pth"
AUDIO_SAMPLE_RATE = 16000
AUDIO_DURATION_SECONDS = 5
