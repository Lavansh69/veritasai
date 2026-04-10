"""
VeritasAI – Audio Deepfake Inference Service
Loads a trained Mel-spectrogram CNN and runs prediction on audio files.
Completely independent from the image/video inference pipeline.
"""

import logging
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from config import (
    AUDIO_DURATION_SECONDS,
    AUDIO_MODEL_WEIGHTS_PATH,
    AUDIO_SAMPLE_RATE,
    DEVICE,
)

logger = logging.getLogger(__name__)

# ── Model cache ────────────────────────────────────────────────────
_audio_model: nn.Module | None = None
_demo_mode: bool = False


# ── Audio Classifier (must match training architecture) ───────────
class _AudioClassifier(nn.Module):
    """Mel-spectrogram CNN for audio deepfake detection."""

    def __init__(self, n_mels: int = 128):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # Block 4
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.flatten(1)
        return self.classifier(x)


def _audio_to_mel_spectrogram(audio_path: str) -> np.ndarray:
    """Load audio file and convert to a Mel spectrogram tensor.

    Returns ndarray of shape (1, 1, n_mels, time_frames) ready for the model.
    """
    import librosa

    y, sr = librosa.load(audio_path, sr=AUDIO_SAMPLE_RATE, mono=True)

    # Pad or truncate to fixed duration
    target_len = AUDIO_SAMPLE_RATE * AUDIO_DURATION_SECONDS
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)), mode="constant")
    else:
        y = y[:target_len]

    mel = librosa.feature.melspectrogram(
        y=y, sr=AUDIO_SAMPLE_RATE, n_mels=128, fmax=8000
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)

    # Normalize to [0, 1]
    mel_db = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-8)

    # Shape: (1, 1, n_mels, time_frames)
    return mel_db[np.newaxis, np.newaxis, :, :].astype(np.float32)


def _load_audio_model() -> nn.Module:
    """Load (or return cached) audio deepfake model."""
    global _audio_model, _demo_mode

    if _audio_model is not None:
        return _audio_model

    model = _AudioClassifier()

    if Path(AUDIO_MODEL_WEIGHTS_PATH).exists():
        state = torch.load(str(AUDIO_MODEL_WEIGHTS_PATH), map_location=DEVICE)
        model.load_state_dict(state)
        logger.info("Loaded audio model from %s", AUDIO_MODEL_WEIGHTS_PATH)
        _demo_mode = False
    else:
        logger.warning(
            "No audio model weights found at %s — running in DEMO mode.",
            AUDIO_MODEL_WEIGHTS_PATH,
        )
        _demo_mode = True

    model.to(DEVICE)
    model.eval()
    _audio_model = model
    return model


def predict_audio_deepfake(audio_path: str) -> dict:
    """Run audio deepfake detection on a single audio file.

    Returns dict with probability (0-1), label, and demo flag.
    """
    model = _load_audio_model()

    if _demo_mode:
        # In demo mode without trained weights, return a plausible random score
        prob = round(random.uniform(0.2, 0.8), 4)
    else:
        mel_tensor = _audio_to_mel_spectrogram(audio_path)
        tensor = torch.from_numpy(mel_tensor).to(DEVICE)
        with torch.no_grad():
            logit = model(tensor)
        prob = round(torch.sigmoid(logit).item(), 4)

    label = (
        "Likely Deepfake Audio"
        if prob >= 0.85
        else ("Suspicious Audio" if prob >= 0.65 else "Likely Authentic Audio")
    )

    return {
        "probability": prob,
        "label": label,
        "demo_mode": _demo_mode,
    }
