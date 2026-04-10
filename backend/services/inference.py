"""
VeritasAI – Deepfake Inference Service
Loads a trained model and runs prediction on preprocessed frames.

Supports optional ensemble mode (VERITAS_ENSEMBLE=1) that averages
EfficientNet + ViT predictions for improved accuracy.
"""

import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torchvision import models

from config import DEVICE, ENSEMBLE_MODE, IMAGE_SIZE, IMAGE_SIZE_V2, MODEL_NAME, MODEL_WEIGHTS_PATH, VIT_WEIGHTS_PATH
from services.media_processing import preprocess_image
from services.model_manager import ModelManager

logger = logging.getLogger(__name__)

# ── Model cache ────────────────────────────────────────────────────
_model: nn.Module | None = None
_vit_model: nn.Module | None = None


def _build_efficientnet() -> nn.Module:
    model = models.efficientnet_b4(weights=None)
    # Must match the exact classifier head used during training
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(p=0.2),
        nn.Linear(256, 1),
    )
    return model


def _build_xceptionnet() -> nn.Module:
    """Simplified Xception-style model using depthwise separable convolutions."""
    class XceptionBlock(nn.Module):
        def __init__(self, in_c, out_c):
            super().__init__()
            self.dw = nn.Conv2d(in_c, in_c, 3, padding=1, groups=in_c)
            self.pw = nn.Conv2d(in_c, out_c, 1)
            self.bn = nn.BatchNorm2d(out_c)
            self.relu = nn.ReLU(inplace=True)
            self.skip = nn.Conv2d(in_c, out_c, 1) if in_c != out_c else nn.Identity()

        def forward(self, x):
            out = self.relu(self.bn(self.pw(self.dw(x))))
            return out + self.skip(x)

    class XceptionNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.entry = nn.Sequential(
                nn.Conv2d(3, 32, 3, stride=2, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
                nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            )
            self.middle = nn.Sequential(
                XceptionBlock(64, 128), nn.MaxPool2d(2),
                XceptionBlock(128, 256), nn.MaxPool2d(2),
                XceptionBlock(256, 512), nn.MaxPool2d(2),
                XceptionBlock(512, 728), nn.MaxPool2d(2),
            )
            self.exit = nn.Sequential(
                XceptionBlock(728, 1024),
                nn.AdaptiveAvgPool2d(1),
            )
            self.fc = nn.Linear(1024, 1)

        def forward(self, x):
            x = self.entry(x)
            x = self.middle(x)
            x = self.exit(x)
            x = x.flatten(1)
            return self.fc(x)

    return XceptionNet()


def _build_vit() -> nn.Module:
    """Build ViT model for inference. Requires timm."""
    try:
        import timm
    except ImportError:
        logger.error("timm not installed – cannot build ViT model")
        raise

    backbone = timm.create_model(
        "vit_base_patch16_384",
        pretrained=False,
        num_classes=0,
        img_size=IMAGE_SIZE_V2,
    )
    embed_dim = backbone.num_features

    class ViTWrapper(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = backbone
            self.classifier = nn.Sequential(
                nn.Dropout(p=0.3),
                nn.Linear(embed_dim, 256),
                nn.GELU(),
                nn.Dropout(p=0.15),
                nn.Linear(256, 1),
            )

        def forward(self, x):
            features = self.backbone(x)
            return self.classifier(features)

    return ViTWrapper()


def _load_model() -> nn.Module:
    global _model
    if _model is not None:
        return _model

    if MODEL_NAME == "xceptionnet":
        model = _build_xceptionnet()
    else:
        model = _build_efficientnet()

    # Use ModelManager to load active versioned model
    try:
        manager = ModelManager()
        weights_path = manager.get_active_model_path()
        version = manager.get_active_version()
    except Exception:
        weights_path = MODEL_WEIGHTS_PATH
        version = 1

    if Path(weights_path).exists():
        state = torch.load(str(weights_path), map_location=DEVICE)
        model.load_state_dict(state)
        logger.info("Loaded model v%d from %s", version, weights_path)
    else:
        logger.warning(
            "No trained weights found at %s — running in DEMO mode.",
            weights_path,
        )

    model.to(DEVICE)
    model.eval()
    _model = model
    return model


def _load_vit_model() -> nn.Module | None:
    """Load the ViT ensemble model (only if ensemble mode is enabled)."""
    global _vit_model
    if _vit_model is not None:
        return _vit_model

    if not ENSEMBLE_MODE:
        return None

    if not Path(VIT_WEIGHTS_PATH).exists():
        logger.warning(
            "Ensemble mode enabled but ViT weights not found at %s. "
            "Using single-model inference.",
            VIT_WEIGHTS_PATH,
        )
        return None

    try:
        model = _build_vit()
        state = torch.load(str(VIT_WEIGHTS_PATH), map_location=DEVICE)
        model.load_state_dict(state)
        model.to(DEVICE)
        model.eval()
        _vit_model = model
        logger.info("Loaded ViT ensemble model from %s", VIT_WEIGHTS_PATH)
        return model
    except Exception as e:
        logger.error("Failed to load ViT model for ensemble: %s", e)
        return None


def reload_model():
    """Force reload the model (e.g. after fine-tuning produces a new version)."""
    global _model, _vit_model
    _model = None
    _vit_model = None
    return _load_model()


# ── Secondary model (v2) for cross-checking ──────────────────────

_v2_model: nn.Module | None = None

def _load_v2_model() -> nn.Module | None:
    """Load the v2 model as a secondary cross-check detector."""
    global _v2_model
    if _v2_model is not None:
        return _v2_model

    v2_path = Path(MODEL_WEIGHTS_PATH).parent / "veritas_model_v2.pth"
    if not v2_path.exists():
        return None

    try:
        model = _build_efficientnet()
        state = torch.load(str(v2_path), map_location=DEVICE, weights_only=True)
        model.load_state_dict(state)
        model.to(DEVICE)
        model.eval()
        _v2_model = model
        logger.info("Loaded v2 secondary model from %s", v2_path)
        return model
    except Exception as e:
        logger.warning("Failed to load v2 secondary model: %s", e)
        return None


def predict_deepfake(frame_paths: list[str]) -> dict:
    """Run deepfake detection on a list of frame image paths.
    
    Uses dual-model approach:
    - Primary (v4): trained on user's data, good at face-based detection
    - Secondary (v2): trained on 60k Kaggle data, catches diverse AI content
    
    If v2 strongly disagrees with v4, the final score is adjusted upward.
    
    Returns dict with probability (0-1) and raw scores per frame.
    """
    model = _load_model()
    v2_model = _load_v2_model()
    vit_model = _load_vit_model()

    all_probs: list[float] = []
    all_v2_logits: list[float] = []
    ensemble_active = vit_model is not None
    dual_model_active = v2_model is not None

    for fpath in frame_paths:
        # Primary model prediction (224x224)
        arr = preprocess_image(fpath, size=IMAGE_SIZE)
        tensor = torch.from_numpy(arr).to(DEVICE)
        with torch.no_grad():
            logit = model(tensor)
        prob = torch.sigmoid(logit).item()

        # Secondary v2 model for cross-checking
        if dual_model_active:
            with torch.no_grad():
                v2_logit = v2_model(tensor)
            all_v2_logits.append(v2_logit.item())

        if ensemble_active:
            # ViT prediction (384x384)
            arr_v2 = preprocess_image(fpath, size=IMAGE_SIZE_V2)
            tensor_v2 = torch.from_numpy(arr_v2).to(DEVICE)
            with torch.no_grad():
                logit_v2 = vit_model(tensor_v2)
            prob_v2 = torch.sigmoid(logit_v2).item()

            # Weighted average: EfficientNet 60% + ViT 40%
            combined_prob = 0.6 * prob + 0.4 * prob_v2
            all_probs.append(combined_prob)
        else:
            all_probs.append(prob)

    avg_prob = float(np.mean(all_probs))

    # ── Dual-model cross-check ────────────────────────────────────
    # v2 was trained on 60k Kaggle images. It's biased toward "fake"
    # for WhatsApp images, BUT higher logits still correlate with 
    # more suspicious content:
    #   - Real WhatsApp photos: v2 logit ~8-13
    #   - AI-generated content: v2 logit ~14-16
    #
    # When v4 says "real" (prob < 0.3) but v2 has a very high logit,
    # it's likely AI content that v4 hasn't been trained on.
    if dual_model_active and all_v2_logits:
        avg_v2_logit = float(np.mean(all_v2_logits))
        
        # Only interfere when v4 thinks it's real
        if avg_prob < 0.3:
            # Map v2 logit to a boost factor:
            # logit < 12: no boost (normal WhatsApp range)
            # logit 12-14: mild boost
            # logit > 14: strong boost (likely AI)
            if avg_v2_logit > 14.0:
                boost = 0.6  # Strong boost for very high v2
            elif avg_v2_logit > 13.0:
                boost = 0.4
            elif avg_v2_logit > 12.0:
                boost = 0.2
            else:
                boost = 0.0
            
            if boost > 0:
                # Scale v2 logit to a probability-like value
                v2_signal = min(0.9, avg_v2_logit / 16.0)
                avg_prob = (1.0 - boost) * avg_prob + boost * v2_signal

    return {
        "probability": round(avg_prob, 4),
        "per_frame_scores": [round(p, 4) for p in all_probs],
        "label": "Likely Deepfake" if avg_prob >= 0.75 else (
            "Suspicious" if avg_prob >= 0.55 else "Likely Authentic"
        ),
        "ensemble": ensemble_active,
        "dual_model": dual_model_active,
    }

