"""
VeritasAI – Live Inference Service
Real-time deepfake detection pipeline optimised for WebSocket streaming.

Key design choices:
  • OpenCV Haar Cascade for face detection (fast, no extra dependencies)
  • Multi-face support: selects the largest / highest-confidence face
  • All processing in-memory (numpy / PIL from raw bytes) — zero disk I/O
  • Mixed-precision inference via torch.autocast when CUDA is available
  • Rolling-average smoothing over a sliding window of recent predictions
  • Grad-CAM is optional and throttled (caller controls frequency)
"""

import base64
import io
import logging
import time
from collections import deque

import cv2
import numpy as np
import torch
from PIL import Image

from config import DEVICE, IMAGE_SIZE

logger = logging.getLogger(__name__)

# ── OpenCV Haar Cascade face detector (ships with cv2) ─────────────
_face_cascade = None


def _get_face_cascade():
    """Lazy-init OpenCV Haar Cascade face detector."""
    global _face_cascade
    if _face_cascade is None:
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        _face_cascade = cv2.CascadeClassifier(cascade_path)
        if _face_cascade.empty():
            logger.warning("Haar cascade file not found at %s", cascade_path)
    return _face_cascade


# ── Singleton LiveDetector ─────────────────────────────────────────
_instance: "LiveDetector | None" = None


def get_live_detector() -> "LiveDetector":
    """Return (or create) the singleton LiveDetector instance."""
    global _instance
    if _instance is None:
        _instance = LiveDetector()
    return _instance


class LiveDetector:
    """High-throughput, in-memory deepfake detector for live video frames."""

    # ImageNet normalisation constants
    _MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    _STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __init__(self, window_size: int = 8):
        # Import and cache the shared model (singleton from inference.py)
        from services.inference import _load_model
        self._model = _load_model()
        self._model.eval()

        # Rolling-average window
        self._window_size = window_size
        self._history: deque[float] = deque(maxlen=window_size)

        # Device flag for mixed-precision
        self._use_cuda = DEVICE == "cuda" and torch.cuda.is_available()

        logger.info(
            "LiveDetector ready — device=%s, window=%d, mixed_precision=%s",
            DEVICE, window_size, self._use_cuda,
        )

    # ── Face Detection ─────────────────────────────────────────────

    def detect_faces(self, frame_bgr: np.ndarray) -> list[dict]:
        """Detect faces via OpenCV Haar Cascade. Returns list of bbox dicts sorted by area
        (largest first). Each dict: {top, right, bottom, left, confidence}."""
        cascade = _get_face_cascade()
        if cascade is None or cascade.empty():
            return []

        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        detections = cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE,
        )

        if len(detections) == 0:
            return []

        faces: list[dict] = []
        for (x, y, w, h) in detections:
            faces.append({
                "top": int(y),
                "right": int(x + w),
                "bottom": int(y + h),
                "left": int(x),
                "confidence": 1.0,  # Haar doesn't give confidence scores
            })

        # Sort by area descending — largest face first
        faces.sort(key=lambda f: (f["bottom"] - f["top"]) * (f["right"] - f["left"]), reverse=True)
        return faces

    # ── Preprocessing ──────────────────────────────────────────────

    def _crop_and_preprocess(
        self, frame_bgr: np.ndarray, bbox: dict | None
    ) -> torch.Tensor:
        """Crop face (if bbox), resize to IMAGE_SIZE, normalise, return tensor (1,3,H,W)."""
        if bbox is not None:
            # Add 20 % padding around the face
            h, w, _ = frame_bgr.shape
            pad_y = int((bbox["bottom"] - bbox["top"]) * 0.2)
            pad_x = int((bbox["right"] - bbox["left"]) * 0.2)
            y1 = max(0, bbox["top"] - pad_y)
            y2 = min(h, bbox["bottom"] + pad_y)
            x1 = max(0, bbox["left"] - pad_x)
            x2 = min(w, bbox["right"] + pad_x)
            crop = frame_bgr[y1:y2, x1:x2]
        else:
            crop = frame_bgr

        # Resize
        resized = cv2.resize(crop, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_LINEAR)
        # BGR → RGB → float32 → normalise
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        normed = (rgb - self._MEAN) / self._STD
        # HWC → CHW, add batch dim
        tensor = torch.from_numpy(normed.transpose(2, 0, 1)).unsqueeze(0).to(DEVICE)
        return tensor

    # ── Grad-CAM (lightweight, in-memory) ──────────────────────────

    def _generate_heatmap(
        self, tensor: torch.Tensor, frame_bgr: np.ndarray, bbox: dict | None
    ) -> str | None:
        """Run Grad-CAM on the given input tensor. Returns base64-encoded JPEG or None."""
        try:
            model = self._model

            # Find last conv layer (EfficientNet)
            target_layer = None
            if hasattr(model, "features"):
                target_layer = model.features[-1]
            else:
                for module in model.modules():
                    if isinstance(module, torch.nn.Conv2d):
                        target_layer = module

            if target_layer is None:
                return None

            activations: dict = {}
            gradients: dict = {}

            def fwd_hook(m, inp, out):
                activations["v"] = out.detach()

            def bwd_hook(m, gi, go):
                gradients["v"] = go[0].detach()

            fh = target_layer.register_forward_hook(fwd_hook)
            bh = target_layer.register_full_backward_hook(bwd_hook)

            inp = tensor.clone().requires_grad_(True)
            out = model(inp)
            model.zero_grad()
            out.backward()

            fh.remove()
            bh.remove()

            grads = gradients.get("v")
            acts = activations.get("v")
            if grads is None or acts is None:
                return None

            weights = grads.mean(dim=[2, 3], keepdim=True)
            cam = torch.relu((weights * acts).sum(dim=1, keepdim=True))
            cam = cam.squeeze().cpu().numpy()

            if cam.max() > 0:
                cam = (cam - cam.min()) / (cam.max() - cam.min())
            cam_uint8 = np.uint8(cam * 255)
            cam_resized = cv2.resize(cam_uint8, (IMAGE_SIZE, IMAGE_SIZE))

            # Create overlay on the cropped region
            if bbox is not None:
                h, w, _ = frame_bgr.shape
                pad_y = int((bbox["bottom"] - bbox["top"]) * 0.2)
                pad_x = int((bbox["right"] - bbox["left"]) * 0.2)
                y1, y2 = max(0, bbox["top"] - pad_y), min(h, bbox["bottom"] + pad_y)
                x1, x2 = max(0, bbox["left"] - pad_x), min(w, bbox["right"] + pad_x)
                crop = frame_bgr[y1:y2, x1:x2]
            else:
                crop = frame_bgr

            crop_resized = cv2.resize(crop, (IMAGE_SIZE, IMAGE_SIZE))
            heatmap_color = cv2.applyColorMap(cam_resized, cv2.COLORMAP_JET)
            overlay = cv2.addWeighted(crop_resized, 0.6, heatmap_color, 0.4, 0)

            # Encode to JPEG in memory → base64
            _, buf = cv2.imencode(".jpg", overlay, [cv2.IMWRITE_JPEG_QUALITY, 75])
            return base64.b64encode(buf.tobytes()).decode("ascii")

        except Exception as e:
            logger.warning("Heatmap generation failed: %s", e)
            return None

    # ── Main Prediction ────────────────────────────────────────────

    def predict(
        self,
        frame_bytes: bytes,
        generate_heatmap: bool = False,
    ) -> dict | None:
        """Full live prediction pipeline with multi-face support.

        Args:
            frame_bytes: Raw JPEG bytes (binary WebSocket message).
            generate_heatmap: Whether to produce a Grad-CAM overlay.

        Returns:
            Result dict or None if the frame is invalid.
        """
        t0 = time.perf_counter()

        # ── Decode ─────────────────────────────────────────────────
        arr = np.frombuffer(frame_bytes, dtype=np.uint8)
        frame_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame_bgr is None:
            return None  # invalid / corrupt frame — skip

        # ── Face detection ─────────────────────────────────────────
        faces = self.detect_faces(frame_bgr)

        # ── Per-face inference ─────────────────────────────────────
        face_results: list[dict] = []

        if faces:
            for face_bbox in faces:
                tensor = self._crop_and_preprocess(frame_bgr, face_bbox)
                with torch.no_grad():
                    if self._use_cuda:
                        with torch.autocast("cuda"):
                            logit = self._model(tensor)
                    else:
                        logit = self._model(tensor)
                prob = torch.sigmoid(logit).item()

                # Per-face label
                if prob >= 0.75:
                    face_label = "Likely Deepfake"
                elif prob >= 0.55:
                    face_label = "Suspicious"
                else:
                    face_label = "Likely Authentic"

                face_results.append({
                    "bbox": {
                        "top": face_bbox["top"],
                        "right": face_bbox["right"],
                        "bottom": face_bbox["bottom"],
                        "left": face_bbox["left"],
                    },
                    "confidence": round(prob, 4),
                    "label": face_label,
                })
        else:
            # No face found — run inference on the full frame
            tensor = self._crop_and_preprocess(frame_bgr, None)
            with torch.no_grad():
                if self._use_cuda:
                    with torch.autocast("cuda"):
                        logit = self._model(tensor)
                else:
                    logit = self._model(tensor)
            prob = torch.sigmoid(logit).item()

        # ── Overall result (primary = largest face or full frame) ──
        primary_prob = face_results[0]["confidence"] if face_results else prob

        # ── Rolling average ────────────────────────────────────────
        self._history.append(primary_prob)
        smoothed = float(np.mean(self._history))

        # ── Overall label (based on smoothed primary) ──────────────
        if smoothed >= 0.75:
            label = "Likely Deepfake"
        elif smoothed >= 0.55:
            label = "Suspicious"
        else:
            label = "Likely Authentic"

        # ── Optional heatmap (largest face only for perf) ──────────
        heatmap_b64: str | None = None
        if generate_heatmap and faces:
            tensor_primary = self._crop_and_preprocess(frame_bgr, faces[0])
            heatmap_b64 = self._generate_heatmap(tensor_primary, frame_bgr, faces[0])

        latency_ms = (time.perf_counter() - t0) * 1000

        # Legacy single-bbox for backward compat
        bbox_response = face_results[0]["bbox"] if face_results else None

        return {
            "label": label,
            "confidence": round(smoothed, 4),
            "raw_confidence": round(primary_prob, 4),
            "bbox": bbox_response,
            "faces": face_results,
            "heatmap": heatmap_b64,
            "latency_ms": round(latency_ms, 1),
            "faces_detected": len(faces),
        }

    def reset(self):
        """Clear history (e.g. on new session)."""
        self._history.clear()
