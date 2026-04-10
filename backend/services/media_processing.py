"""
VeritasAI – Media Processing Utilities
Frame extraction, image validation, resizing, normalization.
"""

import cv2
import numpy as np
from PIL import Image

from config import IMAGE_SIZE, UPLOAD_DIR


def extract_frames(video_path: str, max_frames: int = 16) -> list[str]:
    """Extract evenly spaced frames from a video file.
    
    Returns list of file paths to extracted frame images.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        return []

    indices = np.linspace(0, total_frames - 1, min(max_frames, total_frames), dtype=int)
    frame_paths: list[str] = []

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret:
            continue
        frame_path = str(UPLOAD_DIR / f"frame_{idx}.jpg")
        cv2.imwrite(frame_path, frame)
        frame_paths.append(frame_path)

    cap.release()
    return frame_paths


def preprocess_image(image_path: str, size: int = IMAGE_SIZE) -> np.ndarray:
    """Load and preprocess an image for model inference.
    
    Returns (1, 3, H, W) float32 numpy array normalised to ImageNet stats.
    """
    img = Image.open(image_path).convert("RGB")
    img = img.resize((size, size), Image.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0

    # ImageNet normalization
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    arr = (arr - mean) / std

    # HWC → CHW, add batch dim
    arr = np.transpose(arr, (2, 0, 1))
    arr = np.expand_dims(arr, axis=0)
    return arr
