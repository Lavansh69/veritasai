"""
VeritasAI – Preprocessing Utilities for Training Pipeline
Frame extraction, face detection, alignment, and normalization.
"""

import os
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

try:
    import face_recognition

    _FACE_LIB = True
except ImportError:
    _FACE_LIB = False


def extract_frames_from_video(
    video_path: str, output_dir: str, max_frames: int = 32
) -> list[str]:
    """Extract evenly-spaced frames from a video file.
    
    Args:
        video_path: Path to video file.
        output_dir: Directory to save extracted frames.
        max_frames: Maximum number of frames to extract.
    
    Returns:
        List of paths to saved frame images.
    """
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[WARN] Cannot open video: {video_path}")
        return []

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return []

    indices = np.linspace(0, total - 1, min(max_frames, total), dtype=int)
    paths: list[str] = []
    video_name = Path(video_path).stem

    for i, idx in enumerate(indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret:
            continue
        out_path = os.path.join(output_dir, f"{video_name}_frame_{i:04d}.jpg")
        cv2.imwrite(out_path, frame)
        paths.append(out_path)

    cap.release()
    return paths


def detect_and_crop_face(image_path: str, output_path: str, margin: float = 0.3) -> bool:
    """Detect face in image, crop with margin, and save.
    
    Returns True if a face was found and saved.
    """
    img = cv2.imread(image_path)
    if img is None:
        return False

    if _FACE_LIB:
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        locations = face_recognition.face_locations(rgb, model="hog")
        if not locations:
            # Save original if no face found
            cv2.imwrite(output_path, img)
            return False

        top, right, bottom, left = locations[0]
    else:
        # Fallback: OpenCV Haar cascade
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        faces = cascade.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))
        if len(faces) == 0:
            cv2.imwrite(output_path, img)
            return False
        x, y, w, h = faces[0]
        top, right, bottom, left = y, x + w, y + h, x

    # Add margin
    h_img, w_img = img.shape[:2]
    height = bottom - top
    width = right - left
    top = max(0, int(top - height * margin))
    bottom = min(h_img, int(bottom + height * margin))
    left = max(0, int(left - width * margin))
    right = min(w_img, int(right + width * margin))

    crop = img[top:bottom, left:right]
    cv2.imwrite(output_path, crop)
    return True


def preprocess_dataset(
    input_dir: str,
    output_dir: str,
    image_size: int = 224,
    extract_faces: bool = True,
):
    """Preprocess an entire dataset directory.
    
    Expects input_dir to have 'real/' and 'fake/' subdirectories
    with either images or videos.
    
    Args:
        input_dir: Root dataset directory.
        output_dir: Output directory for preprocessed images.
        image_size: Target image size.
        extract_faces: Whether to detect and crop faces.
    """
    for label in ("real", "fake"):
        src = Path(input_dir) / label
        dst = Path(output_dir) / label
        dst.mkdir(parents=True, exist_ok=True)

        if not src.exists():
            print(f"[WARN] Directory not found: {src}")
            continue

        for f in src.rglob("*"):
            if f.suffix.lower() in (".mp4", ".mov", ".avi"):
                # Extract frames from video
                frame_dir = dst / f.stem
                frame_dir.mkdir(exist_ok=True)
                frames = extract_frames_from_video(str(f), str(frame_dir))
                for frame_path in frames:
                    if extract_faces:
                        detect_and_crop_face(frame_path, frame_path)
                    _resize_image(frame_path, image_size)

            elif f.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp"):
                out_path = str(dst / f.name)
                if extract_faces:
                    detect_and_crop_face(str(f), out_path)
                else:
                    img = cv2.imread(str(f))
                    if img is not None:
                        cv2.imwrite(out_path, img)
                _resize_image(out_path, image_size)

    print(f"[OK] Preprocessing complete. Output: {output_dir}")


def _resize_image(path: str, size: int):
    """Resize image to square dimensions."""
    try:
        img = Image.open(path).convert("RGB")
        img = img.resize((size, size), Image.LANCZOS)
        img.save(path)
    except Exception as e:
        print(f"[WARN] Could not resize {path}: {e}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Preprocess deepfake dataset")
    parser.add_argument("--input", required=True, help="Input dataset directory")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--size", type=int, default=224, help="Image size")
    parser.add_argument("--no-faces", action="store_true", help="Skip face detection")
    args = parser.parse_args()

    preprocess_dataset(args.input, args.output, args.size, not args.no_faces)
