"""
VeritasAI – Face Analysis Service
Face detection & embedding comparison between suspicious and reference media.
"""

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# Use try/except because face_recognition requires dlib which may not be installed
try:
    import face_recognition

    _FACE_LIB_AVAILABLE = True
except ImportError:
    _FACE_LIB_AVAILABLE = False
    logger.warning(
        "face_recognition library not available. "
        "Face consistency checks will use fallback scoring."
    )


def detect_faces(image_path: str) -> list[dict]:
    """Detect faces in an image and return locations + encodings."""
    if not _FACE_LIB_AVAILABLE:
        return []

    try:
        image = face_recognition.load_image_file(image_path)
        locations = face_recognition.face_locations(image, model="hog")
        encodings = face_recognition.face_encodings(image, locations)

        faces = []
        for loc, enc in zip(locations, encodings):
            faces.append({
                "location": {
                    "top": loc[0], "right": loc[1],
                    "bottom": loc[2], "left": loc[3],
                },
                "encoding": enc.tolist(),
            })
        return faces
    except Exception as e:
        logger.error("Face detection failed: %s", e)
        return []


def compare_faces(
    suspicious_path: str, reference_path: str | None
) -> dict:
    """Compare face embeddings between suspicious and reference images.
    
    Returns face consistency score (0-100) and details.
    """
    sus_faces = detect_faces(suspicious_path)

    if not sus_faces:
        return {
            "score": 50.0,
            "faces_detected": 0,
            "match": None,
            "detail": "No faces detected in the suspicious media.",
        }

    if reference_path is None or not Path(reference_path).exists():
        return {
            "score": 50.0,
            "faces_detected": len(sus_faces),
            "match": None,
            "detail": "No reference image provided for identity comparison.",
        }

    ref_faces = detect_faces(reference_path)
    if not ref_faces:
        return {
            "score": 50.0,
            "faces_detected": len(sus_faces),
            "match": None,
            "detail": "No faces detected in the reference image.",
        }

    # Compare closest pair
    if _FACE_LIB_AVAILABLE:
        sus_enc = np.array(sus_faces[0]["encoding"])
        ref_enc = np.array(ref_faces[0]["encoding"])
        distance = float(np.linalg.norm(sus_enc - ref_enc))
        # distance < 0.6 is typically a match
        similarity = max(0.0, 1.0 - distance) * 100
        is_match = distance < 0.6
    else:
        similarity = 50.0
        is_match = None

    return {
        "score": round(similarity, 2),
        "faces_detected": len(sus_faces),
        "match": is_match,
        "detail": (
            f"Face similarity score: {similarity:.1f}%. "
            + ("Faces appear to match." if is_match else "Faces may not match the reference.")
        ),
    }
