"""
VeritasAI – Feedback Storage & Logging Service
Stores user feedback (correct/incorrect predictions) for model improvement.
"""

import json
import logging
import shutil
import uuid
from datetime import datetime
from pathlib import Path

from config import (
    FEEDBACK_DIR,
    FEEDBACK_IMAGES_DIR,
    FEEDBACK_LOG_PATH,
    UPLOAD_DIR,
)

logger = logging.getLogger(__name__)


def submit_feedback(
    analysis_id: str,
    is_correct: bool,
    corrected_label: str | None = None,
    prediction: str | None = None,
    confidence: float | None = None,
) -> dict:
    """Process user feedback for an analysis.

    If the prediction was wrong, copies the uploaded image into the
    feedback dataset under the corrected label folder.

    Args:
        analysis_id: ID of the analysis session.
        is_correct: True if user confirms the prediction was correct.
        corrected_label: 'real' or 'fake' — required when is_correct=False.
        prediction: Original model prediction label.
        confidence: Original model confidence (0-1).

    Returns:
        dict with status and saved path (if applicable).
    """
    log_entry = {
        "id": str(uuid.uuid4()),
        "analysis_id": analysis_id,
        "timestamp": datetime.now().isoformat(),
        "is_correct": is_correct,
        "original_prediction": prediction,
        "confidence": confidence,
        "corrected_label": corrected_label,
        "saved_image": None,
    }

    saved_path = None

    if not is_correct and corrected_label in ("real", "fake"):
        # Find the original uploaded file for this analysis
        source = _find_upload(analysis_id)
        if source:
            dest_dir = FEEDBACK_IMAGES_DIR / corrected_label
            dest_dir.mkdir(parents=True, exist_ok=True)
            ext = source.suffix
            dest_name = f"{analysis_id}_{uuid.uuid4().hex[:8]}{ext}"
            dest = dest_dir / dest_name
            shutil.copy2(str(source), str(dest))
            saved_path = str(dest)
            log_entry["saved_image"] = saved_path
            logger.info(
                "Saved feedback image: %s → %s (label: %s)",
                source.name, dest_name, corrected_label,
            )
        else:
            logger.warning(
                "Could not find uploaded file for analysis %s", analysis_id
            )

    # Append to log
    _append_log(log_entry)

    return {
        "status": "saved",
        "feedback_id": log_entry["id"],
        "saved_image": saved_path,
    }


def get_feedback_stats() -> dict:
    """Return feedback statistics."""
    if not FEEDBACK_LOG_PATH.exists():
        return {"total": 0, "correct": 0, "incorrect": 0, "feedback_images": 0}

    total = 0
    correct = 0
    incorrect = 0

    with open(FEEDBACK_LOG_PATH, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                total += 1
                if entry.get("is_correct"):
                    correct += 1
                else:
                    incorrect += 1
            except json.JSONDecodeError:
                continue

    # Count feedback images
    real_count = len(list((FEEDBACK_IMAGES_DIR / "real").glob("*")))
    fake_count = len(list((FEEDBACK_IMAGES_DIR / "fake").glob("*")))

    return {
        "total": total,
        "correct": correct,
        "incorrect": incorrect,
        "feedback_images": real_count + fake_count,
        "feedback_real": real_count,
        "feedback_fake": fake_count,
    }


def _find_upload(analysis_id: str) -> Path | None:
    """Find the uploaded file for a given analysis ID."""
    for ext in (".jpg", ".jpeg", ".png", ".mp4", ".mov"):
        # Check direct analysis uploads
        candidate = UPLOAD_DIR / f"{analysis_id}{ext}"
        if candidate.exists():
            return candidate
        # Check predict uploads
        candidate = UPLOAD_DIR / f"predict_{analysis_id}{ext}"
        if candidate.exists():
            return candidate
    return None


def _append_log(entry: dict):
    """Append a JSON entry to the feedback log."""
    FEEDBACK_DIR.mkdir(parents=True, exist_ok=True)
    with open(FEEDBACK_LOG_PATH, "a") as f:
        f.write(json.dumps(entry) + "\n")
