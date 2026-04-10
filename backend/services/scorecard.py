"""
VeritasAI – Authenticity Scorecard System
Multi-factor scoring + verdict computation.
Uses real forensic analysis (ELA + frequency) instead of placeholder.
"""

import logging

logger = logging.getLogger(__name__)


def compute_scorecard(
    deepfake_prob: float,
    face_consistency: float,
    metadata_integrity: float,
    forensic_score: float | None = None,
) -> dict:
    """Compute multi-factor authenticity scorecard.
    
    Args:
        deepfake_prob: 0–1 probability of being deepfake
        face_consistency: 0–100 face match score
        metadata_integrity: 0–100 metadata integrity score
        forensic_score: 0–100 forensic suspicion score (ELA + frequency)
    
    Returns:
        dict with individual scores, overall score, and verdict.
    """
    # Use forensic score if provided, otherwise neutral
    artifact_score = forensic_score if forensic_score is not None else 50.0

    # Convert deepfake probability to an "authenticity" direction (lower = worse)
    deepfake_auth = (1.0 - deepfake_prob) * 100

    # Convert forensic suspicion to authenticity (lower suspicion = more authentic)
    forensic_auth = 100.0 - artifact_score

    # When face_consistency is 50 (default/no reference), treat as neutral (75)
    # to avoid penalising images that simply have no reference for comparison.
    effective_face = face_consistency if face_consistency != 50.0 else 75.0

    # Adaptive weighting: when the model is confident, trust it more
    if deepfake_prob > 0.5 or deepfake_prob < 0.1:
        # Model is confident — give it more weight
        weights = {
            "deepfake": 0.60,
            "face_consistency": 0.10,
            "metadata": 0.10,
            "forensic": 0.20,
        }
    else:
        # Model is uncertain — balance with forensic signals
        weights = {
            "deepfake": 0.40,
            "face_consistency": 0.15,
            "metadata": 0.15,
            "forensic": 0.30,
        }

    overall = (
        weights["deepfake"] * deepfake_auth
        + weights["face_consistency"] * effective_face
        + weights["metadata"] * metadata_integrity
        + weights["forensic"] * forensic_auth
    )
    overall = round(max(0, min(100, overall)), 1)

    # Verdict
    if overall >= 60:
        verdict = "Authentic"
        verdict_color = "green"
    elif overall >= 35:
        verdict = "Suspicious"
        verdict_color = "orange"
    else:
        verdict = "Likely Deepfake"
        verdict_color = "red"

    return {
        "scores": {
            "deepfake_probability": round(deepfake_prob * 100, 1),
            "face_consistency": round(face_consistency, 1),
            "metadata_integrity": round(metadata_integrity, 1),
            "artifact_detection": round(artifact_score, 1),
        },
        "overall_score": overall,
        "verdict": verdict,
        "verdict_color": verdict_color,
    }

