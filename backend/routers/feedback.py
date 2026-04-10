"""
VeritasAI – Feedback API Router
Endpoints for submitting user feedback and viewing statistics.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from services.feedback import get_feedback_stats, submit_feedback
from services.model_manager import ModelManager

router = APIRouter(tags=["feedback"])


class FeedbackRequest(BaseModel):
    analysis_id: str
    is_correct: bool
    corrected_label: str | None = None  # "real" or "fake"
    prediction: str | None = None
    confidence: float | None = None


@router.post("/feedback")
async def post_feedback(req: FeedbackRequest):
    """Submit user feedback for an analysis result."""
    if not req.is_correct and req.corrected_label not in ("real", "fake"):
        raise HTTPException(
            400,
            "corrected_label must be 'real' or 'fake' when marking prediction as incorrect.",
        )

    result = submit_feedback(
        analysis_id=req.analysis_id,
        is_correct=req.is_correct,
        corrected_label=req.corrected_label,
        prediction=req.prediction,
        confidence=req.confidence,
    )
    return result


@router.get("/feedback/stats")
async def feedback_stats():
    """Return feedback collection statistics."""
    stats = get_feedback_stats()
    manager = ModelManager()
    stats["model_versions"] = manager.list_versions()
    stats["active_version"] = manager.get_active_version()
    return stats
