"""
VeritasAI – Upload & Analysis Router
"""

import hashlib
import uuid
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import FileResponse

from config import (
    ALLOWED_EXTENSIONS,
    ALLOWED_IMAGE_EXTENSIONS,
    ALLOWED_VIDEO_EXTENSIONS,
    MAX_FILE_SIZE_BYTES,
    REPORT_DIR,
    UPLOAD_DIR,
)
from services.explainability import generate_heatmap
from services.face_analysis import compare_faces
from services.inference import predict_deepfake
from services.media_processing import extract_frames, preprocess_image
from services.forensic_analysis import compute_forensic_score
from services.metadata_analysis import analyze_metadata
from services.report_generator import generate_pdf_report
from services.scorecard import compute_scorecard

router = APIRouter(tags=["analysis"])


def _validate_file(file: UploadFile):
    ext = Path(file.filename or "").suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(400, f"Unsupported file type: {ext}")
    return ext


def _file_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _is_image(ext: str) -> bool:
    return ext in ALLOWED_IMAGE_EXTENSIONS


def _is_video(ext: str) -> bool:
    return ext in ALLOWED_VIDEO_EXTENSIONS


@router.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    """Simple prediction endpoint to be used with Kaggle trained models.
    Returns: {"deepfake_probability": value, "verdict": "Real / Suspicious / Likely Deepfake"}
    """
    ext = _validate_file(file)
    if not _is_image(ext):
        raise HTTPException(400, "Only images are supported for the /predict endpoint.")

    # Save temp
    analysis_id = str(uuid.uuid4())
    path = UPLOAD_DIR / f"predict_{analysis_id}{ext}"
    path.write_bytes(await file.read())

    try:
        # Run inference
        result = predict_deepfake([str(path)])
        return {
            "deepfake_probability": result["probability"],
            "verdict": result["label"]
        }
    except Exception as e:
        raise HTTPException(500, f"Prediction failed: {str(e)}")


@router.post("/analyze")
async def analyze_media(
    media: UploadFile = File(..., description="Suspicious image or video"),
    reference: UploadFile | None = File(None, description="Optional reference image"),
):
    """Run full deepfake analysis pipeline on uploaded media."""
    ext = _validate_file(media)
    analysis_id = str(uuid.uuid4())

    # ── Read & save file ───────────────────────────────────────────
    media_bytes = await media.read()
    if len(media_bytes) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(413, "File exceeds 100 MB limit")

    media_path = UPLOAD_DIR / f"{analysis_id}{ext}"
    media_path.write_bytes(media_bytes)
    file_hash = _file_hash(media_bytes)

    # ── Optional reference ─────────────────────────────────────────
    ref_path = None
    if reference:
        ref_ext = _validate_file(reference)
        ref_bytes = await reference.read()
        ref_path = UPLOAD_DIR / f"{analysis_id}_ref{ref_ext}"
        ref_path.write_bytes(ref_bytes)

    try:
        # ── Frame extraction (video) or direct image ───────────────
        if _is_video(ext):
            frames = extract_frames(str(media_path), max_frames=16)
        else:
            frames = [str(media_path)]

        if not frames:
            raise HTTPException(422, "Could not extract any frames from media")

        # ── Deepfake inference ─────────────────────────────────────
        deepfake_result = predict_deepfake(frames)

        # ── Explainability heatmap ─────────────────────────────────
        heatmap_info = generate_heatmap(frames[0], analysis_id)

        # ── Face consistency ───────────────────────────────────────
        face_consistency = compare_faces(
            frames[0], str(ref_path) if ref_path else None
        )

        # ── Metadata analysis ─────────────────────────────────────
        metadata_result = analyze_metadata(str(media_path))

        # ── Forensic analysis (ELA + frequency) ───────────────────
        forensic_result = compute_forensic_score(frames[0])

        # ── Scorecard ──────────────────────────────────────────────
        scorecard = compute_scorecard(
            deepfake_prob=deepfake_result["probability"],
            face_consistency=face_consistency["score"],
            metadata_integrity=metadata_result["integrity_score"],
            forensic_score=forensic_result["forensic_score"],
        )

        # ── Build response ─────────────────────────────────────────
        result = {
            "analysis_id": analysis_id,
            "file_hash": file_hash,
            "deepfake": deepfake_result,
            "heatmap": heatmap_info,
            "face_consistency": face_consistency,
            "metadata": metadata_result,
            "forensic": forensic_result,
            "scorecard": scorecard,
        }

        # ── Generate PDF ───────────────────────────────────────────
        generate_pdf_report(result)

        return result

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Analysis failed: {str(e)}")


@router.get("/report/{analysis_id}")
async def download_report(analysis_id: str):
    """Download the generated PDF evidence report."""
    report_path = REPORT_DIR / f"{analysis_id}.pdf"
    if not report_path.exists():
        raise HTTPException(404, "Report not found")
    return FileResponse(
        path=str(report_path),
        media_type="application/pdf",
        filename=f"VeritasAI_Report_{analysis_id}.pdf",
    )
