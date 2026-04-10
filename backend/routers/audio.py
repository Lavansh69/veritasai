"""
VeritasAI – Audio Deepfake Analysis Router
Completely separate from the image/video upload router.
"""

import uuid
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile

from config import ALLOWED_AUDIO_EXTENSIONS, MAX_FILE_SIZE_BYTES, UPLOAD_DIR
from services.audio_inference import predict_audio_deepfake

router = APIRouter(prefix="/audio", tags=["audio-analysis"])


def _validate_audio(file: UploadFile) -> str:
    ext = Path(file.filename or "").suffix.lower()
    if ext not in ALLOWED_AUDIO_EXTENSIONS:
        raise HTTPException(
            400,
            f"Unsupported audio format: {ext}. "
            f"Allowed: {', '.join(sorted(ALLOWED_AUDIO_EXTENSIONS))}",
        )
    return ext


@router.post("/analyze")
async def analyze_audio(audio: UploadFile = File(..., description="Suspicious audio file")):
    """Run deepfake detection on an uploaded audio file."""
    ext = _validate_audio(audio)
    analysis_id = str(uuid.uuid4())

    # Read & save file
    audio_bytes = await audio.read()
    if len(audio_bytes) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(413, "Audio file exceeds 100 MB limit")

    audio_path = UPLOAD_DIR / f"audio_{analysis_id}{ext}"
    audio_path.write_bytes(audio_bytes)

    try:
        result = predict_audio_deepfake(str(audio_path))
        return {
            "analysis_id": analysis_id,
            "deepfake_probability": result["probability"],
            "verdict": result["label"],
            "demo_mode": result["demo_mode"],
        }
    except Exception as e:
        raise HTTPException(500, f"Audio analysis failed: {str(e)}")
    finally:
        # Clean up the temp file immediately after analysis
        audio_path.unlink(missing_ok=True)
