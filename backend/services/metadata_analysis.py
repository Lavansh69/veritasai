"""
VeritasAI – Metadata Forensic Analysis
Extracts EXIF / file metadata and flags signs of manipulation.
"""

import os
import logging
from datetime import datetime
from pathlib import Path

from PIL import Image
from PIL.ExifTags import TAGS

logger = logging.getLogger(__name__)

# Software names that indicate editing
KNOWN_EDITORS = {
    "photoshop", "gimp", "lightroom", "aftereffects", "premiere",
    "davinci", "capcut", "faceapp", "faceswap", "deepfacelab",
    "canva", "snapseed", "pixlr", "affinity",
}


def analyze_metadata(file_path: str) -> dict:
    """Analyze file metadata for signs of manipulation.
    
    Returns dict with extracted metadata, warnings, and integrity score.
    """
    path = Path(file_path)
    warnings: list[str] = []
    meta: dict = {}
    integrity_score = 100.0  # Start at 100, deduct for red flags

    # ── Basic file info ────────────────────────────────────────────
    stat = path.stat()
    meta["filename"] = path.name
    meta["file_size_bytes"] = stat.st_size
    meta["file_modified"] = datetime.fromtimestamp(stat.st_mtime).isoformat()

    # ── EXIF extraction (images) ───────────────────────────────────
    exif_data = _extract_exif(file_path)
    if exif_data:
        meta["exif"] = exif_data
    else:
        meta["exif"] = None
        warnings.append("No EXIF metadata found — common for images shared via WhatsApp, Telegram, or social media.")
        integrity_score -= 5

    # ── Check for editing software ─────────────────────────────────
    if exif_data:
        software = exif_data.get("Software", "").lower()
        if software:
            meta["editing_software"] = exif_data["Software"]
            if any(editor in software for editor in KNOWN_EDITORS):
                warnings.append(
                    f"Metadata indicates post-processing using editing software: "
                    f"{exif_data['Software']}"
                )
                integrity_score -= 25

        # Check device model
        make = exif_data.get("Make", "")
        model = exif_data.get("Model", "")
        if make or model:
            meta["device"] = f"{make} {model}".strip()
        else:
            warnings.append("No camera/device information in metadata.")
            integrity_score -= 5

        # Check datetime
        dt = exif_data.get("DateTimeOriginal") or exif_data.get("DateTime")
        if dt:
            meta["creation_date"] = dt
        else:
            warnings.append("No original creation timestamp in metadata.")
            integrity_score -= 5
    else:
        integrity_score -= 0  # No double penalty for missing EXIF

    # ── Compression analysis ───────────────────────────────────────
    ext = path.suffix.lower()
    if ext in (".jpg", ".jpeg"):
        compression_info = _analyze_jpeg_compression(file_path)
        meta["compression"] = compression_info
        if compression_info.get("re_compressed"):
            warnings.append("Image appears to have been re-compressed (possible manipulation).")
            integrity_score -= 15

    integrity_score = max(0.0, integrity_score)

    return {
        "metadata": meta,
        "warnings": warnings,
        "integrity_score": round(integrity_score, 2),
        "summary": (
            "Metadata analysis complete. "
            + (f"Found {len(warnings)} warning(s)." if warnings else "No issues detected.")
        ),
    }


def _extract_exif(file_path: str) -> dict | None:
    """Extract EXIF tags from an image."""
    try:
        img = Image.open(file_path)
        exif = img.getexif()
        if not exif:
            return None
        return {TAGS.get(k, k): str(v) for k, v in exif.items()}
    except Exception as e:
        logger.debug("EXIF extraction failed: %s", e)
        return None


def _analyze_jpeg_compression(file_path: str) -> dict:
    """Analyze JPEG compression characteristics."""
    try:
        img = Image.open(file_path)
        info = img.info
        quality = info.get("quality")
        quantization = info.get("quantization")

        re_compressed = False
        if quantization:
            # Multiple quantization tables may indicate re-compression
            if isinstance(quantization, dict) and len(quantization) > 2:
                re_compressed = True

        return {
            "quality": quality,
            "has_quantization_tables": quantization is not None,
            "re_compressed": re_compressed,
        }
    except Exception:
        return {"quality": None, "has_quantization_tables": False, "re_compressed": False}
