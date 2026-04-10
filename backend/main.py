"""
VeritasAI – FastAPI Application Entry Point
"""

import asyncio
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from config import (
    CORS_ORIGINS,
    FILE_TTL_SECONDS,
    HEATMAP_DIR,
    RATE_LIMIT,
    REPORT_DIR,
    UPLOAD_DIR,
)
from routers import upload, feedback, audio
# from routers import live_detection  # DISABLED — Live Detection deactivated


from services.inference import _load_model, reload_model

# ── Temp-file cleanup background task ──────────────────────────────
async def _cleanup_loop():
    """Delete temp files older than FILE_TTL_SECONDS every 5 minutes."""
    while True:
        now = time.time()
        for folder in (UPLOAD_DIR, REPORT_DIR, HEATMAP_DIR):
            for f in folder.iterdir():
                if f.is_file() and (now - f.stat().st_mtime) > FILE_TTL_SECONDS:
                    f.unlink(missing_ok=True)
        await asyncio.sleep(300)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Preload the model so inference connects instantly on the first request
    import logging
    logger = logging.getLogger("veritasai")
    logging.basicConfig(level=logging.INFO)
    logger.info("="*50)
    logger.info("  VeritasAI Backend Starting")
    logger.info("="*50)
    model = _load_model()
    from services.model_manager import ModelManager
    try:
        mgr = ModelManager()
        logger.info("  Active model version: v%s", mgr.get_active_version())
        logger.info("  Model path: %s", mgr.get_active_model_path())
    except Exception:
        pass
    logger.info("="*50)
    task = asyncio.create_task(_cleanup_loop())
    yield
    task.cancel()


# ── App setup ──────────────────────────────────────────────────────
limiter = Limiter(key_func=get_remote_address, default_limits=[RATE_LIMIT])

app = FastAPI(
    title="VeritasAI",
    description="AI-powered deepfake detection & forensic analysis platform",
    version="1.0.0",
    lifespan=lifespan,
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve heatmap images and reports as static files
app.mount("/static/heatmaps", StaticFiles(directory=str(HEATMAP_DIR)), name="heatmaps")
app.mount("/static/reports", StaticFiles(directory=str(REPORT_DIR)), name="reports")

# ── Routers ────────────────────────────────────────────────────────
app.include_router(upload.router, prefix="/api")
app.include_router(feedback.router, prefix="/api")
app.include_router(audio.router, prefix="/api")
# app.include_router(live_detection.router, prefix="/api")  # DISABLED — Live Detection deactivated


@app.get("/api/health")
async def health():
    return {"status": "ok", "service": "VeritasAI"}


@app.post("/api/reload-model")
async def api_reload_model():
    """Force reload the model from disk (after deploying new weights)."""
    model = reload_model()
    from services.model_manager import ModelManager
    try:
        mgr = ModelManager()
        version = mgr.get_active_version()
    except Exception:
        version = "unknown"
    return {"status": "reloaded", "version": f"v{version}"}
