"""
VeritasAI – Live Detection WebSocket Router

Streams real-time deepfake detection results to the browser.

Protocol:
  Client → Server:
    • Binary message  — raw JPEG frame bytes
    • Text message    — JSON config, e.g. {"type":"config","heatmap":true,"frame_skip":5}
  Server → Client:
    • Text message    — JSON result per processed frame

Frame-skip: only every Nth received binary frame is processed (default N=3).
Grad-CAM: off by default; when enabled runs every 5th *processed* frame.
"""

import json
import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from services.live_inference import get_live_detector

logger = logging.getLogger(__name__)

router = APIRouter()


@router.websocket("/live-detection")
async def live_detection_ws(ws: WebSocket):
    await ws.accept()
    logger.info("Live detection WebSocket connected: %s", ws.client)

    detector = get_live_detector()
    detector.reset()  # fresh rolling-average per session

    # ── Session config (mutable via client text messages) ──────────
    frame_skip: int = 3           # process every Nth frame
    heatmap_enabled: bool = False
    heatmap_interval: int = 5     # run Grad-CAM every Kth *processed* frame

    frame_index: int = 0          # total frames received
    processed_count: int = 0      # frames actually inferred

    try:
        while True:
            message = await ws.receive()

            # ── Text message: config update ────────────────────────
            if "text" in message:
                try:
                    payload = json.loads(message["text"])
                    if payload.get("type") == "config":
                        if "heatmap" in payload:
                            heatmap_enabled = bool(payload["heatmap"])
                        if "frame_skip" in payload:
                            frame_skip = max(1, min(10, int(payload["frame_skip"])))
                        logger.debug(
                            "Config updated: frame_skip=%d, heatmap=%s",
                            frame_skip, heatmap_enabled,
                        )
                except (json.JSONDecodeError, ValueError):
                    pass  # ignore malformed config
                continue

            # ── Binary message: JPEG frame ─────────────────────────
            if "bytes" in message:
                frame_bytes: bytes = message["bytes"]
                frame_index += 1

                # Frame-skip gate
                if frame_index % frame_skip != 0:
                    continue

                # Decide whether this processed frame gets a heatmap
                processed_count += 1
                do_heatmap = (
                    heatmap_enabled and processed_count % heatmap_interval == 0
                )

                result = detector.predict(frame_bytes, generate_heatmap=do_heatmap)
                if result is None:
                    # Invalid / corrupt frame — skip silently
                    continue

                result["frame_index"] = frame_index
                await ws.send_json(result)

    except WebSocketDisconnect:
        logger.info("Live detection WebSocket disconnected: %s", ws.client)
    except Exception as e:
        logger.error("Live detection WebSocket error: %s", e)
        try:
            await ws.close(code=1011, reason=str(e)[:120])
        except Exception:
            pass
