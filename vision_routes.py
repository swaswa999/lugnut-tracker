"""Local capture/label/detect routes, deliberately without robot control APIs."""
import base64
from collections import deque
import json
from pathlib import Path
import time

import cv2
from fastapi import APIRouter, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from starlette.concurrency import run_in_threadpool

from alignment import Alignment, point
from vision_data import Dataset
from vision_model import LugnutDetector

router = APIRouter()
dataset = Dataset()
detector = LugnutDetector()
ROOT = Path(__file__).resolve().parent


def decode_image(data):
    # Reuse the same input bound as the original tracking app without a circular import.
    from live_app import decode
    return decode(data)


@router.get("/align")
def alignment_page():
    return FileResponse(ROOT / "alignment.html")


@router.get("/alignment.js")
def alignment_script():
    return FileResponse(ROOT / "alignment.js", media_type="text/javascript")


@router.get("/vision/status")
def status():
    return {"detector": detector.status(), "dataset": dataset.status(), "camera_mount": "wrist"}


@router.post("/vision/examples")
async def save_example(request: Request):
    if request.headers.get("origin") != f"http://{request.headers.get('host')}":
        raise HTTPException(403, "Local same-origin requests only.")
    body = bytearray()
    async for chunk in request.stream():
        body.extend(chunk)
        if len(body) > 3_000_000:
            raise HTTPException(413, "Example is too large.")
    try:
        payload = json.loads(body)
        if not isinstance(payload, dict):
            raise ValueError("Invalid example.")
        frame = decode_image(base64.b64decode(payload["frame"], validate=True))
        return await run_in_threadpool(dataset.save, frame, payload["boxes"],
                                      payload["session"], payload["split"], payload.get("reviewed"))
    except (ValueError, KeyError, TypeError, cv2.error) as exc:
        raise HTTPException(400, str(exc)[:250]) from exc


@router.websocket("/vision/detect")
async def detect(socket: WebSocket):
    if socket.headers.get("origin") != f"http://{socket.headers.get('host')}":
        await socket.close(code=1008)
        return
    await socket.accept()
    alignment = Alignment()
    last_detections = []
    last_frame = None
    last_finished = 0.
    recent_frames = deque(maxlen=4)
    confidence = .5
    try:
        while True:
            message = await socket.receive()
            if message["type"] == "websocket.disconnect":
                break
            try:
                if message.get("text") is None or len(message["text"]) > 3_000_000:
                    raise ValueError("Invalid or oversized request.")
                payload = json.loads(message["text"])
                if not isinstance(payload, dict):
                    raise ValueError("Invalid request.")
                kind = payload.get("type")
                if kind == "configure":
                    confidence = float(payload.get("confidence", .5))
                    if not .05 <= confidence <= .95:
                        raise ValueError("Confidence must be between 0.05 and 0.95.")
                    alignment = Alignment(payload.get("aim"), float(payload.get("tolerance_px", 5)))
                    last_detections, last_frame = [], None
                    recent_frames.clear()
                    await socket.send_json({"type": "configured"})
                elif kind == "select":
                    selected = next((f for f in recent_frames if f[0] == payload.get("frame_id")), None)
                    if selected is None or time.monotonic() - selected[2] > .75:
                        raise ValueError("Detection is stale. Select a lug nut on a fresh frame.")
                    alignment.select(selected[1], point(payload["point"]))
                    await socket.send_json({"type": "selected"})
                elif kind == "clear":
                    alignment.clear()
                    await socket.send_json({"type": "cleared"})
                elif kind == "frame":
                    frame_id = payload["frame_id"]
                    if not isinstance(frame_id, int) or isinstance(frame_id, bool) or frame_id < 0:
                        raise ValueError("Invalid frame ID.")
                    if last_frame is not None and frame_id <= last_frame:
                        raise ValueError("Frame IDs must increase.")
                    started = time.monotonic()
                    frame = decode_image(base64.b64decode(payload["frame"], validate=True))
                    found = await run_in_threadpool(detector.predict, frame, confidence)
                    finished = time.monotonic()
                    if finished - started > .75:
                        alignment.clear("stale")
                        found = []
                    guidance = alignment.update(found, frame.shape[1], frame.shape[0], now=finished)
                    last_frame, last_detections, last_finished = frame_id, found, finished
                    recent_frames.append((frame_id, found, finished))
                    await socket.send_json({"type": "result", "frame_id": frame_id,
                                            "detections": found, "alignment": guidance,
                                            "processing_ms": round((finished-started)*1000, 1)})
                else:
                    raise ValueError("Unknown request.")
            except (ValueError, KeyError, TypeError, cv2.error, RuntimeError, OSError) as exc:
                alignment.clear("error")
                last_detections = []
                recent_frames.clear()
                await socket.send_json({"type": "error", "message": str(exc)[:250],
                                        "alignment": {"image_aligned": False, "robot_motion_enabled": False}})
    except WebSocketDisconnect:
        pass
