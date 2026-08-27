"""FastAPI WebSocket gateway for the ANABELLE avatar."""

from __future__ import annotations

import asyncio
import json
import logging
import os

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from anabelle.engine.core import AnabelleEngine
from anabelle.utils.compat import apply_runtime_patches
from anabelle.utils.device import get_device_info
from anabelle.utils.paths import ensure_data_dirs, get_model_dir

apply_runtime_patches()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("AnabelleGateway")

app = FastAPI(
    title="ANABELLE Backend",
    description="Hybrid affective inference gateway for the ANABELLE avatar",
    version="1.0.0",
)

engine: AnabelleEngine | None = None


@app.on_event("startup")
async def startup() -> None:
    global engine
    ensure_data_dirs()
    if not get_model_dir().is_dir():
        logger.warning("SenseVoice model directory missing: %s", get_model_dir())
    engine = AnabelleEngine(enable_ser=True)
    device = get_device_info()
    logger.info("Gateway ready on %s (SER=%s)", device.label, engine.ser_available)


@app.get("/health")
async def health() -> dict:
    """Load-balancer / k8s health probe."""
    device = get_device_info()
    return {
        "status": "ok" if engine is not None else "starting",
        "device": device.device,
        "device_label": device.label,
        "ser_available": bool(engine and engine.ser_available),
        "model_dir": str(get_model_dir()),
    }


@app.websocket("/ws/anabelle")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    logger.info("Client connected to ANABELLE Gateway.")

    if engine is None:
        await websocket.close(code=1011, reason="Engine not initialized")
        return

    try:
        while True:
            data = await websocket.receive_bytes()
            audio_array = np.frombuffer(data, dtype=np.float32)

            result = await asyncio.to_thread(engine.analyze_chunk, audio_array)
            logger.info("Engine Decision: %s (%s)", result["emotion"], result["source"])
            await websocket.send_text(json.dumps(result))

    except WebSocketDisconnect:
        logger.info("Client disconnected.")
    except Exception as exc:
        logger.error("Gateway Error: %s", exc)
    finally:
        try:
            await websocket.close()
        except Exception:
            pass
