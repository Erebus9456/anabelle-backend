"""Background inference worker for dual-path realtime WebSocket responses."""

from __future__ import annotations

import asyncio
import json
import logging
import queue
import threading
import time
from typing import TYPE_CHECKING, Any

from anabelle.config import InferenceConfig
from anabelle.engine.smoother import EmotionSmoother

if TYPE_CHECKING:
    from fastapi import WebSocket

    from anabelle.engine.core import AnabelleEngine

logger = logging.getLogger("AnabellePipeline")


class RealtimePipeline:
    """
    Per-connection pipeline:
    - immediate acoustic reflex on receive
    - heavy hybrid inference on a background worker thread
    - stale result dropping when newer audio arrives
    """

    def __init__(
        self,
        engine: AnabelleEngine,
        websocket: WebSocket,
        loop: asyncio.AbstractEventLoop,
        config: InferenceConfig,
    ) -> None:
        self.engine = engine
        self.websocket = websocket
        self.loop = loop
        self.config = config
        self._seq = 0
        self._lock = threading.Lock()
        self._queue: queue.Queue[tuple[int, Any] | None] = queue.Queue()
        self._smoother = (
            EmotionSmoother(config.smoothing_window) if config.enable_smoothing else None
        )
        self._worker = threading.Thread(target=self._worker_loop, daemon=True, name="anabelle-infer")
        self._worker.start()
        self._last_chunk_time = 0.0
        self._dropped_chunks = 0

    async def handle_audio(self, audio_data) -> None:
        current_time = time.time()
        time_since_last = current_time - self._last_chunk_time

        # Rate limit: drop chunks that arrive too frequently
        if time_since_last < self.config.min_chunk_interval:
            self._dropped_chunks += 1
            if self._dropped_chunks % 10 == 0:  # Log every 10 drops to avoid spam
                logger.warning(
                    "Rate limiter: dropped %d chunks (interval %.3fs < %.3fs)",
                    self._dropped_chunks,
                    time_since_last,
                    self.config.min_chunk_interval,
                )
            return

        self._last_chunk_time = current_time

        with self._lock:
            self._seq += 1
            seq = self._seq

        reflex = await asyncio.to_thread(self.engine.analyze_reflex, audio_data)
        await self._send({"type": "reflex", "seq": seq, **reflex})

        self._queue.put((seq, audio_data))

    async def close(self) -> None:
        self._queue.put(None)
        self._worker.join(timeout=2.0)

    async def _send(self, payload: dict) -> None:
        await self.websocket.send_text(json.dumps(payload))

    def _dispatch(self, payload: dict) -> None:
        future = asyncio.run_coroutine_threadsafe(self._send(payload), self.loop)
        future.add_done_callback(self._log_send_error)

    @staticmethod
    def _log_send_error(future) -> None:
        try:
            future.result()
        except Exception as exc:
            logger.debug("WebSocket send skipped: %s", exc)

    def _worker_loop(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:
                return

            seq, audio_data = item
            with self._lock:
                if seq < self._seq:
                    continue

            try:
                result = self.engine.analyze_chunk(audio_data)
            except Exception as exc:
                logger.error("Pipeline inference error: %s", exc)
                continue

            with self._lock:
                if seq < self._seq:
                    continue

            emotion = result["emotion"]
            if self._smoother is not None:
                emotion = self._smoother.update(emotion)
                result = {**result, "emotion": emotion}

            self._dispatch({"type": "emotion", "seq": seq, **result})
