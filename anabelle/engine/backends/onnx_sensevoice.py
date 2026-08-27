"""ONNX Runtime SenseVoice via funasr-onnx (INT8 optional)."""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

logger = logging.getLogger("AnabelleSenseVoiceONNX")


class OnnxSenseVoiceBackend:
    backend_name = "onnx"

    def __init__(self, model_path: str, *, quantize: bool) -> None:
        try:
            from funasr_onnx import SenseVoiceSmall
            from funasr_onnx.utils.postprocess_utils import rich_transcription_postprocess
        except ImportError as exc:
            raise RuntimeError(
                "ONNX backend requires funasr-onnx and onnxruntime. "
                "Install with: pip install funasr-onnx onnxruntime"
            ) from exc

        self._postprocess = rich_transcription_postprocess
        self.quantize_label = "int8" if quantize else "fp32"
        logger.info(
            "SenseVoice backend: ONNX (%s quantize=%s)",
            model_path,
            self.quantize_label,
        )
        self.model = SenseVoiceSmall(model_path, batch_size=1, quantize=quantize)

    def generate(
        self,
        audio: np.ndarray,
        *,
        language: str = "auto",
        use_itn: bool = True,
        cache: dict | None = None,
        sample_rate: int = 16000,
    ) -> list[dict]:
        del cache  # ONNX runtime does not support FunASR streaming cache.

        audio = np.asarray(audio, dtype=np.float32).flatten()
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as handle:
            temp_path = Path(handle.name)

        try:
            sf.write(temp_path, audio, sample_rate)
            raw = self.model([str(temp_path)], language=language, use_itn=use_itn)
        finally:
            temp_path.unlink(missing_ok=True)

        if not raw:
            return [{"text": ""}]

        texts = raw if isinstance(raw, list) else [raw]
        processed = [self._postprocess(str(item)) for item in texts]
        return [{"text": processed[0] if processed else ""}]
