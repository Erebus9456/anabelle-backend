"""PyTorch FunASR AutoModel backend (default)."""

from __future__ import annotations

import logging

import numpy as np
import torch
from funasr import AutoModel

logger = logging.getLogger("AnabelleSenseVoicePyTorch")


class PyTorchSenseVoiceBackend:
    backend_name = "pytorch"

    def __init__(self, model_path: str, *, device: str, use_fp16: bool) -> None:
        self.quantize_label = "fp16" if use_fp16 and device == "cuda" else "fp32"
        model_kwargs = {
            "model": model_path,
            "device": device,
            "disable_update": True,
            "model_revision": "master",
        }
        if device == "cuda":
            model_kwargs["ngpu"] = 1

        logger.info(
            "SenseVoice backend: PyTorch (%s, %s)",
            device,
            self.quantize_label,
        )
        self.model = AutoModel(**model_kwargs)
        self.device = device
        self.use_fp16 = use_fp16 and device == "cuda"

        if self.device == "cuda":
            torch.backends.cudnn.benchmark = True

    def generate(
        self,
        audio: np.ndarray,
        *,
        language: str = "auto",
        use_itn: bool = True,
        cache: dict | None = None,
        sample_rate: int = 16000,
    ) -> list[dict]:
        del sample_rate  # FunASR expects 16 kHz float32 input.
        return self.model.generate(
            input=audio,
            cache=cache if cache is not None else {},
            language=language,
            use_itn=use_itn,
        )
