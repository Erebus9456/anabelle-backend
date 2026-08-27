"""Abstract SenseVoice backend."""

from __future__ import annotations

from typing import Protocol

import numpy as np


class SenseVoiceBackend(Protocol):
    backend_name: str
    quantize_label: str

    def generate(
        self,
        audio: np.ndarray,
        *,
        language: str = "auto",
        use_itn: bool = True,
        cache: dict | None = None,
        sample_rate: int = 16000,
    ) -> list[dict]:
        """Run SenseVoice and return FunASR-style result list."""
