"""Voice-activity gating before heavy inference."""

from __future__ import annotations

import logging
from typing import Literal

import numpy as np

from anabelle.config import InferenceConfig, VadMode

logger = logging.getLogger("AnabelleVAD")

GateReason = Literal["rms", "silero", "off", "speech"]


class VadGate:
    """Cheap pre-filter: skip SenseVoice/SER when input is likely silence/noise."""

    def __init__(self, config: InferenceConfig) -> None:
        self.mode: VadMode = config.vad_mode
        self.rms_threshold = config.vad_rms_threshold
        self._silero_model = None
        self._silero_utils = None

    @staticmethod
    def rms(audio: np.ndarray) -> float:
        audio = np.asarray(audio, dtype=np.float32).flatten()
        if audio.size == 0:
            return 0.0
        return float(np.sqrt(np.mean(audio**2)))

    def _ensure_silero(self) -> None:
        if self._silero_model is not None:
            return
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("Silero VAD requires PyTorch") from exc

        logger.info("Loading Silero VAD (one-time torch.hub fetch)...")
        model, utils = torch.hub.load(  # type: ignore[no-untyped-call]
            "snakers4/silero-vad",
            "silero_vad",
            trust_repo=True,
        )
        self._silero_model = model
        self._silero_utils = utils

    def _silero_has_speech(self, audio: np.ndarray, sample_rate: int) -> bool:
        self._ensure_silero()
        import torch

        waveform = torch.from_numpy(np.asarray(audio, dtype=np.float32).flatten())
        get_speech_timestamps = self._silero_utils[0]
        timestamps = get_speech_timestamps(
            waveform,
            self._silero_model,
            sampling_rate=sample_rate,
            return_seconds=False,
        )
        return bool(timestamps)

    def should_infer(self, audio: np.ndarray, sample_rate: int = 16000) -> tuple[bool, GateReason]:
        """
        Return (should_run_heavy_models, reason).

        When False, callers should skip SenseVoice/SER and return a lightweight result.
        """
        if self.mode == "off":
            return True, "off"

        level = self.rms(audio)
        if self.mode == "rms":
            if level < self.rms_threshold:
                return False, "rms"
            return True, "speech"

        # silero mode: RMS pre-check then Silero confirmation
        if level < self.rms_threshold:
            return False, "rms"

        try:
            if self._silero_has_speech(audio, sample_rate):
                return True, "speech"
            return False, "silero"
        except Exception as exc:
            logger.warning("Silero VAD failed, falling back to RMS gate: %s", exc)
            if level < self.rms_threshold:
                return False, "rms"
            return True, "speech"
