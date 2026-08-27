"""Runtime inference configuration (env vars + CLI overrides)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal

Backend = Literal["pytorch", "onnx"]
Quantize = Literal["fp32", "fp16", "int8"]
VadMode = Literal["rms", "silero", "off"]
SerMode = Literal["always", "smart", "off"]


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class InferenceConfig:
    """Select model backend, quantization, and realtime pipeline features."""

    backend: Backend = "pytorch"
    quantize: Quantize = "fp32"
    vad_mode: VadMode = "rms"
    vad_rms_threshold: float = 0.02
    enable_ser: bool = True
    ser_mode: SerMode = "smart"
    enable_semantic: bool = True
    enable_smoothing: bool = True
    smoothing_window: int = 3
    dual_path: bool = True
    min_chunk_interval: float = 0.5  # Minimum seconds between audio chunks

    @classmethod
    def from_env(cls) -> InferenceConfig:
        backend = os.environ.get("ANABELLE_BACKEND", "pytorch").strip().lower()
        quantize = os.environ.get("ANABELLE_QUANTIZE", "fp32").strip().lower()
        vad_mode = os.environ.get("ANABELLE_VAD", "rms").strip().lower()
        ser_mode = os.environ.get("ANABELLE_SER_MODE", "smart").strip().lower()

        if backend not in {"pytorch", "onnx"}:
            raise ValueError(f"Unsupported ANABELLE_BACKEND: {backend!r}")
        if quantize not in {"fp32", "fp16", "int8"}:
            raise ValueError(f"Unsupported ANABELLE_QUANTIZE: {quantize!r}")
        if vad_mode not in {"rms", "silero", "off"}:
            raise ValueError(f"Unsupported ANABELLE_VAD: {vad_mode!r}")
        if ser_mode not in {"always", "smart", "off"}:
            raise ValueError(f"Unsupported ANABELLE_SER_MODE: {ser_mode!r}")

        return cls(
            backend=backend,  # type: ignore[arg-type]
            quantize=quantize,  # type: ignore[arg-type]
            vad_mode=vad_mode,  # type: ignore[arg-type]
            vad_rms_threshold=float(os.environ.get("ANABELLE_VAD_RMS", "0.02")),
            enable_ser=_env_bool("ANABELLE_ENABLE_SER", True),
            ser_mode=ser_mode,  # type: ignore[arg-type]
            enable_semantic=_env_bool("ANABELLE_ENABLE_SEMANTIC", True),
            enable_smoothing=_env_bool("ANABELLE_ENABLE_SMOOTHING", True),
            smoothing_window=int(os.environ.get("ANABELLE_SMOOTHING_WINDOW", "3")),
            dual_path=_env_bool("ANABELLE_DUAL_PATH", True),
            min_chunk_interval=float(os.environ.get("ANABELLE_MIN_CHUNK_INTERVAL", "0.5")),
        )

    def apply_to_env(self) -> None:
        """Publish this config to process environment (CLI bootstrap)."""
        os.environ["ANABELLE_BACKEND"] = self.backend
        os.environ["ANABELLE_QUANTIZE"] = self.quantize
        os.environ["ANABELLE_VAD"] = self.vad_mode
        os.environ["ANABELLE_VAD_RMS"] = str(self.vad_rms_threshold)
        os.environ["ANABELLE_ENABLE_SER"] = "1" if self.enable_ser else "0"
        os.environ["ANABELLE_SER_MODE"] = self.ser_mode
        os.environ["ANABELLE_ENABLE_SEMANTIC"] = "1" if self.enable_semantic else "0"
        os.environ["ANABELLE_ENABLE_SMOOTHING"] = "1" if self.enable_smoothing else "0"
        os.environ["ANABELLE_SMOOTHING_WINDOW"] = str(self.smoothing_window)
        os.environ["ANABELLE_DUAL_PATH"] = "1" if self.dual_path else "0"
        os.environ["ANABELLE_MIN_CHUNK_INTERVAL"] = str(self.min_chunk_interval)

    def summary(self) -> dict[str, str | bool | float | int]:
        return {
            "backend": self.backend,
            "quantize": self.quantize,
            "vad_mode": self.vad_mode,
            "vad_rms_threshold": self.vad_rms_threshold,
            "enable_ser": self.enable_ser,
            "ser_mode": self.ser_mode,
            "enable_semantic": self.enable_semantic,
            "enable_smoothing": self.enable_smoothing,
            "smoothing_window": self.smoothing_window,
            "dual_path": self.dual_path,
            "min_chunk_interval": self.min_chunk_interval,
        }
