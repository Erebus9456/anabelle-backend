"""SenseVoice inference backends (PyTorch FunASR vs ONNX Runtime)."""

from __future__ import annotations

from anabelle.engine.backends.base import SenseVoiceBackend
from anabelle.engine.backends.onnx_sensevoice import OnnxSenseVoiceBackend
from anabelle.engine.backends.pytorch_sensevoice import PyTorchSenseVoiceBackend

__all__ = [
    "SenseVoiceBackend",
    "create_sensevoice_backend",
]


def create_sensevoice_backend(
    *,
    backend: str,
    quantize: str,
    model_path: str,
    device: str,
) -> SenseVoiceBackend:
    if backend == "onnx":
        use_int8 = quantize == "int8"
        return OnnxSenseVoiceBackend(model_path, quantize=use_int8)

    use_fp16 = quantize == "fp16"
    return PyTorchSenseVoiceBackend(
        model_path,
        device=device,
        use_fp16=use_fp16,
    )
