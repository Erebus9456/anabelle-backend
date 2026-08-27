"""Shared GPU / accelerator detection for ANABELLE."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch

logger = logging.getLogger("AnabelleDevice")


@dataclass(frozen=True)
class DeviceInfo:
    device: str
    label: str
    use_fp16: bool


def get_device_info() -> DeviceInfo:
    """Pick the best available inference device and whether FP16 is safe."""
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        capability = torch.cuda.get_device_capability(0)
        use_fp16 = capability[0] >= 7
        logger.info("CUDA device: %s (compute %s.%s)", name, capability[0], capability[1])
        return DeviceInfo(device="cuda", label=name, use_fp16=use_fp16)

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() and torch.backends.mps.is_built():
        logger.info("Apple Metal (MPS) acceleration enabled")
        return DeviceInfo(device="mps", label="Apple MPS", use_fp16=False)

    logger.info("No GPU detected; falling back to CPU inference")
    return DeviceInfo(device="cpu", label="CPU", use_fp16=False)
