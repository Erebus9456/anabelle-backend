"""Compatibility shims for Google Colab and NumPy 2.x runtimes."""

from __future__ import annotations

import logging
import sys

logger = logging.getLogger("AnabelleCompat")


def is_colab() -> bool:
    try:
        import google.colab  # noqa: F401

        return True
    except ImportError:
        return False


def apply_runtime_patches() -> None:
    """Apply environment-specific fixes before importing funasr/torch stacks."""
    if is_colab():
        logger.info("Colab runtime detected — applying compatibility patches")

    _patch_numpy2_warnings()
    _ensure_funasr_registration()


def _patch_numpy2_warnings() -> None:
    major = sys.version_info.major
    minor = sys.version_info.minor

    if major == 3 and minor >= 13:
        logger.warning(
            "Python 3.13+ detected. NumPy 1.x is unavailable; using NumPy 2.x with "
            "best-effort compatibility. For production, prefer Python 3.10–3.12."
        )

    try:
        import numpy as np
    except ImportError:
        return

    if np.__version__.startswith("2."):
        logger.warning(
            "NumPy %s detected. Some AI wheels were built against NumPy 1.x; "
            "if imports fail, use Python 3.11 with numpy==1.26.4.",
            np.__version__,
        )


def _ensure_funasr_registration() -> None:
    """
    Force-import funasr submodules so SenseVoiceSmall registers before AutoModel
    tries to resolve a local model path as a remote ModelScope ID (404 errors).
    """
    try:
        import funasr  # noqa: F401
    except ImportError:
        return

    registration_imports = (
        "funasr.models.sense_voice.model",
        "funasr.models.sense_voice",
        "fun_text_processing",
    )

    for module_name in registration_imports:
        try:
            __import__(module_name)
        except ImportError as exc:
            logger.debug("Optional funasr import skipped (%s): %s", module_name, exc)
