"""Resolve persistent data locations (models, test audio) for local and Colab."""

from __future__ import annotations

import os
from pathlib import Path

DEFAULT_COLAB_DATA_DIR = Path("/content/anabelle-data")
ENV_DATA_DIR = "ANABELLE_DATA_DIR"

# Repository root (parent of the `anabelle` package).
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def is_colab() -> bool:
    try:
        import google.colab  # noqa: F401

        return True
    except ImportError:
        return False


def get_data_dir() -> Path:
    """
    Return the root directory for downloaded assets.

    Priority:
    1. ANABELLE_DATA_DIR env var
    2. /content/anabelle-data on Colab
    3. Repository root for local development
    """
    if explicit := os.environ.get(ENV_DATA_DIR):
        return Path(explicit).expanduser().resolve()

    if is_colab():
        return DEFAULT_COLAB_DATA_DIR

    return PROJECT_ROOT


def get_model_dir() -> Path:
    return get_data_dir() / "models" / "SenseVoiceSmall"


def get_test_audio_dir() -> Path:
    return get_data_dir() / "test" / "audio"


def get_ravdess_cache_dir() -> Path:
    return get_data_dir() / "test" / ".ravdess-download"


def get_test_reports_dir() -> Path:
    return get_data_dir() / "test" / "reports"


def ensure_data_dirs() -> Path:
    data_dir = get_data_dir()
    get_model_dir().mkdir(parents=True, exist_ok=True)
    get_test_audio_dir().mkdir(parents=True, exist_ok=True)
    get_ravdess_cache_dir().mkdir(parents=True, exist_ok=True)
    get_test_reports_dir().mkdir(parents=True, exist_ok=True)
    return data_dir
