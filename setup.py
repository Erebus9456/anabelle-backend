#!/usr/bin/env python3
"""
One-shot ANABELLE setup.

Installs dependencies, downloads model weights, and fetches RAVDESS test data.
Works locally and in Google Colab.
"""

from __future__ import annotations

import argparse
import platform
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent


def run_command(command: list[str], *, label: str) -> None:
    print(f"\n>>> {label}")
    print(" ".join(command))
    subprocess.run(command, check=True, cwd=PROJECT_ROOT)


def detect_profile(force_colab: bool = False) -> str:
    if force_colab:
        return "colab"

    try:
        import google.colab  # noqa: F401

        return "colab"
    except ImportError:
        pass

    system = platform.system().lower()
    if system == "darwin":
        return "mac"
    return "local"


def install_torch(profile: str) -> None:
    if profile == "colab":
        run_command(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--upgrade",
                "torch",
                "torchaudio",
                "--index-url",
                "https://download.pytorch.org/whl/cu124",
            ],
            label="Installing PyTorch (Colab CUDA)",
        )
        return

    if profile == "mac":
        run_command(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "torch==2.2.2",
                "torchaudio==2.2.2",
            ],
            label="Installing PyTorch (macOS)",
        )
        return

    run_command(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "torch",
            "torchaudio",
            "--index-url",
            "https://download.pytorch.org/whl/cu124",
        ],
        label="Installing PyTorch (CUDA)",
    )


def install_numpy(profile: str) -> None:
    version = sys.version_info
    if (version.major, version.minor) >= (3, 13):
        run_command(
            [sys.executable, "-m", "pip", "install", "numpy>=2.0.0"],
            label="Installing NumPy 2.x (Python 3.13+)",
        )
        return

    run_command(
        [sys.executable, "-m", "pip", "install", "numpy==1.26.4"],
        label="Installing NumPy 1.26.4",
    )


def install_requirements(profile: str) -> None:
    requirements_file = (
        PROJECT_ROOT / "requirements-colab.txt"
        if profile == "colab"
        else PROJECT_ROOT / "requirements.txt"
    )
    run_command(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--upgrade",
            "-r",
            str(requirements_file),
        ],
        label=f"Installing project requirements ({requirements_file.name})",
    )


def verify_environment() -> None:
    from colab_compat import apply_runtime_patches

    apply_runtime_patches()

    import numpy as np
    import torch
    from funasr import AutoModel  # noqa: F401

    print("\n--- Environment Check ---")
    print(f"Python:  {sys.version.split()[0]}")
    print(f"NumPy:   {np.__version__}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA:    {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU:     {torch.cuda.get_device_name(0)}")
    print("FunASR registration: OK")


def main() -> None:
    parser = argparse.ArgumentParser(description="ANABELLE one-click setup")
    parser.add_argument("--colab", action="store_true", help="Force Colab profile")
    parser.add_argument("--skip-deps", action="store_true", help="Skip pip installs")
    parser.add_argument("--skip-models", action="store_true", help="Skip model download")
    parser.add_argument(
        "--skip-test-data",
        action="store_true",
        help="Skip RAVDESS test dataset download",
    )
    args = parser.parse_args()

    profile = detect_profile(force_colab=args.colab)
    print(f"Setup profile: {profile}")

    if not args.skip_deps:
        run_command(
            [sys.executable, "-m", "pip", "install", "--upgrade", "pip"],
            label="Upgrading pip",
        )
        install_torch(profile)
        install_numpy(profile)
        install_requirements(profile)

    if not args.skip_models:
        from download_models import download_models

        download_models(PROJECT_ROOT)

    if not args.skip_test_data:
        from download_test_data import download_test_data

        download_test_data(PROJECT_ROOT)

    verify_environment()
    print("\nANABELLE setup complete.")
    print("Start the gateway:  python run.py serve")
    print("Run static tests:   python run.py test")


if __name__ == "__main__":
    main()
