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


def run_command(
    command: list[str],
    *,
    label: str,
    check: bool = True,
) -> subprocess.CompletedProcess:
    print(f"\n>>> {label}")
    print(" ".join(command))
    return subprocess.run(command, check=check, cwd=PROJECT_ROOT)


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


def is_python_313_or_newer() -> bool:
    return (sys.version_info.major, sys.version_info.minor) >= (3, 13)


def pip_base(*extra: str) -> list[str]:
    command = [sys.executable, "-m", "pip", "install", "--upgrade", "--prefer-binary"]
    command.extend(extra)
    return command


def install_torch(profile: str) -> None:
    if profile == "colab":
        run_command(
            pip_base(
                "torch",
                "torchaudio",
                "--index-url",
                "https://download.pytorch.org/whl/cu124",
            ),
            label="Installing PyTorch (Colab CUDA)",
        )
        return

    if profile == "mac":
        run_command(
            pip_base("torch==2.2.2", "torchaudio==2.2.2"),
            label="Installing PyTorch (macOS)",
        )
        return

    run_command(
        pip_base(
            "torch",
            "torchaudio",
            "--index-url",
            "https://download.pytorch.org/whl/cu124",
        ),
        label="Installing PyTorch (CUDA)",
    )


def install_numpy() -> None:
    if is_python_313_or_newer():
        run_command(
            pip_base("numpy>=2.0.0"),
            label="Installing NumPy 2.x (Python 3.13+)",
        )
        return

    run_command(
        pip_base("numpy==1.26.4"),
        label="Installing NumPy 1.26.4",
    )


def install_base_requirements(profile: str) -> None:
    requirements_file = (
        PROJECT_ROOT / "requirements-colab.txt"
        if profile == "colab"
        else PROJECT_ROOT / "requirements.txt"
    )
    run_command(
        pip_base("-r", str(requirements_file)),
        label=f"Installing base requirements ({requirements_file.name})",
    )


def install_ai_stack(profile: str) -> None:
    """
    Install funasr + transformers + tokenizers without compiling tokenizers from source.

    On Python 3.13, tokenizers <0.20 has no cp313 wheels and fails to build.
    We pre-install binary wheels, then install funasr with --no-deps.
    """
    if is_python_313_or_newer():
        constraints = PROJECT_ROOT / "constraints-py313.txt"
        run_command(
            pip_base(
                "--constraint",
                str(constraints),
                "--only-binary=tokenizers",
                "tokenizers>=0.21.0",
                "transformers>=4.46.0,<4.50",
            ),
            label="Installing tokenizers + transformers (Python 3.13 wheels)",
        )
        run_command(
            pip_base("-r", str(PROJECT_ROOT / "requirements-funasr-runtime.txt")),
            label="Installing FunASR runtime dependencies",
        )
        run_command(
            pip_base("funasr==1.4.4", "--no-deps"),
            label="Installing funasr (no dependency re-resolve)",
        )
        run_command(
            pip_base("modelscope>=1.15.0"),
            label="Installing modelscope",
        )
        return

    if profile == "colab":
        constraints = PROJECT_ROOT / "constraints-colab.txt"
        run_command(
            pip_base(
                "--constraint",
                str(constraints),
                "--only-binary=tokenizers",
                "tokenizers>=0.19.0",
                "transformers>=4.44.0,<4.45",
                "funasr==1.4.4",
                "modelscope>=1.15.0",
            ),
            label="Installing FunASR stack (Colab)",
        )
        return

    run_command(
        pip_base(
            "transformers>=4.44.0,<4.45",
            "funasr==1.4.4",
            "modelscope>=1.15.0",
        ),
        label="Installing FunASR stack (local)",
    )


def install_optional_text_processing() -> None:
    """WeTextProcessing helps FunASR registration; skip gracefully if it cannot build."""
    result = run_command(
        pip_base("WeTextProcessing>=1.0.3"),
        label="Installing WeTextProcessing (optional ITN support)",
        check=False,
    )
    if result.returncode != 0:
        print(
            "Warning: WeTextProcessing install skipped (common on Python 3.13). "
            "FunASR built-in text processing will still be used."
        )


def verify_environment() -> None:
    from colab_compat import apply_runtime_patches

    apply_runtime_patches()

    import numpy as np
    import torch
    from funasr import AutoModel  # noqa: F401

    from paths import get_data_dir, get_model_dir

    print("\n--- Environment Check ---")
    print(f"Python:   {sys.version.split()[0]}")
    print(f"NumPy:    {np.__version__}")
    print(f"PyTorch:  {torch.__version__}")
    print(f"CUDA:     {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU:      {torch.cuda.get_device_name(0)}")
    print(f"Data dir: {get_data_dir()}")
    print(f"Model at: {get_model_dir()}")
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

    from paths import ensure_data_dirs, get_data_dir

    data_dir = ensure_data_dirs()
    print(f"Persistent data directory: {data_dir}")
    if profile == "colab":
        print(
            "Colab assets live outside the git repo — safe to git pull without re-downloading."
        )

    if not args.skip_deps:
        run_command(
            [sys.executable, "-m", "pip", "install", "--upgrade", "pip"],
            label="Upgrading pip",
        )
        install_torch(profile)
        install_numpy()
        install_base_requirements(profile)
        install_ai_stack(profile)
        if profile == "colab":
            install_optional_text_processing()

    if not args.skip_models:
        from download_models import download_models

        download_models()

    if not args.skip_test_data:
        from download_test_data import download_test_data

        download_test_data()

    verify_environment()
    print("\nANABELLE setup complete.")
    print("Start the gateway:  python run.py serve")
    print("Run static tests:   python run.py test")


if __name__ == "__main__":
    main()
