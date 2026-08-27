#!/usr/bin/env python3
"""Quick SER smoke test on one RAVDESS clip."""

from __future__ import annotations

from test_utils import bootstrap_project_root

bootstrap_project_root()

from colab_compat import apply_runtime_patches

apply_runtime_patches()

import librosa

from device_utils import get_device_info
from paths import get_test_audio_dir
from ser_engine import SerEngine


def main() -> None:
    audio_dir = get_test_audio_dir()
    if not audio_dir.is_dir():
        raise FileNotFoundError(
            f"Test audio not found at {audio_dir}. Run: python setup.py --skip-deps"
        )

    sample = next(audio_dir.glob("Actor_*/**/*.wav"))
    audio, sr = librosa.load(sample, sr=16000)

    device = get_device_info().device
    ser = SerEngine(device=device)
    result = ser.predict(audio, sample_rate=sr)

    print(f"Sample: {sample}")
    print(f"Device: {device}")
    print(f"SER result: {result}")


if __name__ == "__main__":
    main()
