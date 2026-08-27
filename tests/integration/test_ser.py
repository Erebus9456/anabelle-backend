"""Quick SER smoke test on one RAVDESS clip."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import librosa

from anabelle.engine.ser import SerEngine
from anabelle.utils.compat import apply_runtime_patches
from anabelle.utils.device import get_device_info
from anabelle.utils.paths import get_test_audio_dir

apply_runtime_patches()


def main() -> None:
    audio_dir = get_test_audio_dir()
    if not audio_dir.is_dir():
        raise FileNotFoundError(
            f"Test audio not found at {audio_dir}. Run: python setup.py"
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
