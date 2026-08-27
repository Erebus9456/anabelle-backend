#!/usr/bin/env python3
"""Download SenseVoiceSmall model weights from Hugging Face."""

from __future__ import annotations

import sys
from pathlib import Path
from urllib.request import Request, urlopen

from anabelle.utils.paths import ensure_data_dirs, get_model_dir

BASE_URL = "https://huggingface.co/FunAudioLLM/SenseVoiceSmall/resolve/main"
MODEL_FILES = (
    "model.pt",
    "config.yaml",
    "am.mvn",
    "chn_jpn_yue_eng_ko_spectok.bpe.model",
    "configuration.json",
)


def download_file(url: str, destination: Path) -> None:
    request = Request(url, headers={"User-Agent": "anabelle-backend/1.0"})
    with urlopen(request) as response, destination.open("wb") as handle:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            handle.write(chunk)


def download_models() -> Path:
    ensure_data_dirs()
    model_dir = get_model_dir()
    model_dir.mkdir(parents=True, exist_ok=True)

    print("--- ANABELLE Model Downloader ---")
    print(f"Target: {model_dir}")
    for filename in MODEL_FILES:
        target = model_dir / filename
        if target.exists() and target.stat().st_size > 0:
            print(f"[EXISTS] {filename}")
            continue

        url = f"{BASE_URL}/{filename}?download=true"
        print(f"[DOWNLOAD] {filename}")
        try:
            download_file(url, target)
            print(f"[SUCCESS] {filename}")
        except Exception as exc:
            print(f"[ERROR] Failed to download {filename}: {exc}", file=sys.stderr)
            raise

    print("--- Download Sync Complete ---")
    return model_dir


if __name__ == "__main__":
    download_models()
