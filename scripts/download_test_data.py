#!/usr/bin/env python3
"""Download and extract the RAVDESS speech test dataset."""

from __future__ import annotations

import shutil
import sys
import zipfile
from pathlib import Path
from urllib.request import Request, urlopen

from anabelle.utils.paths import ensure_data_dirs, get_ravdess_cache_dir, get_test_audio_dir

DATASET_URL = "https://zenodo.org/records/1188976/files/Audio_Speech_Actors_01-24.zip?download=1"
ACTORS = tuple(f"Actor_{index:02d}" for index in range(1, 25))
EXPECTED_WAV_COUNT = 60


def download_file(url: str, destination: Path) -> None:
    request = Request(url, headers={"User-Agent": "anabelle-backend/1.0"})
    with urlopen(request) as response, destination.open("wb") as handle:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            handle.write(chunk)


def dataset_is_complete(data_dir: Path) -> bool:
    for actor in ACTORS:
        actor_dir = data_dir / actor
        if not actor_dir.is_dir():
            return False
        if len(list(actor_dir.glob("*.wav"))) != EXPECTED_WAV_COUNT:
            return False
    return True


def download_test_data() -> Path:
    ensure_data_dirs()
    data_dir = get_test_audio_dir()
    download_dir = get_ravdess_cache_dir()
    archive_path = download_dir / "Audio_Speech_Actors_01-24.zip"
    extract_dir = download_dir / "extracted"

    data_dir.mkdir(parents=True, exist_ok=True)
    download_dir.mkdir(parents=True, exist_ok=True)

    if dataset_is_complete(data_dir):
        print(f"RAVDESS audio is already complete in {data_dir}")
        return data_dir

    print(f"Downloading RAVDESS speech audio to {data_dir} ...")
    download_file(DATASET_URL, archive_path)

    if extract_dir.exists():
        shutil.rmtree(extract_dir)
    extract_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(archive_path) as archive:
        archive.extractall(extract_dir)

    actor_roots = list(extract_dir.rglob("Actor_01"))
    if not actor_roots:
        raise RuntimeError("Downloaded archive does not contain Actor_01")

    source_root = actor_roots[0].parent
    for actor in ACTORS:
        if not (source_root / actor).is_dir():
            raise RuntimeError(f"Downloaded archive is missing {actor}")

    for actor in ACTORS:
        actor_target = data_dir / actor
        if actor_target.exists():
            shutil.rmtree(actor_target)
        shutil.move(str(source_root / actor), str(actor_target))

    shutil.rmtree(extract_dir, ignore_errors=True)
    print(f"RAVDESS audio is ready in {data_dir}")
    return data_dir


if __name__ == "__main__":
    try:
        download_test_data()
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
