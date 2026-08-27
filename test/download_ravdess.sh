#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
DATA_DIR="$SCRIPT_DIR/audio"
DOWNLOAD_DIR="$SCRIPT_DIR/.ravdess-download"
ARCHIVE_PATH="$DOWNLOAD_DIR/Audio_Speech_Actors_01-24.zip"
EXTRACT_DIR="$DOWNLOAD_DIR/extracted"
DATASET_URL="https://zenodo.org/records/1188976/files/Audio_Speech_Actors_01-24.zip?download=1"

mkdir -p "$DATA_DIR" "$DOWNLOAD_DIR"

echo "Synchronizing SenseVoice model files..."
(cd "$PROJECT_DIR" && ./download_models.sh)

dataset_is_complete() {
    local actor

    for actor in $(printf 'Actor_%02d\n' {1..24}); do
        if [[ ! -d "$DATA_DIR/$actor" ]] || [[ "$(find "$DATA_DIR/$actor" -maxdepth 1 -type f -iname '*.wav' | wc -l | tr -d ' ')" -ne 60 ]]; then
            return 1
        fi
    done
}

if dataset_is_complete; then
    echo "RAVDESS audio is already complete in $DATA_DIR"
    exit 0
fi

echo "Downloading RAVDESS speech audio..."
curl --fail --location --show-error --retry 3 --continue-at - \
    "$DATASET_URL" \
    --output "$ARCHIVE_PATH"

rm -rf "$EXTRACT_DIR"
mkdir -p "$EXTRACT_DIR"
unzip -q "$ARCHIVE_PATH" -d "$EXTRACT_DIR"

SOURCE_ROOT="$(find "$EXTRACT_DIR" -type d -name 'Actor_01' -print -quit)"
if [[ -z "$SOURCE_ROOT" ]]; then
    echo "Error: downloaded archive does not contain Actor_01" >&2
    exit 1
fi
SOURCE_ROOT="$(dirname "$SOURCE_ROOT")"

for actor in $(printf 'Actor_%02d\n' {1..24}); do
    if [[ ! -d "$SOURCE_ROOT/$actor" ]]; then
        echo "Error: downloaded archive is missing $actor" >&2
        exit 1
    fi
done

for actor in $(printf 'Actor_%02d\n' {1..24}); do
    rm -rf "$DATA_DIR/$actor"
    mv "$SOURCE_ROOT/$actor" "$DATA_DIR/$actor"
done

rm -rf "$EXTRACT_DIR"
echo "RAVDESS audio is ready in $DATA_DIR"