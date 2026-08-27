#!/bin/bash
# Wrapper — downloads models + RAVDESS test data via Python.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "Synchronizing SenseVoice model files..."
python3 "$PROJECT_DIR/download_models.py"
python3 "$PROJECT_DIR/download_test_data.py"
