#!/bin/bash
# Wrapper — delegates to the cross-platform Python downloader.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python3 "$SCRIPT_DIR/download_models.py"
