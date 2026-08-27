"""Shared helpers for test scripts run from the test/ directory."""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def bootstrap_project_root() -> Path:
    root = str(PROJECT_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    return PROJECT_ROOT


def build_ravdess_report_path(
    reports_dir: Path,
    *,
    language: str,
    ai_only: bool,
    no_ser: bool,
    diagnose: bool,
    sample_limit: int,
) -> Path:
    """Build a unique report filename from benchmark parameters."""
    mode = "ai-only" if ai_only else "hybrid"
    ser = "no-ser" if (no_ser or ai_only) else "ser"
    scope = f"n{sample_limit}" if sample_limit > 0 else "full"
    diag = "diag" if diagnose else "nodiag"

    slug = f"lang-{language}_{mode}_{ser}_{scope}_{diag}"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    reports_dir.mkdir(parents=True, exist_ok=True)
    return reports_dir / f"ravdess_{slug}_{timestamp}.txt"
