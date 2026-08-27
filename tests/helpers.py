"""Test helpers."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path


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
