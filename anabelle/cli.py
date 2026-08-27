"""Command-line interface for ANABELLE backend."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from anabelle.utils.paths import PROJECT_ROOT


def serve() -> None:
    import uvicorn

    host = os.environ.get("ANABELLE_HOST", "0.0.0.0")
    port = int(os.environ.get("ANABELLE_PORT", "8000"))
    uvicorn.run("anabelle.app:app", host=host, port=port, reload=False)


def test(extra_args: list[str] | None = None) -> None:
    test_script = PROJECT_ROOT / "tests" / "benchmark" / "test_ravdess.py"
    command = [sys.executable, str(test_script)]
    if extra_args:
        command.extend(extra_args)
    subprocess.run(command, check=True, cwd=PROJECT_ROOT)


def main() -> None:
    parser = argparse.ArgumentParser(description="ANABELLE backend CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("serve", help="Start the FastAPI WebSocket gateway")
    test_parser = subparsers.add_parser("test", help="Run the RAVDESS static benchmark")
    test_parser.add_argument("--diagnose", action="store_true")
    test_parser.add_argument("--sample-limit", type=int, default=0)
    test_parser.add_argument("--language", default="en")
    test_parser.add_argument("--ai-only", action="store_true")
    test_parser.add_argument("--no-ser", action="store_true")

    args = parser.parse_args()
    if args.command == "serve":
        serve()
    elif args.command == "test":
        extra: list[str] = []
        if args.diagnose:
            extra.append("--diagnose")
        if args.sample_limit:
            extra.extend(["--sample-limit", str(args.sample_limit)])
        if args.language != "en":
            extra.extend(["--language", args.language])
        if args.ai_only:
            extra.append("--ai-only")
        if args.no_ser:
            extra.append("--no-ser")
        test(extra)


if __name__ == "__main__":
    main()
