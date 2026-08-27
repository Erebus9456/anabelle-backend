#!/usr/bin/env python3
"""Single entry point for running the ANABELLE gateway or static tests."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent


def serve() -> None:
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)


def test() -> None:
    test_script = PROJECT_ROOT / "test" / "test_ravdess.py"
    subprocess.run([sys.executable, str(test_script)], check=True, cwd=PROJECT_ROOT)


def main() -> None:
    parser = argparse.ArgumentParser(description="ANABELLE runtime commands")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("serve", help="Start the FastAPI WebSocket gateway")
    subparsers.add_parser("test", help="Run the RAVDESS static accuracy test")

    args = parser.parse_args()
    if args.command == "serve":
        serve()
    elif args.command == "test":
        test()


if __name__ == "__main__":
    main()
