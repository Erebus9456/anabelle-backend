"""Command-line interface for ANABELLE backend."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

from anabelle.config import InferenceConfig
from anabelle.utils.paths import PROJECT_ROOT


def _add_inference_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--backend",
        choices=["pytorch", "onnx"],
        default=os.environ.get("ANABELLE_BACKEND", "pytorch"),
        help="SenseVoice runtime backend (default: pytorch)",
    )
    parser.add_argument(
        "--quantize",
        choices=["fp32", "fp16", "int8"],
        default=os.environ.get("ANABELLE_QUANTIZE", "fp32"),
        help="Model precision: fp32/fp16 (PyTorch) or int8 (ONNX)",
    )
    parser.add_argument(
        "--vad",
        choices=["rms", "silero", "off"],
        default=os.environ.get("ANABELLE_VAD", "rms"),
        help="Voice-activity gate before heavy inference",
    )
    parser.add_argument(
        "--vad-rms",
        type=float,
        default=float(os.environ.get("ANABELLE_VAD_RMS", "0.02")),
        help="RMS threshold for VAD gate (default: 0.02)",
    )
    parser.add_argument(
        "--no-ser",
        action="store_true",
        help="Disable emotion2vec SER fallback",
    )
    parser.add_argument(
        "--ser-mode",
        choices=["always", "smart", "off"],
        default=os.environ.get("ANABELLE_SER_MODE", "smart"),
        help="SER policy: always (run on every chunk), smart (skip when text/low RMS), off (disable)",
    )
    parser.add_argument(
        "--no-semantic",
        action="store_true",
        help="Disable transcript keyword emotion shortcut",
    )
    parser.add_argument(
        "--no-smoothing",
        action="store_true",
        help="Disable emotion hysteresis on WebSocket stream",
    )
    parser.add_argument(
        "--no-dual-path",
        action="store_true",
        help="Disable reflex+emotion dual WebSocket responses",
    )
    parser.add_argument(
        "--min-chunk-interval",
        type=float,
        default=float(os.environ.get("ANABELLE_MIN_CHUNK_INTERVAL", "0.5")),
        help="Minimum seconds between audio chunks (default: 0.5)",
    )


def _config_from_args(args: argparse.Namespace) -> InferenceConfig:
    base = InferenceConfig.from_env()
    return replace(
        base,
        backend=args.backend,
        quantize=args.quantize,
        vad_mode=args.vad,
        vad_rms_threshold=args.vad_rms,
        enable_ser=not args.no_ser,
        ser_mode=args.ser_mode,
        enable_semantic=not args.no_semantic,
        enable_smoothing=not args.no_smoothing,
        dual_path=not args.no_dual_path,
        min_chunk_interval=args.min_chunk_interval,
    )


def serve(config: InferenceConfig) -> None:
    import uvicorn

    config.apply_to_env()
    host = os.environ.get("ANABELLE_HOST", "0.0.0.0")
    port = int(os.environ.get("ANABELLE_PORT", "8000"))
    uvicorn.run("anabelle.app:app", host=host, port=port, reload=False)


def test(config: InferenceConfig, extra_args: list[str] | None = None) -> None:
    config.apply_to_env()
    test_script = PROJECT_ROOT / "tests" / "benchmark" / "test_ravdess.py"
    command = [sys.executable, str(test_script)]
    if extra_args:
        command.extend(extra_args)
    subprocess.run(command, check=True, cwd=PROJECT_ROOT)


def main() -> None:
    parser = argparse.ArgumentParser(description="ANABELLE backend CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    serve_parser = subparsers.add_parser("serve", help="Start the FastAPI WebSocket gateway")
    _add_inference_args(serve_parser)

    test_parser = subparsers.add_parser("test", help="Run the RAVDESS static benchmark")
    _add_inference_args(test_parser)
    test_parser.add_argument("--diagnose", action="store_true")
    test_parser.add_argument("--sample-limit", type=int, default=0)
    test_parser.add_argument("--language", default="en")
    test_parser.add_argument("--ai-only", action="store_true")

    args = parser.parse_args()
    config = _config_from_args(args)

    if args.command == "serve":
        serve(config)
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
        if args.backend != "pytorch":
            extra.extend(["--backend", args.backend])
        if args.quantize != "fp32":
            extra.extend(["--quantize", args.quantize])
        if args.vad != "rms":
            extra.extend(["--vad", args.vad])
        if args.vad_rms != 0.02:
            extra.extend(["--vad-rms", str(args.vad_rms)])
        if args.no_semantic:
            extra.append("--no-semantic")
        test(config, extra)


if __name__ == "__main__":
    main()
