"""RAVDESS static accuracy benchmark."""

from __future__ import annotations

import argparse
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import librosa
from tqdm import tqdm

from anabelle.engine.core import SENSEVOICE_EMOTION_TAGS, AnabelleEngine
from anabelle.utils.compat import apply_runtime_patches
from anabelle.utils.paths import get_test_audio_dir, get_test_reports_dir
from tests.helpers import build_ravdess_report_path

apply_runtime_patches()

RAVDESS_MAP = {
    "01": "NEUTRAL",
    "02": "NEUTRAL",
    "03": "HAPPY",
    "04": "SAD",
    "05": "ANGRY",
    "06": "SAD",
    "07": "ANGRY",
    "08": "EXCITED",
}

RAVDESS_CODE_TO_LABEL = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised",
}


def summarize_accuracy(results, label: str) -> list[str]:
    total = len(results)
    correct = sum(1 for r in results if r["correct"])
    accuracy = (correct / total) * 100 if total else 0.0
    lines = [
        f"{label}",
        f"  Overall: {accuracy:.2f}% ({correct}/{total})",
    ]
    by_expected = defaultdict(list)
    for row in results:
        by_expected[row["expected"]].append(row)

    for emotion in sorted(by_expected):
        rows = by_expected[emotion]
        hits = sum(1 for r in rows if r["correct"])
        lines.append(f"  {emotion:8}: {(hits / len(rows)) * 100:6.2f}% ({hits}/{len(rows)})")
    return lines


def run_static_test(
    *,
    diagnose: bool = False,
    sample_limit: int = 0,
    language: str = "en",
    ai_only: bool = False,
    no_ser: bool = False,
):
    engine = AnabelleEngine(enable_ser=not no_ser and not ai_only)
    base_path = get_test_audio_dir()
    report_path = build_ravdess_report_path(
        get_test_reports_dir(),
        language=language,
        ai_only=ai_only,
        no_ser=no_ser,
        diagnose=diagnose,
        sample_limit=sample_limit,
    )

    results = []
    source_stats = Counter()
    raw_emotion_tags = Counter()
    ser_label_stats = Counter()
    missing_tag_samples = []

    print("\n--- Starting Static Accuracy Test ---")
    print(f"Audio directory: {base_path}")
    print(f"Language hint:   {language}")
    print(f"AI-only mode:    {ai_only}")
    print(f"SER requested:   {not no_ser and not ai_only}")
    print(f"SER loaded:      {engine.ser_available}")
    print(f"Report file:     {report_path}")

    if not base_path.is_dir():
        raise FileNotFoundError(
            f"Test audio not found at {base_path}. Run: python setup.py --skip-deps"
        )

    files_to_eval = []
    for actor_path in sorted(p for p in base_path.iterdir() if p.is_dir()):
        for filename in sorted(os.listdir(actor_path)):
            if filename.lower().endswith(".wav"):
                files_to_eval.append((actor_path.name, filename))

    if sample_limit > 0:
        files_to_eval = files_to_eval[:sample_limit]

    for actor_dir, filename in tqdm(files_to_eval, desc="Evaluating RAVDESS"):
        parts = filename.split("-")
        if len(parts) < 3:
            continue

        ravdess_code = parts[2]
        expected_emotion = RAVDESS_MAP.get(ravdess_code, "NEUTRAL")
        file_full_path = base_path / actor_dir / filename

        audio, _ = librosa.load(str(file_full_path), sr=16000)
        prediction = engine.analyze_chunk(
            audio,
            language=language,
            allow_acoustic_fallback=not ai_only,
            allow_ser_fallback=not no_ser and not ai_only,
        )

        source = prediction.get("source", "UNKNOWN")
        source_stats[source] += 1

        tags = prediction.get("tags") or []
        sensevoice_emotion = prediction.get("sensevoice_emotion") or next(
            (t for t in tags if t in SENSEVOICE_EMOTION_TAGS), None
        )
        if sensevoice_emotion:
            raw_emotion_tags[sensevoice_emotion] += 1
        elif len(missing_tag_samples) < 12:
            missing_tag_samples.append(
                {
                    "file": f"{actor_dir}/{filename}",
                    "expected": expected_emotion,
                    "raw_text": prediction.get("raw_text", ""),
                    "tags": tags,
                }
            )

        ser_label = prediction.get("ser_label")
        if ser_label:
            ser_label_stats[ser_label] += 1

        results.append(
            {
                "actor": actor_dir,
                "file": filename,
                "ravdess_code": ravdess_code,
                "ravdess_label": RAVDESS_CODE_TO_LABEL.get(ravdess_code, "?"),
                "expected": expected_emotion,
                "predicted": prediction["emotion"],
                "correct": expected_emotion == prediction["emotion"],
                "source": source,
                "sensevoice_emotion": sensevoice_emotion or "NONE",
                "ser_label": ser_label or "-",
                "raw_text": prediction.get("raw_text", ""),
            }
        )

    total = len(results)
    correct = sum(1 for r in results if r["correct"])
    accuracy = (correct / total) * 100 if total else 0.0
    ai_results = [r for r in results if r["source"] == "AI_MODEL"]
    ser_results = [r for r in results if r["source"] == "SER_MODEL"]
    acoustic_results = [r for r in results if r["source"] == "ACOUSTIC_DNA"]

    report = [
        "=" * 60,
        "ANABELLE AFFECTIVE ENGINE - STATIC TEST REPORT",
        "=" * 60,
        f"Generated at:       {datetime.now().isoformat(timespec='seconds')}",
        f"Report file:        {report_path}",
        f"Total Files Tested: {total}",
        f"Passed:             {correct}",
        f"Failed:             {total - correct}",
        f"Overall Accuracy:   {accuracy:.2f}%",
        f"Language hint:      {language}",
        f"AI-only mode:       {ai_only}",
        f"SER requested:      {not no_ser and not ai_only}",
        f"SER loaded:         {engine.ser_available}",
        "-" * 60,
        "ENGINE LOGIC DISTRIBUTION",
    ]
    for src in ("AI_MODEL", "SER_MODEL", "ACOUSTIC_DNA", "ERROR_RECOVERY"):
        count = source_stats.get(src, 0)
        pct = (count / total) * 100 if total else 0.0
        report.append(f"{src:18}: {count} files ({pct:.1f}%)")

    report.extend(["-" * 60, "SENSEVOICE RAW EMOTION TAG DISTRIBUTION"])
    if raw_emotion_tags:
        for tag, count in raw_emotion_tags.most_common():
            pct = (count / total) * 100 if total else 0.0
            report.append(f"{tag:14}: {count:4} ({pct:.1f}%)")
    else:
        report.append("No SenseVoice emotion tags detected.")

    report.extend(["-" * 60, "EMOTION2VEC LABEL DISTRIBUTION (SER_MODEL cases)"])
    if ser_label_stats:
        for label, count in ser_label_stats.most_common():
            report.append(f"{label:14}: {count:4}")
    else:
        report.append("No SER_MODEL predictions.")

    report.extend(["-" * 60])
    report.extend(summarize_accuracy(results, "HYBRID ACCURACY (current pipeline)"))
    if ai_results:
        report.append("")
        report.extend(summarize_accuracy(ai_results, "AI_MODEL ONLY"))
    if ser_results:
        report.append("")
        report.extend(summarize_accuracy(ser_results, "SER_MODEL ONLY"))
    if acoustic_results:
        report.append("")
        report.extend(summarize_accuracy(acoustic_results, "ACOUSTIC_DNA ONLY"))

    report.extend(["-" * 60, "ACCURACY BY RAVDESS EMOTION CODE"])
    for code in sorted(RAVDESS_CODE_TO_LABEL):
        label = RAVDESS_CODE_TO_LABEL[code]
        mapped = RAVDESS_MAP[code]
        code_rows = [r for r in results if r["ravdess_code"] == code]
        if not code_rows:
            continue
        hits = sum(1 for r in code_rows if r["correct"])
        report.append(
            f"{code} {label:10} -> {mapped:8}: {(hits / len(code_rows)) * 100:6.2f}% ({hits}/{len(code_rows)})"
        )

    if missing_tag_samples:
        report.extend(["-" * 60, "SAMPLE FILES WITH NO PARSED EMOTION TAG"])
        for sample in missing_tag_samples:
            report.append(f"{sample['file']} | expected={sample['expected']}")
            report.append(f"  tags={sample['tags']}")
            report.append(f"  raw={sample['raw_text'][:160]}")

    if diagnose:
        report.extend(["-" * 60, "DIAGNOSTIC SAMPLE (first 24 files)"])
        for row in results[:24]:
            report.append(
                f"{row['file']} | ravdess={row['ravdess_label']} | "
                f"exp={row['expected']} | pred={row['predicted']} | "
                f"sv_tag={row['sensevoice_emotion']} | source={row['source']}"
            )
            report.append(f"  raw={row['raw_text'][:160]}")

    report.extend(["-" * 60, "DETAILED FILE LOG"])
    for row in results:
        status = "PASS" if row["correct"] else "FAIL"
        report.append(
            f"[{status}] [{row['source']:12}] {row['actor']}/{row['file']} | "
            f"Exp: {row['expected']} | Pred: {row['predicted']} | "
            f"SV: {row['sensevoice_emotion']} | SER: {row['ser_label']}"
        )

    final_output = "\n".join(report)
    print("\n" + "\n".join(report[:20]))
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(final_output, encoding="utf-8")
    print(f"\nFull report saved to: {report_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="RAVDESS static accuracy benchmark")
    parser.add_argument("--diagnose", action="store_true")
    parser.add_argument("--sample-limit", type=int, default=0)
    parser.add_argument("--language", default="en")
    parser.add_argument("--ai-only", action="store_true")
    parser.add_argument("--no-ser", action="store_true")
    args = parser.parse_args()
    run_static_test(
        diagnose=args.diagnose,
        sample_limit=args.sample_limit,
        language=args.language,
        ai_only=args.ai_only,
        no_ser=args.no_ser,
    )


if __name__ == "__main__":
    main()
