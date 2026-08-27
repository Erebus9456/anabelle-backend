import argparse
import os
import sys
from collections import Counter, defaultdict

import librosa
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from colab_compat import apply_runtime_patches

apply_runtime_patches()
from engine import AnabelleEngine, SENSEVOICE_EMOTION_TAGS
from paths import get_data_dir, get_test_audio_dir

RAVDESS_MAP = {
    "01": "NEUTRAL",
    "02": "NEUTRAL",
    "03": "HAPPY",
    "04": "SAD",
    "05": "ANGRY",
    "06": "SAD",      # fearful -> avatar sad
    "07": "ANGRY",    # disgust -> avatar angry
    "08": "EXCITED",  # surprised -> avatar excited
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
):
    engine = AnabelleEngine()
    base_path = get_test_audio_dir()
    report_path = get_data_dir() / "test" / "anabelle_static_test_report.txt"

    results = []
    source_stats = Counter()
    raw_emotion_tags = Counter()
    missing_tag_samples = []

    print("\n--- Starting Static Accuracy Test ---")
    print(f"Audio directory: {base_path}")
    print(f"Language hint:   {language}")
    print(f"AI-only mode:    {ai_only}")

    if not base_path.is_dir():
        raise FileNotFoundError(
            f"Test audio not found at {base_path}. Run: python setup.py --skip-deps"
        )

    actor_dirs = sorted(p for p in base_path.iterdir() if p.is_dir())
    files_to_eval = []
    for actor_path in actor_dirs:
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
        )

        predicted_emotion = prediction["emotion"]
        source = prediction.get("source", "UNKNOWN")
        source_stats[source] += 1

        tags = prediction.get("tags") or []
        sensevoice_emotion = next((t for t in tags if t in SENSEVOICE_EMOTION_TAGS), None)
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

        results.append(
            {
                "actor": actor_dir,
                "file": filename,
                "ravdess_code": ravdess_code,
                "ravdess_label": RAVDESS_CODE_TO_LABEL.get(ravdess_code, "?"),
                "expected": expected_emotion,
                "predicted": predicted_emotion,
                "correct": expected_emotion == predicted_emotion,
                "source": source,
                "sensevoice_emotion": sensevoice_emotion or "NONE",
                "raw_text": prediction.get("raw_text", ""),
            }
        )

    total = len(results)
    correct = sum(1 for r in results if r["correct"])
    accuracy = (correct / total) * 100 if total else 0.0
    ai_results = [r for r in results if r["source"] == "AI_MODEL"]
    acoustic_results = [r for r in results if r["source"] == "ACOUSTIC_DNA"]

    report = []
    report.append("=" * 60)
    report.append("ANABELLE AFFECTIVE ENGINE - STATIC TEST REPORT")
    report.append("=" * 60)
    report.append(f"Total Files Tested: {total}")
    report.append(f"Passed:             {correct}")
    report.append(f"Failed:             {total - correct}")
    report.append(f"Overall Accuracy:   {accuracy:.2f}%")
    report.append(f"Language hint:      {language}")
    report.append(f"AI-only mode:       {ai_only}")
    report.append("-" * 60)
    report.append("ENGINE LOGIC DISTRIBUTION")
    for src in ("AI_MODEL", "ACOUSTIC_DNA", "ERROR_RECOVERY"):
        count = source_stats.get(src, 0)
        pct = (count / total) * 100 if total else 0.0
        report.append(f"{src:18}: {count} files ({pct:.1f}%)")
    report.append("-" * 60)
    report.append("SENSEVOICE RAW EMOTION TAG DISTRIBUTION")
    if raw_emotion_tags:
        for tag, count in raw_emotion_tags.most_common():
            pct = (count / total) * 100 if total else 0.0
            report.append(f"{tag:14}: {count:4} ({pct:.1f}%)")
    else:
        report.append("No SenseVoice emotion tags detected.")
    report.append("-" * 60)
    report.extend(summarize_accuracy(results, "HYBRID ACCURACY (current pipeline)"))
    if ai_results:
        report.append("")
        report.extend(summarize_accuracy(ai_results, "AI_MODEL ONLY (where tag was parsed)"))
    if acoustic_results:
        report.append("")
        report.extend(
            summarize_accuracy(acoustic_results, "ACOUSTIC_DNA ONLY (fallback cases)")
        )
    report.append("-" * 60)
    report.append("ACCURACY BY RAVDESS EMOTION CODE")
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
        report.append("-" * 60)
        report.append("SAMPLE FILES WITH NO PARSED EMOTION TAG")
        for sample in missing_tag_samples:
            report.append(f"{sample['file']} | expected={sample['expected']}")
            report.append(f"  tags={sample['tags']}")
            report.append(f"  raw={sample['raw_text'][:160]}")

    if diagnose:
        report.append("-" * 60)
        report.append("DIAGNOSTIC SAMPLE (first 24 files)")
        for row in results[:24]:
            report.append(
                f"{row['file']} | ravdess={row['ravdess_label']} | "
                f"exp={row['expected']} | pred={row['predicted']} | "
                f"sv_tag={row['sensevoice_emotion']} | source={row['source']}"
            )
            report.append(f"  raw={row['raw_text'][:160]}")

    report.append("-" * 60)
    report.append("DETAILED FILE LOG")
    for row in results:
        status = "PASS" if row["correct"] else "FAIL"
        report.append(
            f"[{status}] [{row['source']:12}] {row['actor']}/{row['file']} | "
            f"Exp: {row['expected']} | Pred: {row['predicted']} | SV: {row['sensevoice_emotion']}"
        )

    final_output = "\n".join(report)
    print("\n" + "\n".join(report[:20]))
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as handle:
        handle.write(final_output)

    print(f"\nFull report saved to: {report_path}")


def main():
    parser = argparse.ArgumentParser(description="RAVDESS static accuracy benchmark")
    parser.add_argument("--diagnose", action="store_true", help="Include extra diagnostic samples")
    parser.add_argument("--sample-limit", type=int, default=0, help="Limit number of files tested")
    parser.add_argument(
        "--language",
        default="en",
        help="Language hint passed to SenseVoice (default: en for RAVDESS)",
    )
    parser.add_argument(
        "--ai-only",
        action="store_true",
        help="Disable acoustic fallback to measure pure SenseVoice tag accuracy",
    )
    args = parser.parse_args()
    run_static_test(
        diagnose=args.diagnose,
        sample_limit=args.sample_limit,
        language=args.language,
        ai_only=args.ai_only,
    )


if __name__ == "__main__":
    main()
