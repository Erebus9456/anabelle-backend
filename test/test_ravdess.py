import os
import sys
import librosa
from tqdm import tqdm

# Import engine from parent
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from colab_compat import apply_runtime_patches

apply_runtime_patches()
from engine import AnabelleEngine

def run_static_test():
    engine = AnabelleEngine()
    base_path = os.path.join(os.path.dirname(__file__), "audio")
    report_path = os.path.join(os.path.dirname(__file__), "anabelle_static_test_report.txt")

    # RAVDESS Mapping
    ravdess_map = {
        "01": "NEUTRAL", "02": "NEUTRAL", "03": "HAPPY",
        "04": "SAD",     "05": "ANGRY",   "06": "SAD",
        "07": "ANGRY",   "08": "EXCITED"
    }

    results = []
    source_stats = {"AI_MODEL": 0, "ACOUSTIC_DNA": 0, "ERROR_RECOVERY": 0}

    print("\n--- Starting Static Accuracy Test ---")

    for actor_dir in sorted(os.listdir(base_path)):
        actor_path = os.path.join(base_path, actor_dir)
        if not os.path.isdir(actor_path): continue

        files = [f for f in os.listdir(actor_path) if f.lower().endswith(".wav")]
        for filename in tqdm(files, desc=f"Evaluating {actor_dir}"):
            parts = filename.split("-")
            if len(parts) < 3: continue

            expected_emotion = ravdess_map.get(parts[2], "NEUTRAL")
            file_full_path = os.path.join(actor_path, filename)
            
            # Load Audio (16kHz)
            audio, _ = librosa.load(file_full_path, sr=16000)
            
            # Run Inference
            prediction = engine.analyze_chunk(audio)
            predicted_emotion = prediction["emotion"]
            source = prediction.get("source", "UNKNOWN")
            
            # Track which engine made the decision
            source_stats[source] = source_stats.get(source, 0) + 1

            results.append({
                "actor": actor_dir,
                "file": filename,
                "expected": expected_emotion,
                "predicted": predicted_emotion,
                "correct": expected_emotion == predicted_emotion,
                "source": source
            })

    # --- GENERATE REPORT ---
    total = len(results)
    correct = sum(1 for r in results if r["correct"])
    accuracy = (correct / total) * 100 if total > 0 else 0

    report = []
    report.append("=" * 60)
    report.append("ANABELLE AFFECTIVE ENGINE - STATIC TEST REPORT")
    report.append("=" * 60)
    report.append(f"Total Files Tested: {total}")
    report.append(f"Passed:             {correct}")
    report.append(f"Failed:             {total - correct}")
    report.append(f"Overall Accuracy:   {accuracy:.2f}%")
    report.append("-" * 60)
    report.append("ENGINE LOGIC DISTRIBUTION")
    for src, count in source_stats.items():
        report.append(f"{src:18}: {count} files ({ (count/total)*100 :.1f}%)")
    report.append("-" * 60)
    report.append("ACCURACY BY EMOTION")
    
    for emo in sorted(set(ravdess_map.values())):
        emo_results = [r for r in results if r["expected"] == emo]
        if not emo_results: continue
        emo_correct = sum(1 for r in emo_results if r["correct"])
        emo_acc = (emo_correct / len(emo_results)) * 100
        report.append(f"{emo:10}: {emo_acc:6.2f}% ({emo_correct}/{len(emo_results)})")

    report.append("-" * 60)
    report.append("DETAILED FILE LOG")
    for r in results:
        status = "PASS" if r["correct"] else "FAIL"
        report.append(f"[{status}] [{r['source']:12}] {r['actor']}/{r['file']} | Exp: {r['expected']} | Pred: {r['predicted']}")

    # Output and Save
    final_output = "\n".join(report)
    print("\n" + "\n".join(report[:15])) # Print summary to console
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(final_output)

    print(f"\nFull report saved to: {report_path}")

if __name__ == "__main__":
    run_static_test()