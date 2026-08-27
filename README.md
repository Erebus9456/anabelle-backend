# ANABELLE Backend

Enterprise affective inference engine for the ANABELLE avatar. This service receives live audio over WebSocket, runs **SenseVoiceSmall** emotion detection, and falls back to acoustic heuristics when the model does not emit a usable tag.

---

## Quick Start

### One-click setup (recommended)

```bash
git clone <repo-url> anabelle-backend && cd anabelle-backend
python setup.py
```

This single command:

1. Installs PyTorch for your platform (CUDA on Linux/Colab, MPS-friendly build on Mac)
2. Pins NumPy to a compatible version
3. Installs all project dependencies
4. Downloads the ~1 GB SenseVoiceSmall model weights
5. Downloads the RAVDESS speech test dataset (24 actors × 60 clips)
6. Verifies FunASR model registration

### Run the gateway

```bash
python run.py serve
```

WebSocket endpoint: `ws://localhost:8000/ws/anabelle`

### Run static accuracy tests

```bash
python run.py test
python run.py test --diagnose --ai-only   # measure pure SenseVoice tag accuracy
python run.py test --sample-limit 48      # quick smoke test
```

Report written to `<data-dir>/test/anabelle_static_test_report.txt`.

**Interpreting results:** RAVDESS uses acted studio speech. SenseVoice was trained on real-world multilingual audio, so raw tag accuracy on RAVDESS is often **45–65%** even when the pipeline is correct. The report now separates:

- **Hybrid accuracy** — current production pipeline (AI tags + acoustic fallback)
- **AI_MODEL only** — files where SenseVoice emitted a parseable emotion tag
- **Raw tag distribution** — shows if the model is tagging `EMO_UNKNOWN` or `NEUTRAL` too often

---

## Google Colab

1. In Colab, select **Runtime → Change runtime type → GPU**
2. Clone once, then **pull code updates** without re-downloading models:

```python
# First time
!git clone <repo-url> anabelle-backend
%cd anabelle-backend
!python setup.py --colab

# Later sessions — pull latest code only (models/tests stay cached)
%cd /content/anabelle-backend
!git pull
!python setup.py --colab --skip-models --skip-test-data
```

3. Run tests and start the server:

```python
!python run.py test
!python run.py serve   # keep this cell running
```

Or open [`Anabelle_Colab.ipynb`](Anabelle_Colab.ipynb).

Use Colab's port forwarding or `ngrok` to expose port `8000` to your frontend.

### Persistent data on Colab

Models and test audio are stored **outside the git repo** so `git pull` never triggers re-downloads:

| Asset | Default Colab path |
|-------|-------------------|
| SenseVoice weights | `/content/anabelle-data/models/SenseVoiceSmall/` |
| RAVDESS test audio | `/content/anabelle-data/test/audio/` |
| Test report | `/content/anabelle-data/test/anabelle_static_test_report.txt` |

Override with an environment variable:

```python
import os
os.environ["ANABELLE_DATA_DIR"] = "/content/my-cache"
!python setup.py --colab
```

---

## Project Structure

```
anabelle-backend/
├── setup.py              # One-click installer (local + Colab)
├── run.py                  # serve | test entry point
├── main.py                 # FastAPI WebSocket gateway
├── engine.py               # SenseVoice + acoustic fallback logic
├── device_utils.py         # CUDA / MPS / CPU detection
├── paths.py                # Persistent data paths (Colab vs local)
├── colab_compat.py         # Colab + NumPy 2.x compatibility shims
├── constraints-py313.txt   # Wheel pins for Python 3.13 (tokenizers)
├── constraints-colab.txt   # Wheel pins for Colab Python 3.10–3.12
├── download_models.py      # Hugging Face model downloader
├── download_test_data.py   # RAVDESS dataset downloader
├── requirements.txt        # Local / Mac pinned deps
├── requirements-colab.txt  # Colab-specific deps (torch via setup)
├── Anabelle_Colab.ipynb    # Ready-made Colab notebook
└── test/
    ├── test_ravdess.py     # Static accuracy benchmark
    └── audio/              # RAVDESS clips (local dev; Colab uses /content/anabelle-data)
```

---

## Architecture

```
Browser (React) ──WebSocket──► main.py (FastAPI)
                                   │
                                   ▼
                              engine.py
                         ┌────────┴────────┐
                         │                 │
                   SenseVoiceSmall    Acoustic DNA
                   (GPU if avail.)    (RMS + ZCR fallback)
                         │                 │
                         └────────┬────────┘
                                   ▼
                          { emotion, source, raw_text }
```

**Inference device priority:** CUDA → Apple MPS → CPU

The engine uses `@torch.inference_mode()`, enables cuDNN benchmark mode on CUDA, and passes `ngpu=1` to FunASR when a GPU is present.

---

## Requirements (Definitive Stack)

Three stacks must be present and **version-aligned**:

### Stack 1 — AI Inference Engine

| Package | Version | Role |
|---------|---------|------|
| `funasr` | `1.4.4` | Core SenseVoice framework |
| `modelscope` | latest | Secure model downloading |
| `torch` | `2.2.2` (Mac) / `2.5+` (Colab CUDA) | Math engine |
| `transformers` | `< 4.45` (Py ≤ 3.12) / `4.46+` (Py 3.13) | Audio ↔ text coordination |
| `tokenizers` | `≥ 0.21` (Py 3.13 wheels) | Required by transformers; must not build from source |
| `WeTextProcessing` | `≥ 1.0.3` | Text normalization / ITN (required for model registration) |

### Stack 2 — Acoustic DNA Layer

| Package | Version | Role |
|---------|---------|------|
| `numpy` | **`1.26.4`** (Python ≤ 3.12) | Must stay on NumPy 1.x |
| `librosa` | `0.11.0` | WAV loading, ZCR analysis |
| `numba` | `≥ 0.59` (Py 3.12+) | Accelerates audio math |
| `scipy` | `≥ 1.11` | Signal processing |

### Stack 3 — Communication Gateway

| Package | Version | Role |
|---------|---------|------|
| `fastapi` | `0.141.1` | Web server |
| `uvicorn` | `0.52.4` | ASGI runner |
| `websockets` | `17.1` | Live binary audio stream |

> **Do not** `pip install` these manually unless you know your Python version. Always use `python setup.py` so torch/numpy are resolved correctly.

---

## Known Version Conflicts

### A. The NumPy 2.0 Conflict

In June 2024, NumPy 2.0 shipped breaking ABI changes. Most AI wheels (PyTorch, FunASR, Librosa) were compiled against NumPy 1.x. Installing NumPy 2.x causes instant import/runtime crashes.

**Rule:** `numpy==1.26.4` on Python 3.10–3.12. `setup.py` enforces this automatically.

### B. The Python 3.12 / 3.13 "Numba" Trap

`librosa` depends on `numba`, which lags new Python releases.

| Python | numba requirement | numpy |
|--------|-------------------|-------|
| 3.10–3.11 | `≥ 0.58` | `1.26.4` |
| 3.12 | `≥ 0.59` | `1.26.4` |
| 3.13+ (Colab) | latest available | `2.x` (1.x unavailable) |

On Python 3.13, NumPy 1.x cannot be installed, creating a compatibility circle. `colab_compat.py` applies best-effort shims and logs warnings. **For production, use Python 3.10 or 3.11.**

### C. The "Model Registration" Failure

FunASR discovers models via registration keys (e.g. `SenseVoiceSmall`). If a sub-dependency like text normalization is missing, registration fails silently. FunASR then treats your local `models/SenseVoiceSmall` path as a remote ModelScope ID and returns **404 Not Found**.

**Fix:** `setup.py` installs `WeTextProcessing`, and `colab_compat.py` force-imports `funasr.models.sense_voice.model` before `AutoModel` is constructed.

### D. The `tokenizers` Build Failure (Python 3.13 / Colab)

`transformers` depends on `tokenizers`. Older `tokenizers` releases have no Python 3.13 wheels and pip tries to compile from Rust source, which fails in Colab.

**Fix:** `setup.py` pre-installs `tokenizers>=0.21` with `--only-binary=tokenizers`, then installs `funasr` with `--no-deps` so pip cannot trigger a source build.

---

## GPU Support

| Platform | Device string | Notes |
|----------|--------------|-------|
| NVIDIA (Colab, Linux) | `cuda` | FP16 when compute capability ≥ 7.0 |
| Apple Silicon | `mps` | Metal acceleration |
| Fallback | `cpu` | Functional but slower |

Verify GPU at runtime:

```bash
python -c "from device_utils import get_device_info; print(get_device_info())"
```

In Colab, ensure **Runtime → Change runtime type → GPU** is selected before running `setup.py`.

---

## WebSocket Protocol

**Endpoint:** `GET /ws/anabelle` (WebSocket upgrade)

| Direction | Format | Description |
|-----------|--------|-------------|
| Client → Server | Binary `float32` PCM | 16 kHz mono audio chunk (0.5–2 s ideal) |
| Server → Client | JSON text | `{ "emotion": "HAPPY", "source": "AI_MODEL", "raw_text": "<|HAPPY|> hello" }` |

**Emotion values:** `HAPPY`, `SAD`, `ANGRY`, `NEUTRAL`, `EXCITED`

**Source values:** `AI_MODEL`, `ACOUSTIC_DNA`, `ERROR_RECOVERY`

---

## Setup Options

```bash
python setup.py                  # Full setup (auto-detect platform)
python setup.py --colab          # Force Colab profile
python setup.py --skip-deps      # Skip pip installs
python setup.py --skip-models    # Skip model download
python setup.py --skip-test-data # Skip RAVDESS download
```

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| `ImportError: numpy` ABI error | NumPy 2.x installed | `pip install numpy==1.26.4` (Py ≤ 3.12) or re-run `setup.py` |
| 404 on model load | Registration failure | Re-run `setup.py`; check `WeTextProcessing` installed |
| Slow inference | Running on CPU | Enable GPU runtime (Colab) or verify CUDA drivers |
| `numba` import error | Python too new | Use Python 3.11, or upgrade numba |
| Failed building wheel for `tokenizers` | Python 3.13, old pip resolve order | Pull latest code and re-run `python setup.py --colab` |
| Re-downloads models on every pull | Assets inside repo dir | Use default `/content/anabelle-data` (automatic on Colab) |
| Empty model folder | Download interrupted | Re-run `python download_models.py` (resumes partial files) |

---

## License

Proprietary — ANABELLE / LUKYX Engine.
