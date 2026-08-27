# ANABELLE Backend

Hybrid affective inference service for the ANABELLE avatar. Receives live audio over WebSocket, runs a **3-tier emotion pipeline** (SenseVoice → emotion2vec → acoustic fallback), and returns avatar-ready emotion states.

> **Stack:** FastAPI + Uvicorn (not Flask). GPU-accelerated via PyTorch (CUDA / Apple MPS).

---

## Table of Contents

- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Architecture](#architecture)
- [API Reference](#api-reference)
- [Testing](#testing)
- [Environment Variables](#environment-variables)
- [Colab Workflow](#colab-workflow)
- [Dependencies](#dependencies)
- [Troubleshooting](#troubleshooting)

---

## Quick Start

```bash
git clone https://github.com/Erebus9456/anabelle-backend && cd anabelle-backend
python setup.py                 # install deps + download models + RAVDESS data
python run.py serve             # start gateway on :8000
python run.py test              # run RAVDESS benchmark (~80% hybrid accuracy)
```

For CPU optimization (ONNX Runtime + INT8 quantization):

```bash
python setup.py --with-onnx     # install ONNX Runtime for faster CPU inference
export ANABELLE_BACKEND=onnx
export ANABELLE_QUANTIZE=int8
python run.py serve
```

Alternative entry points:

```bash
python -m anabelle serve
uvicorn anabelle.app:app --host 0.0.0.0 --port 8000
```

---

## Project Structure

```
anabelle-backend/
├── README.md
├── pyproject.toml              # package metadata
├── setup.py                    # → scripts/setup.py
├── run.py                      # → anabelle.cli
├── main.py                     # → anabelle.app (uvicorn compat)
│
├── anabelle/                   # Application package
│   ├── app.py                  # FastAPI gateway + WebSocket
│   ├── cli.py                  # serve | test commands
│   ├── engine/
│   │   ├── core.py             # AnabelleEngine (3-tier pipeline)
│   │   └── ser.py              # emotion2vec+ SER model
│   └── utils/
│       ├── paths.py            # data dir resolution (Colab vs local)
│       ├── device.py           # CUDA / MPS / CPU detection
│       └── compat.py           # Colab + FunASR registration shims
│
├── scripts/
│   ├── setup.py                # one-click installer
│   ├── download_models.py      # SenseVoice weights (~1 GB)
│   ├── download_test_data.py   # RAVDESS dataset
│   └── shell/                  # bash wrappers
│
├── requirements/
│   ├── base.txt                # local deps
│   ├── colab.txt               # Colab deps
│   ├── funasr-runtime.txt      # funasr --no-deps companions
│   ├── constraints-py313.txt
│   └── constraints-colab.txt
│
├── tests/
│   ├── benchmark/
│   │   └── test_ravdess.py     # static accuracy benchmark
│   └── integration/
│       └── test_ser.py         # emotion2vec smoke test
│
└── notebooks/
    └── Anabelle_Colab.ipynb
```

### Data directories (outside git)

| Environment | Root | Models | Test audio | Reports |
|-------------|------|--------|------------|---------|
| **Colab** | `/content/anabelle-data` | `.../models/SenseVoiceSmall/` | `.../test/audio/` | `.../test/reports/` |
| **Local** | repo root | `models/SenseVoiceSmall/` | `test/audio/` | `test/reports/` |

Override with `ANABELLE_DATA_DIR=/path/to/cache`.

---

## Architecture

### System overview

```
┌─────────────┐     WebSocket      ┌──────────────────────────────────────────┐
│  React App  │ ────────────────► │  anabelle/app.py  (FastAPI + Uvicorn)      │
│  (browser)  │ ◄──────────────── │  GET /health  ·  WS /ws/anabelle           │
└─────────────┘   JSON per chunk  └──────────────────┬───────────────────────┘
                                                     │
                                                     ▼
                                    ┌────────────────────────────────────────┐
                                    │  anabelle/engine/core.py               │
                                    │  AnabelleEngine.analyze_chunk()        │
                                    └──────────────────┬─────────────────────┘
                                                       │
              ┌────────────────────────────────────────┼────────────────────────┐
              │                    │                   │                        │
              ▼                    ▼                   ▼                        ▼
     ┌────────────────┐  ┌────────────────┐  ┌──────────────┐      ┌──────────────┐
     │  SenseVoice    │  │  emotion2vec+  │  │  Acoustic    │      │   ERROR      │
     │  Small         │  │  (SER)         │  │  DNA         │      │   RECOVERY   │
     │  AI_MODEL      │  │  SER_MODEL     │  │  ACUSTIC_DNA │      │              │
     └────────────────┘  └────────────────┘  └──────────────┘      └──────────────┘
```

### Decision tiers

| Tier | Source | When | RAVDESS accuracy |
|------|--------|------|------------------|
| 1 | `AI_MODEL` | SenseVoice emits confident tag (HAPPY, ANGRY, NEUTRAL, …) | ~60% |
| 2 | `SER_MODEL` | SenseVoice returns `EMO_UNKNOWN` (~63% of clips) | ~93% |
| 3 | `ACOUSTIC_DNA` | SER confidence too low | fallback |
| 4 | `ERROR_RECOVERY` | Inference exception | safe default |

**Production hybrid accuracy on RAVDESS: ~81%** (1440 clips, GPU, SER enabled).

### Inference device priority

CUDA → Apple MPS → CPU. Set via PyTorch auto-detection in `anabelle/utils/device.py`.

---

## API Reference

Base URL: `http://<host>:8000` (default port **8000**).

### `GET /health`

Readiness probe for load balancers and monitoring.

**Response** `200 OK`:

```json
{
  "status": "ok",
  "device": "cuda",
  "device_label": "Tesla T4",
  "ser_available": true,
  "model_dir": "/content/anabelle-data/models/SenseVoiceSmall"
}
```

| Field | Type | Description |
|-------|------|-------------|
| `status` | string | `"ok"` when engine is loaded |
| `device` | string | `"cuda"`, `"mps"`, or `"cpu"` |
| `device_label` | string | Human-readable device name |
| `ser_available` | boolean | emotion2vec+ loaded successfully |
| `model_dir` | string | Path to SenseVoice weights |

---

### `WebSocket /ws/anabelle`

Live emotion inference stream for the avatar.

#### Connection

```
ws://<host>:8000/ws/anabelle
```

No authentication in current version. Accepts one binary audio message per inference cycle.

#### Client → Server

| Property | Value |
|----------|-------|
| Format | Binary |
| Encoding | IEEE 754 **float32** little-endian PCM |
| Sample rate | **16 000 Hz** |
| Channels | **Mono** |
| Chunk size | **0.5 – 2.0 seconds** recommended |

Example (JavaScript):

```javascript
const ws = new WebSocket("ws://localhost:8000/ws/anabelle");
ws.binaryType = "arraybuffer";

// float32Array: 16 kHz mono PCM in [-1.0, 1.0]
ws.send(float32Array.buffer);
```

#### Server → Client

JSON text frame per audio chunk:

```json
{
  "emotion": "HAPPY",
  "source": "SER_MODEL",
  "raw_text": "<|en|><|EMO_UNKNOWN|><|Speech|><|withitn|>kids are talking by the door",
  "tags": ["EN", "EMO_UNKNOWN", "SPEECH", "WITHITN"],
  "sensevoice_emotion": "EMO_UNKNOWN",
  "ser_label": "happy",
  "ser_confidence": 0.87
}
```

| Field | Type | Description |
|-------|------|-------------|
| `emotion` | string | Avatar state — see [Emotion values](#emotion-values) |
| `source` | string | `AI_MODEL`, `SER_MODEL`, `ACOUSTIC_DNA`, or `ERROR_RECOVERY` |
| `raw_text` | string | Full SenseVoice output including tags |
| `tags` | string[] | Parsed `<\|TAG\|>` tokens (uppercase) |
| `sensevoice_emotion` | string \| null | Raw SenseVoice emotion tag |
| `ser_label` | string \| null | emotion2vec label when `source=SER_MODEL` |
| `ser_confidence` | number \| null | emotion2vec softmax score (0–1) |

#### Emotion values

| Value | Description |
|-------|-------------|
| `HAPPY` | Joy, amusement |
| `SAD` | Sadness, fear (mapped from `FEARFUL`) |
| `ANGRY` | Anger, disgust (mapped from `DISGUSTED`) |
| `NEUTRAL` | Calm, neutral |
| `EXCITED` | Surprise, excitement (mapped from `SURPRISED`) |

#### Disconnect behaviour

- Normal client disconnect → connection closed cleanly
- Server error → logged, connection closed
- Engine not ready → close code `1011`

---

## Testing

### RAVDESS static benchmark

Evaluates the full pipeline against 1440 acted speech clips (24 actors × 60 utterances).

```bash
# Production pipeline (recommended)
python run.py test

# SenseVoice tags only (~28% — diagnostic)
python run.py test --ai-only

# Disable SER fallback
python run.py test --no-ser

# Quick smoke (96 files)
python run.py test --sample-limit 96

# Extra diagnostics in report
python run.py test --diagnose
```

Reports are saved to `<data-dir>/test/reports/` with unique filenames:

```
ravdess_lang-en_hybrid_ser_full_nodiag_20260827_124222.txt
         │         │      │    │    │
         │         │      │    │    └── diagnose on/off
         │         │      │    └── full dataset or n96
         │         │      └── ser / no-ser
         │         └── hybrid / ai-only
         └── language hint
```

### SER smoke test

```bash
python tests/integration/test_ser.py
```

Runs emotion2vec on a single RAVDESS clip. Expect:

```
SER result: {'emotion': 'SAD', 'confidence': 0.72, 'raw_label': 'sad'}
```

### Benchmark results (reference)

| Mode | Overall | SER loaded |
|------|---------|------------|
| `--ai-only` | 27.78% | No |
| **Default hybrid** | **80.76%** | Yes |

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ANABELLE_DATA_DIR` | Colab: `/content/anabelle-data`, local: repo root | Model + test data root |
| `ANABELLE_HOST` | `0.0.0.0` | Gateway bind address |
| `ANABELLE_PORT` | `8000` | Gateway port |
| `ANABELLE_SER_MODEL` | `iic/emotion2vec_plus_large` | emotion2vec model ID |
| `ANABELLE_MODEL_HUB` | `hf` | Model hub (`hf` or `ms`) |
| `ANABELLE_SER_HUBS` | `hf,ms` | Hub fallback order |
| `ANABELLE_SER_MIN_CONF` | `0.20` | Minimum SER softmax confidence |
| `ANABELLE_BACKEND` | `pytorch` | SenseVoice backend: `pytorch` or `onnx` |
| `ANABELLE_QUANTIZE` | `fp32` | Model precision: `fp32`, `fp16` (CUDA only), or `int8` (ONNX) |
| `ANABELLE_VAD` | `rms` | Voice activity detection: `rms`, `silero`, or `off` |
| `ANABELLE_VAD_RMS` | `0.02` | RMS threshold for VAD gate |
| `ANABELLE_SER_MODE` | `smart` | SER policy: `always`, `smart`, or `off` |
| `ANABELLE_DUAL_PATH` | `1` | Enable dual-path WebSocket (reflex + deferred emotion) |
| `ANABELLE_ENABLE_SEMANTIC` | `1` | Enable semantic text-based emotion short-circuit |
| `ANABELLE_ENABLE_SMOOTHING` | `1` | Enable emotion hysteresis smoothing |
| `ANABELLE_SMOOTHING_WINDOW` | `3` | Smoothing window size |
| `ANABELLE_MIN_CHUNK_INTERVAL` | `0.5` | Minimum seconds between audio chunks (rate limiter) |

---

## Performance Optimization for CPU

For real-time performance on CPU/MPS devices, use these low-latency options (with slight accuracy trade-offs):

### Recommended CPU configuration

```bash
# Smart SER mode (default) - skips heavy SER when text is detected
export ANABELLE_SER_MODE=smart

# ONNX backend with INT8 quantization (2-4x faster than PyTorch FP32)
export ANABELLE_BACKEND=onnx
export ANABELLE_QUANTIZE=int8

# RMS VAD gate (skips inference on silence)
export ANABELLE_VAD=rms
export ANABELLE_VAD_RMS=0.02

# Dual-path enabled (default) - instant reflex response
export ANABELLE_DUAL_PATH=1
```

Or via CLI:

```bash
python -m anabelle.cli serve --backend onnx --quantize int8 --ser-mode smart --vad rms
```

### Performance modes

| Mode | Latency | Accuracy | Use case |
|------|---------|----------|----------|
| **Default (smart)** | ~0.4-0.5s/chunk | ~75-80% | Real-time avatar with good accuracy |
| **CPU-optimized** | ~0.3-0.4s/chunk | ~70-75% | CPU-only, maximum responsiveness |
| **Maximum speed** | ~0.2-0.3s/chunk | ~60-65% | Latency-critical, accept lower accuracy |
| **GPU default** | ~0.1-0.2s/chunk | ~80-81% | Best quality with GPU |

### CPU-optimized configuration

```bash
# Maximum CPU performance (lower accuracy)
export ANABELLE_SER_MODE=off           # Disable SER entirely
export ANABELLE_BACKEND=onnx           # Use ONNX runtime
export ANABELLE_QUANTIZE=int8          # INT8 quantization
export ANABELLE_VAD=rms                # RMS gate
export ANABELLE_VAD_RMS=0.03           # Higher threshold (more aggressive gating)
export ANABELLE_DUAL_PATH=1            # Keep dual-path for responsiveness
```

### Trade-offs explained

- **`ANABELLE_SER_MODE=smart`**: Skips SER (RTF ~5.3) when SenseVoice returns text or RMS is low. Cuts ~1s per chunk with minimal accuracy loss.
- **`ANABELLE_SER_MODE=off`**: Disables SER entirely. Uses acoustic fallback only. Fastest but ~10-15% accuracy drop.
- **`ANABELLE_BACKEND=onnx` + `ANABELLE_QUANTIZE=int8`**: ONNX Runtime with INT8 quantization is 2-4x faster than PyTorch FP32 on CPU. **Note**: First startup exports PyTorch model to ONNX (can take 5-10 minutes on CPU/MPS). For immediate use, stick with PyTorch backend + smart SER mode.
- **`ANABELLE_VAD=rms`**: Skips all inference when audio RMS is below threshold. Near 0% CPU during pauses.
- **`ANABELLE_DUAL_PATH=1`**: Sends instant "reflex" response (acoustic DNA) while AI processes. Avatar feels responsive even with slower inference.
- **`ANABELLE_MIN_CHUNK_INTERVAL`**: Server-side rate limiter that drops chunks arriving faster than the specified interval. Prevents CPU overload from overly aggressive frontend sampling.

### Recommended immediate setup (no ONNX export delay)

```bash
# Use PyTorch backend with smart SER mode - no export delay, still fast
export ANABELLE_BACKEND=pytorch
export ANABELLE_QUANTIZE=fp32
export ANABELLE_SER_MODE=smart
export ANABELLE_VAD=rms
export ANABELLE_DUAL_PATH=1
export ANABELLE_MIN_CHUNK_INTERVAL=0.5  # Rate limit: minimum 0.5s between chunks
python run.py serve
```

Or via CLI:

```bash
python -m anabelle.cli serve --backend pytorch --ser-mode smart --min-chunk-interval 0.5
```

---

## Colab Workflow

```python
# First time
!git clone https://github.com/Erebus9456/anabelle-backend
%cd anabelle-backend
!python setup.py --colab

# Later sessions
%cd /content/anabelle-backend
!git pull
!python setup.py --colab --skip-models --skip-test-data

!python run.py test
!python run.py serve   # keep cell running; expose port 8000 via ngrok
```

Open [`notebooks/Anabelle_Colab.ipynb`](notebooks/Anabelle_Colab.ipynb) for a ready-made notebook.

---

## Dependencies

Always install via `python setup.py` — do not manually `pip install` torch/numpy/funasr.

| Stack | Packages |
|-------|----------|
| **AI inference** | funasr 1.4.4, modelscope, transformers, SenseVoiceSmall, emotion2vec+ |
| **Acoustic** | numpy 1.26.4 (≤ Py 3.12), librosa, numba, scipy |
| **Gateway** | fastapi, uvicorn, websockets |
| **CPU optimization (optional)** | funasr-onnx, onnxruntime (install with `--with-onnx`) |

See [`requirements/`](requirements/) for pinned versions.

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `ModuleNotFoundError: anabelle` | Run commands from repo root; `pip install -e .` optional |
| `ModuleNotFoundError: funasr_onnx` | Install ONNX Runtime: `python setup.py --with-onnx` or `pip install -r requirements/onnx.txt` |
| `OverflowError: cannot convert longdouble infinity to integer` (Windows) | NumPy 1.26.4 has a float128 bug on Windows. Re-run `python setup.py` to install NumPy 2.x, or manually: `pip install numpy>=2.0.0` |
| `SER loaded: False` | Re-run `python setup.py`; try `ANABELLE_MODEL_HUB=ms` |
| No `SER_MODEL` in reports | Pull latest code (bilingual label fix in `ser.py`) |
| 404 on model load | Re-run setup; check `WeTextProcessing` installed |
| `tokenizers` build fail (Py 3.13) | Use `python setup.py --colab` (handles wheel pins) |
| Port already in use | `ANABELLE_PORT=8001 python run.py serve` |

---

## License

Proprietary — ANABELLE / LUKYX Engine.
