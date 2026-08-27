"""Dedicated speech-emotion recognition via emotion2vec+ (FunASR)."""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from funasr import AutoModel

logger = logging.getLogger("AnabelleSER")

EMOTION2VEC_TO_AVATAR = {
    "angry": "ANGRY",
    "disgusted": "ANGRY",
    "fearful": "SAD",
    "happy": "HAPPY",
    "neutral": "NEUTRAL",
    "sad": "SAD",
    "surprised": "EXCITED",
}

DEFAULT_SER_MODEL = os.environ.get("ANABELLE_SER_MODEL", "iic/emotion2vec_plus_large")
DEFAULT_SER_HUB = os.environ.get("ANABELLE_MODEL_HUB", "hf")
DEFAULT_MIN_CONFIDENCE = float(os.environ.get("ANABELLE_SER_MIN_CONF", "0.20"))
HUB_CANDIDATES = tuple(
    hub.strip()
    for hub in os.environ.get("ANABELLE_SER_HUBS", "hf,ms").split(",")
    if hub.strip()
)


def normalize_emotion2vec_label(label: str) -> str:
    """
    emotion2vec token file uses bilingual labels like '生气/angry'.
    Extract the English slug for mapping.
    """
    cleaned = str(label).strip().lower()
    if "/" in cleaned:
        cleaned = cleaned.rsplit("/", 1)[-1]
    if cleaned in {"<unk>", "unk"}:
        return "unknown"
    return cleaned


def parse_generate_result(result) -> dict | None:
    """Normalize FunASR generate output to a single result dict."""
    if result is None:
        return None

    if isinstance(result, tuple):
        result = result[0]

    if isinstance(result, list):
        if not result:
            return None
        result = result[0]

    if isinstance(result, dict):
        return result

    return None


class SerEngine:
    """Wraps emotion2vec+ for utterance-level emotion classification."""

    def __init__(self, device: str = "cpu") -> None:
        last_error: Exception | None = None

        for hub in HUB_CANDIDATES:
            try:
                logger.info(
                    "Loading SER model: %s (hub=%s, device=%s)",
                    DEFAULT_SER_MODEL,
                    hub,
                    device,
                )
                self.model = AutoModel(
                    model=DEFAULT_SER_MODEL,
                    hub=hub,
                    device=device,
                    disable_update=True,
                )
                self.hub = hub
                logger.info("SER model ready via hub=%s on %s", hub, device)
                return
            except Exception as exc:
                last_error = exc
                logger.warning("SER hub %s failed: %s", hub, exc)

        raise RuntimeError(
            f"Unable to load SER model {DEFAULT_SER_MODEL} from hubs {HUB_CANDIDATES}"
        ) from last_error

    def _generate(self, audio: np.ndarray, sample_rate: int) -> dict | None:
        audio = np.asarray(audio, dtype=np.float32).flatten()

        # Primary path: pass numpy directly (FunASR-supported).
        try:
            raw = self.model.generate(
                input=audio,
                granularity="utterance",
                extract_embedding=False,
                fs=sample_rate,
            )
            parsed = parse_generate_result(raw)
            if parsed and parsed.get("labels") and parsed.get("scores"):
                return parsed
        except Exception as exc:
            logger.debug("SER numpy inference failed: %s", exc)

        # Fallback: temp WAV (most compatible with emotion2vec frontend).
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as handle:
            temp_path = Path(handle.name)

        try:
            sf.write(temp_path, audio, sample_rate)
            raw = self.model.generate(
                input=str(temp_path),
                granularity="utterance",
                extract_embedding=False,
            )
            return parse_generate_result(raw)
        finally:
            temp_path.unlink(missing_ok=True)

    @torch.inference_mode()
    def predict(
        self,
        audio: np.ndarray,
        *,
        sample_rate: int = 16000,
        min_confidence: float = DEFAULT_MIN_CONFIDENCE,
    ) -> dict:
        audio = np.asarray(audio, dtype=np.float32).flatten()
        if audio.size == 0:
            return {"emotion": None, "confidence": 0.0, "raw_label": "empty"}

        try:
            parsed = self._generate(audio, sample_rate)
        except Exception as exc:
            logger.warning("SER inference error: %s", exc)
            return {"emotion": None, "confidence": 0.0, "raw_label": "error"}

        if not parsed:
            return {"emotion": None, "confidence": 0.0, "raw_label": "empty"}

        labels = parsed.get("labels") or []
        scores = parsed.get("scores") or []
        if not labels or not scores or len(labels) != len(scores):
            logger.debug("SER returned invalid labels/scores: %s", parsed)
            return {"emotion": None, "confidence": 0.0, "raw_label": "empty"}

        best_idx = int(np.argmax(scores))
        raw_label = normalize_emotion2vec_label(labels[best_idx])
        confidence = float(scores[best_idx])

        if raw_label in {"other", "unknown"}:
            # Pick the best usable emotion even if "other" wins narrowly.
            ranked = sorted(
                (
                    (normalize_emotion2vec_label(label), float(score))
                    for label, score in zip(labels, scores)
                ),
                key=lambda item: item[1],
                reverse=True,
            )
            for candidate, score in ranked:
                if candidate in EMOTION2VEC_TO_AVATAR and score >= min_confidence:
                    return {
                        "emotion": EMOTION2VEC_TO_AVATAR[candidate],
                        "confidence": score,
                        "raw_label": candidate,
                    }
            return {"emotion": None, "confidence": confidence, "raw_label": raw_label}

        avatar = EMOTION2VEC_TO_AVATAR.get(raw_label)
        if avatar is None:
            logger.debug("Unmapped SER label %r from %r", raw_label, labels[best_idx])
            return {"emotion": None, "confidence": confidence, "raw_label": raw_label}

        if confidence < min_confidence:
            return {"emotion": None, "confidence": confidence, "raw_label": raw_label}

        return {"emotion": avatar, "confidence": confidence, "raw_label": raw_label}
