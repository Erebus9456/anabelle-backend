"""Dedicated speech-emotion recognition via emotion2vec+ (FunASR)."""

from __future__ import annotations

import logging
import os

import numpy as np
import torch
from funasr import AutoModel

logger = logging.getLogger("AnabelleSER")

# emotion2vec+ class index -> label
EMOTION2VEC_TO_AVATAR = {
    "angry": "ANGRY",
    "disgusted": "ANGRY",
    "fearful": "SAD",
    "happy": "HAPPY",
    "neutral": "NEUTRAL",
    "sad": "SAD",
    "surprised": "EXCITED",
}

DEFAULT_SER_MODEL = os.environ.get(
    "ANABELLE_SER_MODEL", "iic/emotion2vec_plus_large"
)
DEFAULT_SER_HUB = os.environ.get("ANABELLE_MODEL_HUB", "hf")
DEFAULT_MIN_CONFIDENCE = float(os.environ.get("ANABELLE_SER_MIN_CONF", "0.30"))


class SerEngine:
    """Wraps emotion2vec+ for utterance-level emotion classification."""

    def __init__(self, device: str = "cpu") -> None:
        logger.info("Loading SER model: %s (hub=%s)", DEFAULT_SER_MODEL, DEFAULT_SER_HUB)
        self.model = AutoModel(
            model=DEFAULT_SER_MODEL,
            hub=DEFAULT_SER_HUB,
            device=device,
            disable_update=True,
        )
        logger.info("SER model ready on %s", device)

    @torch.inference_mode()
    def predict(
        self,
        audio: np.ndarray,
        *,
        sample_rate: int = 16000,
        min_confidence: float = DEFAULT_MIN_CONFIDENCE,
    ) -> dict:
        """
        Return avatar emotion, confidence, and raw emotion2vec label.
        Returns emotion=None when confidence is too low or label is other/unknown.
        """
        audio = np.asarray(audio, dtype=np.float32).flatten()
        if audio.size == 0:
            return {"emotion": None, "confidence": 0.0, "raw_label": "empty"}

        result = self.model.generate(
            input=audio,
            granularity="utterance",
            extract_embedding=False,
        )

        if not result:
            return {"emotion": None, "confidence": 0.0, "raw_label": "empty"}

        labels = result[0].get("labels") or []
        scores = result[0].get("scores") or []
        if not labels or not scores:
            return {"emotion": None, "confidence": 0.0, "raw_label": "empty"}

        best_idx = int(np.argmax(scores))
        raw_label = str(labels[best_idx]).lower()
        confidence = float(scores[best_idx])

        if raw_label in {"other", "unknown"}:
            return {"emotion": None, "confidence": confidence, "raw_label": raw_label}

        avatar = EMOTION2VEC_TO_AVATAR.get(raw_label)
        if avatar is None or confidence < min_confidence:
            return {"emotion": None, "confidence": confidence, "raw_label": raw_label}

        return {"emotion": avatar, "confidence": confidence, "raw_label": raw_label}
