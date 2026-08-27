"""Hybrid affective inference: SenseVoice + emotion2vec + acoustic fallback."""

from __future__ import annotations

import logging
import re
from dataclasses import replace

import librosa
import numpy as np
import torch

from anabelle.config import InferenceConfig
from anabelle.engine.backends import create_sensevoice_backend
from anabelle.engine.semantic import match_semantic_emotion
from anabelle.engine.ser import SerEngine
from anabelle.engine.vad import VadGate
from anabelle.utils.compat import apply_runtime_patches
from anabelle.utils.device import get_device_info
from anabelle.utils.paths import get_model_dir

apply_runtime_patches()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("AnabelleEngine")

model_path = str(get_model_dir())

SENSEVOICE_EMOTION_TAGS = frozenset(
    {
        "HAPPY",
        "SAD",
        "ANGRY",
        "NEUTRAL",
        "FEARFUL",
        "DISGUSTED",
        "SURPRISED",
        "EMO_UNKNOWN",
    }
)

CONFIDENT_SENSEVOICE_TAGS = frozenset(
    {"HAPPY", "SAD", "ANGRY", "FEARFUL", "DISGUSTED", "SURPRISED"}
)


class AnabelleEngine:
    def __init__(
        self,
        *,
        enable_ser: bool | None = None,
        config: InferenceConfig | None = None,
    ):
        self.config = config or InferenceConfig.from_env()
        if enable_ser is not None:
            self.config = replace(self.config, enable_ser=enable_ser)

        logger.info("Initializing Hybrid Engine from: %s", model_path)
        device_info = get_device_info()
        self.device = device_info.device

        if self.config.quantize == "fp16" and self.config.backend != "pytorch":
            logger.warning("FP16 quantize applies to PyTorch backend only; using INT8/FP32 for ONNX")
        if self.config.quantize == "fp16" and self.device != "cuda":
            logger.warning("FP16 requested but device is %s; falling back to FP32", self.device)

        self.sensevoice = create_sensevoice_backend(
            backend=self.config.backend,
            quantize=self.config.quantize,
            model_path=model_path,
            device=self.device,
        )
        self.vad = VadGate(self.config)
        self._sensevoice_cache: dict = {}

        self.emotion_map = {
            "HAPPY": "HAPPY",
            "SAD": "SAD",
            "ANGRY": "ANGRY",
            "NEUTRAL": "NEUTRAL",
            "FEARFUL": "SAD",
            "DISGUSTED": "ANGRY",
            "SURPRISED": "EXCITED",
        }

        self.event_map = {
            "LAUGHTER": "HAPPY",
            "CRY": "SAD",
            "SING": "EXCITED",
            "SPEECH": None,
            "APPLAUSE": "EXCITED",
            "BGM": None,
            "BREATH": None,
            "COUGH": None,
            "SNEEZE": None,
        }

        self.ser_engine: SerEngine | None = None
        self.ser_available = False
        if self.config.enable_ser:
            try:
                self.ser_engine = SerEngine(device=self.device)
                self.ser_available = True
            except Exception as exc:
                logger.warning("SER model unavailable, acoustic fallback only: %s", exc)

        logger.info(
            "ANABELLE Engine loaded (backend=%s, quantize=%s, vad=%s, SER=%s).",
            self.sensevoice.backend_name,
            self.sensevoice.quantize_label,
            self.config.vad_mode,
            "ready" if self.ser_available else "disabled",
        )

    @staticmethod
    def preprocess_audio(audio_data: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
        audio = np.asarray(audio_data, dtype=np.float32).flatten()
        peak = np.max(np.abs(audio))
        if peak > 0:
            audio = (audio / peak) * 0.95
        return audio

    def parse_tags(self, text: str) -> list[str]:
        return [tag.upper() for tag in re.findall(r"<\|([^|]+)\|>", text)]

    def extract_raw_sensevoice_emotion(self, text: str) -> str | None:
        for tag in self.parse_tags(text):
            if tag in SENSEVOICE_EMOTION_TAGS:
                return tag
        return None

    def extract_event_emotion(self, text: str) -> str | None:
        for tag in self.parse_tags(text):
            mapped = self.event_map.get(tag)
            if mapped:
                return mapped
        return None

    def get_acoustic_fallback(self, audio_data: np.ndarray, sample_rate: int = 16000) -> str:
        audio = self.preprocess_audio(audio_data, sample_rate)
        if audio.size == 0:
            return "NEUTRAL"

        rms = float(np.sqrt(np.mean(audio**2)))
        if rms < 0.008:
            return "NEUTRAL"

        zcr = float(np.mean(librosa.feature.zero_crossing_rate(y=audio)))
        centroid = float(np.mean(librosa.feature.spectral_centroid(y=audio, sr=sample_rate)))
        rolloff = float(
            np.mean(librosa.feature.spectral_rolloff(y=audio, sr=sample_rate, roll_percent=0.85))
        )

        pitches, magnitudes = librosa.piptrack(y=audio, sr=sample_rate, fmin=75, fmax=400)
        pitch_mask = magnitudes > np.percentile(magnitudes, 75)
        voiced = pitches[pitch_mask]
        voiced = voiced[voiced > 0]
        pitch_mean = float(np.mean(voiced)) if voiced.size else 160.0

        if rms > 0.07 and (zcr > 0.075 or centroid > 2200):
            return "ANGRY"
        if pitch_mean > 195 and rms > 0.045:
            return "EXCITED"
        if pitch_mean > 170 and rms > 0.035:
            return "HAPPY"
        if rms < 0.04 and (pitch_mean < 155 or centroid < 1400 or zcr < 0.045):
            return "SAD"
        if rolloff < 1800 and rms < 0.05:
            return "SAD"
        if rms < 0.03:
            return "NEUTRAL"
        return "HAPPY"

    def _result(
        self,
        *,
        emotion: str,
        source: str,
        raw_text: str,
        sensevoice_emotion: str | None = None,
        ser_label: str | None = None,
        ser_confidence: float | None = None,
        rms: float | None = None,
        gated: bool | None = None,
        gate_reason: str | None = None,
    ) -> dict:
        payload = {
            "emotion": emotion,
            "source": source,
            "raw_text": raw_text,
            "tags": self.parse_tags(raw_text),
            "sensevoice_emotion": sensevoice_emotion,
            "ser_label": ser_label,
            "ser_confidence": ser_confidence,
        }
        if rms is not None:
            payload["rms"] = rms
        if gated is not None:
            payload["gated"] = gated
        if gate_reason is not None:
            payload["gate_reason"] = gate_reason
        return payload

    def analyze_reflex(self, audio_data, sample_rate: int = 16000) -> dict:
        """Fast local reflex: VAD gate + acoustic DNA (no heavy models)."""
        audio = np.asarray(audio_data, dtype=np.float32).flatten()
        rms = VadGate.rms(audio)
        should_infer, gate_reason = self.vad.should_infer(audio, sample_rate)

        if not should_infer:
            return {
                "emotion": "NEUTRAL",
                "source": "VAD_GATE",
                "intensity": rms,
                "rms": rms,
                "gated": True,
                "gate_reason": gate_reason,
            }

        preprocessed = self.preprocess_audio(audio, sample_rate)
        emotion = self.get_acoustic_fallback(preprocessed, sample_rate)
        intensity = min(1.0, rms / 0.08)

        return {
            "emotion": emotion,
            "source": "ACOUSTIC_DNA",
            "intensity": intensity,
            "rms": rms,
            "gated": False,
            "gate_reason": gate_reason,
        }

    @torch.inference_mode()
    def analyze_chunk(
        self,
        audio_data,
        *,
        language: str = "auto",
        allow_acoustic_fallback: bool = True,
        allow_ser_fallback: bool = True,
        sample_rate: int = 16000,
    ):
        audio = np.asarray(audio_data, dtype=np.float32).flatten()
        rms = VadGate.rms(audio)
        should_infer, gate_reason = self.vad.should_infer(audio, sample_rate)

        if not should_infer:
            return self._result(
                emotion="NEUTRAL",
                source="VAD_GATE",
                raw_text="",
                rms=rms,
                gated=True,
                gate_reason=gate_reason,
            )

        audio = self.preprocess_audio(audio, sample_rate)

        try:
            res = self.sensevoice.generate(
                audio,
                language=language,
                use_itn=True,
                cache=self._sensevoice_cache,
                sample_rate=sample_rate,
            )

            if not res:
                raise ValueError("Empty model response")

            raw_text = res[0].get("text", "")
            sv_emotion = self.extract_raw_sensevoice_emotion(raw_text)

            if sv_emotion in CONFIDENT_SENSEVOICE_TAGS:
                return self._result(
                    emotion=self.emotion_map[sv_emotion],
                    source="AI_MODEL",
                    raw_text=raw_text,
                    sensevoice_emotion=sv_emotion,
                    rms=rms,
                    gated=False,
                    gate_reason=gate_reason,
                )

            if sv_emotion == "NEUTRAL":
                return self._result(
                    emotion="NEUTRAL",
                    source="AI_MODEL",
                    raw_text=raw_text,
                    sensevoice_emotion=sv_emotion,
                    rms=rms,
                    gated=False,
                    gate_reason=gate_reason,
                )

            event_emotion = self.extract_event_emotion(raw_text)
            if event_emotion:
                return self._result(
                    emotion=event_emotion,
                    source="AI_MODEL",
                    raw_text=raw_text,
                    sensevoice_emotion=sv_emotion,
                    rms=rms,
                    gated=False,
                    gate_reason=gate_reason,
                )

            if self.config.enable_semantic:
                semantic = match_semantic_emotion(raw_text)
                if semantic:
                    return self._result(
                        emotion=semantic,
                        source="SEMANTIC",
                        raw_text=raw_text,
                        sensevoice_emotion=sv_emotion,
                        rms=rms,
                        gated=False,
                        gate_reason=gate_reason,
                    )

            # Smart SER mode: skip SER when not needed for real-time performance
            if self.config.ser_mode == "smart":
                # Skip SER if we have any text (likely confident enough for acoustic fallback)
                # or if RMS is low (quiet speech doesn't need heavy SER)
                if raw_text.strip() or rms < 0.04:
                    logger.debug("Smart SER: skipping (has_text=%s, rms=%.3f)", bool(raw_text.strip()), rms)
                    if allow_acoustic_fallback:
                        return self._result(
                            emotion=self.get_acoustic_fallback(audio, sample_rate),
                            source="ACOUSTIC_DNA",
                            raw_text=raw_text,
                            sensevoice_emotion=sv_emotion,
                            rms=rms,
                            gated=False,
                            gate_reason=gate_reason,
                        )

            if self.config.ser_mode == "off":
                logger.debug("SER disabled by config")
                if allow_acoustic_fallback:
                    return self._result(
                        emotion=self.get_acoustic_fallback(audio, sample_rate),
                        source="ACOUSTIC_DNA",
                        raw_text=raw_text,
                        sensevoice_emotion=sv_emotion,
                        rms=rms,
                        gated=False,
                        gate_reason=gate_reason,
                    )

            if allow_ser_fallback and self.ser_engine is not None:
                ser = self.ser_engine.predict(audio, sample_rate=sample_rate)
                if ser["emotion"]:
                    return self._result(
                        emotion=ser["emotion"],
                        source="SER_MODEL",
                        raw_text=raw_text,
                        sensevoice_emotion=sv_emotion,
                        ser_label=ser["raw_label"],
                        ser_confidence=ser["confidence"],
                        rms=rms,
                        gated=False,
                        gate_reason=gate_reason,
                    )

            if allow_acoustic_fallback:
                return self._result(
                    emotion=self.get_acoustic_fallback(audio, sample_rate),
                    source="ACOUSTIC_DNA",
                    raw_text=raw_text,
                    sensevoice_emotion=sv_emotion,
                    rms=rms,
                    gated=False,
                    gate_reason=gate_reason,
                )

            return self._result(
                emotion="NEUTRAL",
                source="AI_MODEL",
                raw_text=raw_text,
                sensevoice_emotion=sv_emotion,
                rms=rms,
                gated=False,
                gate_reason=gate_reason,
            )

        except Exception as e:
            logger.error("Engine Error: %s", e)
            fallback = (
                self.get_acoustic_fallback(audio, sample_rate)
                if allow_acoustic_fallback
                else "NEUTRAL"
            )
            return self._result(
                emotion=fallback,
                source="ERROR_RECOVERY",
                raw_text="",
                sensevoice_emotion=None,
                rms=rms,
                gated=False,
                gate_reason=gate_reason,
            )
