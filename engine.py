from colab_compat import apply_runtime_patches

apply_runtime_patches()

from funasr import AutoModel
import logging
import re
import torch
import numpy as np
import librosa

from device_utils import get_device_info
from paths import get_model_dir
from ser_engine import SerEngine

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

# SenseVoice tags we trust directly (model was explicit, not uncertain)
CONFIDENT_SENSEVOICE_TAGS = frozenset(
    {"HAPPY", "SAD", "ANGRY", "FEARFUL", "DISGUSTED", "SURPRISED"}
)


class AnabelleEngine:
    def __init__(self, *, enable_ser: bool = True):
        logger.info(f"Initializing Hybrid Engine from: {model_path}")
        device_info = get_device_info()
        self.device = device_info.device
        self.use_fp16 = device_info.use_fp16 and self.device == "cuda"
        self.enable_ser = enable_ser

        model_kwargs = {
            "model": model_path,
            "device": self.device,
            "disable_update": True,
            "model_revision": "master",
        }
        if self.device == "cuda":
            model_kwargs["ngpu"] = 1

        logger.info(
            "Using inference device: %s (%s)%s",
            self.device,
            device_info.label,
            " with FP16" if self.use_fp16 else "",
        )
        self.model = AutoModel(**model_kwargs)

        if self.device == "cuda":
            torch.backends.cudnn.benchmark = True

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
        if enable_ser:
            try:
                self.ser_engine = SerEngine(device=self.device)
            except Exception as exc:
                logger.warning("SER model unavailable, acoustic fallback only: %s", exc)

        logger.info("ANABELLE Engine loaded successfully.")

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
    ) -> dict:
        return {
            "emotion": emotion,
            "source": source,
            "raw_text": raw_text,
            "tags": self.parse_tags(raw_text),
            "sensevoice_emotion": sensevoice_emotion,
            "ser_label": ser_label,
            "ser_confidence": ser_confidence,
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
        audio = self.preprocess_audio(audio_data, sample_rate)

        try:
            res = self.model.generate(
                input=audio,
                cache={},
                language=language,
                use_itn=True,
            )

            if not res:
                raise ValueError("Empty model response")

            raw_text = res[0].get("text", "")
            sv_emotion = self.extract_raw_sensevoice_emotion(raw_text)

            # Tier 1: confident SenseVoice emotion tags
            if sv_emotion in CONFIDENT_SENSEVOICE_TAGS:
                return self._result(
                    emotion=self.emotion_map[sv_emotion],
                    source="AI_MODEL",
                    raw_text=raw_text,
                    sensevoice_emotion=sv_emotion,
                )

            # Tier 2: SenseVoice explicitly said neutral
            if sv_emotion == "NEUTRAL":
                return self._result(
                    emotion="NEUTRAL",
                    source="AI_MODEL",
                    raw_text=raw_text,
                    sensevoice_emotion=sv_emotion,
                )

            # Tier 2b: strong audio events (laughter/cry/etc.)
            event_emotion = self.extract_event_emotion(raw_text)
            if event_emotion:
                return self._result(
                    emotion=event_emotion,
                    source="AI_MODEL",
                    raw_text=raw_text,
                    sensevoice_emotion=sv_emotion,
                )

            # Tier 3: EMO_UNKNOWN / missing tag -> emotion2vec SER
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
                    )

            # Tier 4: prosody heuristics
            if allow_acoustic_fallback:
                return self._result(
                    emotion=self.get_acoustic_fallback(audio, sample_rate),
                    source="ACOUSTIC_DNA",
                    raw_text=raw_text,
                    sensevoice_emotion=sv_emotion,
                )

            return self._result(
                emotion="NEUTRAL",
                source="AI_MODEL",
                raw_text=raw_text,
                sensevoice_emotion=sv_emotion,
            )

        except Exception as e:
            logger.error(f"Engine Error: {e}")
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
            )


if __name__ == "__main__":
    engine = AnabelleEngine()
    print("Engine Test Ready.")
