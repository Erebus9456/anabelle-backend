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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("AnabelleEngine")

model_path = str(get_model_dir())

# Official SenseVoice emotion tags (always present in model output when working correctly)
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

SENSEVOICE_EVENT_TAGS = frozenset(
    {"SPEECH", "BGM", "APPLAUSE", "LAUGHTER", "CRY", "SING", "COUGH", "SNEEZE", "BREATH"}
)


class AnabelleEngine:
    def __init__(self):
        logger.info(f"Initializing Hybrid Engine from: {model_path}")
        device_info = get_device_info()
        self.device = device_info.device
        self.use_fp16 = device_info.use_fp16 and self.device == "cuda"

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

        # SenseVoice tag -> avatar state
        self.emotion_map = {
            "HAPPY": "HAPPY",
            "SAD": "SAD",
            "ANGRY": "ANGRY",
            "NEUTRAL": "NEUTRAL",
            "FEARFUL": "SAD",
            "DISGUSTED": "ANGRY",
            "SURPRISED": "EXCITED",
            "EMO_UNKNOWN": "NEUTRAL",
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

        logger.info("ANABELLE Engine loaded successfully.")

    @staticmethod
    def preprocess_audio(audio_data: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
        """Peak-normalize float32 PCM for consistent SenseVoice inference."""
        audio = np.asarray(audio_data, dtype=np.float32).flatten()
        peak = np.max(np.abs(audio))
        if peak > 0:
            audio = (audio / peak) * 0.95
        return audio

    def parse_tags(self, text: str) -> list[str]:
        return [tag.upper() for tag in re.findall(r"<\|([^|]+)\|>", text)]

    def extract_state_from_tags(self, text: str) -> str | None:
        """
        Parse SenseVoice rich tags using the official emotion/event vocabulary.
        Returns avatar emotion or None if no usable tag was found.
        """
        tags = self.parse_tags(text)

        for tag in tags:
            if tag in SENSEVOICE_EMOTION_TAGS:
                return self.emotion_map[tag]

        for tag in tags:
            mapped = self.event_map.get(tag)
            if mapped:
                return mapped

        return None

    def get_acoustic_fallback(self, audio_data: np.ndarray, sample_rate: int = 16000) -> str:
        """
        Prosody-based fallback for live chunks when SenseVoice omits an emotion tag.
        Uses energy, zero-crossing rate, spectral centroid, and pitch cues.
        """
        audio = self.preprocess_audio(audio_data, sample_rate)
        if audio.size == 0:
            return "NEUTRAL"

        rms = float(np.sqrt(np.mean(audio**2)))
        if rms < 0.008:
            return "NEUTRAL"

        zcr = float(np.mean(librosa.feature.zero_crossing_rate(y=audio)))
        centroid = float(
            np.mean(librosa.feature.spectral_centroid(y=audio, sr=sample_rate))
        )
        rolloff = float(
            np.mean(librosa.feature.spectral_rolloff(y=audio, sr=sample_rate, roll_percent=0.85))
        )

        pitches, magnitudes = librosa.piptrack(y=audio, sr=sample_rate, fmin=75, fmax=400)
        pitch_mask = magnitudes > np.percentile(magnitudes, 75)
        voiced = pitches[pitch_mask]
        voiced = voiced[voiced > 0]
        pitch_mean = float(np.mean(voiced)) if voiced.size else 160.0

        # High energy + harsh spectrum -> angry
        if rms > 0.07 and (zcr > 0.075 or centroid > 2200):
            return "ANGRY"

        # High pitch + high energy -> excited; moderate -> happy
        if pitch_mean > 195 and rms > 0.045:
            return "EXCITED"
        if pitch_mean > 170 and rms > 0.035:
            return "HAPPY"

        # Low energy, low pitch, dull spectrum -> sad
        if rms < 0.04 and (pitch_mean < 155 or centroid < 1400 or zcr < 0.045):
            return "SAD"
        if rolloff < 1800 and rms < 0.05:
            return "SAD"

        if rms < 0.03:
            return "NEUTRAL"

        return "HAPPY"

    @torch.inference_mode()
    def analyze_chunk(
        self,
        audio_data,
        *,
        language: str = "auto",
        allow_acoustic_fallback: bool = True,
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
            inferred_emotion = self.extract_state_from_tags(raw_text)

            if inferred_emotion:
                return {
                    "emotion": inferred_emotion,
                    "source": "AI_MODEL",
                    "raw_text": raw_text,
                    "tags": self.parse_tags(raw_text),
                }

            if allow_acoustic_fallback:
                return {
                    "emotion": self.get_acoustic_fallback(audio, sample_rate),
                    "source": "ACOUSTIC_DNA",
                    "raw_text": raw_text,
                    "tags": self.parse_tags(raw_text),
                }

            return {
                "emotion": "NEUTRAL",
                "source": "AI_MODEL",
                "raw_text": raw_text,
                "tags": self.parse_tags(raw_text),
            }

        except Exception as e:
            logger.error(f"Engine Error: {e}")
            fallback = (
                self.get_acoustic_fallback(audio, sample_rate)
                if allow_acoustic_fallback
                else "NEUTRAL"
            )
            return {
                "emotion": fallback,
                "source": "ERROR_RECOVERY",
                "raw_text": "",
                "tags": [],
            }


if __name__ == "__main__":
    engine = AnabelleEngine()
    print("Engine Test Ready.")
