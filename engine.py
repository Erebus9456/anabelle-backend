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

        # 1. EMOTION MAPPING (AI Tag -> Avatar State)
        self.emotion_map = {
            "HAPPY": "HAPPY",
            "SAD": "SAD",
            "ANGRY": "ANGRY",
            "NEUTRAL": "NEUTRAL",
            "FEARFUL": "SAD",
            "DISGUSTED": "ANGRY",
            "SURPRISED": "EXCITED",
            "EMO_UNKNOWN": None
        }

        # 2. EVENT MAPPING (AI Event -> Avatar State)
        # This ensures the avatar reacts to sounds like laughter or crying
        self.event_map = {
            "LAUGHTER": "HAPPY",
            "CRY": "SAD",
            "SING": "EXCITED",
            "SPEECH": None, # Handled by emotion
            "APPLAUSE": "EXCITED",
            "BGM": None,
            "BREATH": None,
            "COUGH": None,
            "SNEEZE": None
        }

        logger.info("ANABELLE Engine loaded successfully.")

    def get_acoustic_fallback(self, audio_data):
        """Refined Acoustic Heuristics for the LUKYX Engine."""
        rms = np.sqrt(np.mean(audio_data**2))
        zcr = np.mean(librosa.feature.zero_crossing_rate(audio_data))
        
        if rms < 0.01: return "NEUTRAL"
        
        if rms > 0.07:
            return "ANGRY" if zcr > 0.08 else "EXCITED"
        if zcr < 0.04:
            return "SAD"
        return "HAPPY"

    def extract_state_from_tags(self, text):
        """
        Processes your emoji_dict logic: 
        1. Checks for primary emotions first.
        2. Falls back to events (Laughter/Cry).
        """
        tags = re.findall(r"<\|(\w+)\|>", text.upper())
        
        # Check for direct emotions first
        for tag in tags:
            if tag in self.emotion_map and self.emotion_map[tag]:
                return self.emotion_map[tag]
        
        # Check for events if no emotion was found
        for tag in tags:
            if tag in self.event_map and self.event_map[tag]:
                return self.event_map[tag]
                
        return None

    @torch.inference_mode()
    def analyze_chunk(self, audio_data):
        try:
            res = self.model.generate(
                input=audio_data,
                cache={},
                language="auto",
                use_itn=True
            )
            
            if res and len(res) > 0:
                raw_text = res[0]['text']
                
                # Use the comprehensive tag parser
                inferred_emotion = self.extract_state_from_tags(raw_text)
                
                if inferred_emotion:
                    return {
                        "emotion": inferred_emotion,
                        "source": "AI_MODEL",
                        "raw_text": raw_text
                    }
                else:
                    # AI text exists but no specific emotion/event tag was triggered
                    return {
                        "emotion": self.get_acoustic_fallback(audio_data),
                        "source": "ACOUSTIC_DNA",
                        "raw_text": raw_text
                    }
                
        except Exception as e:
            logger.error(f"Engine Error: {e}")
            return {"emotion": "NEUTRAL", "source": "ERROR_RECOVERY"}

if __name__ == "__main__":
    engine = AnabelleEngine()
    print("Engine Test Ready.")
