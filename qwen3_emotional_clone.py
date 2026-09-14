# """
# credit to original emotion-control file --> Dawizzer -- github.com/Dawizzer

# Requires a text file indicated by MY_INPUT_TEXT to work.
#  default is "qwen3_emotional_line_input.txt
# to rename the dir created for output audios rename the variable --> OUTPUT_DIR_NAME

#  formatting for text file lines: 
# text|emotion1|emotion2|emotion3|intensity float (e.g. 1.1)

# Example of a valid line for the input text:
# Hi there|happy|none|none|1.3


# """

import os
import re
import wave

import sys
import torch
import numpy as np
import folder_paths
from typing import Dict, Any, Tuple

# Import from the existing nodes.py structure
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# custom file for input controls
INPUT_FILE = os.path.join(current_dir, MY_INPUT_TEXT)
MY_INPUT_TEXT = "qwen3_emotional_line_input.txt"
OUTPUT_DIR_NAME = "emotional_audio_output"

# Import shared utilities from nodes.py
from .nodes import (
    load_qwen_model,
    DEMO_LANGUAGES,
    LANGUAGE_MAP,
)

class FB_Qwen3TTSEmotionalVoiceClone:
    # Comprehensive emotion presets
    EMOTION_PRESETS = {
        # Core emotions
        "neutral": {"temp": 0.0, "rep_pen": 0.0, "top_p": 0.0},
        "angry": {"temp": 0.3, "rep_pen": 0.1, "top_p": 0.05},
        "sad": {"temp": -0.2, "rep_pen": -0.05, "top_p": -0.1},
        "happy": {"temp": 0.2, "rep_pen": 0.05, "top_p": 0.05},
        "fearful": {"temp": 0.25, "rep_pen": 0.15, "top_p": 0.0},
        "disgusted": {"temp": 0.1, "rep_pen": 0.1, "top_p": -0.05},
        "surprised": {"temp": 0.35, "rep_pen": 0.1, "top_p": 0.1},
        
        # Intensity variations
        "furious": {"temp": 0.5, "rep_pen": 0.2, "top_p": 0.1},
        "enraged": {"temp": 0.6, "rep_pen": 0.25, "top_p": 0.15},
        "annoyed": {"temp": 0.15, "rep_pen": 0.05, "top_p": 0.0},
        "melancholic": {"temp": -0.3, "rep_pen": -0.1, "top_p": -0.15},
        "disappointed": {"temp": -0.15, "rep_pen": -0.05, "top_p": -0.05},
        "ecstatic": {"temp": 0.5, "rep_pen": 0.2, "top_p": 0.15},
        "content": {"temp": -0.1, "rep_pen": 0.0, "top_p": 0.0},
        "terrified": {"temp": 0.4, "rep_pen": 0.25, "top_p": 0.05},
        "anxious": {"temp": 0.2, "rep_pen": 0.15, "top_p": 0.0},
        
        # Social/Interpersonal
        "sarcastic": {"temp": 0.15, "rep_pen": 0.05, "top_p": 0.0},
        "mocking": {"temp": 0.2, "rep_pen": 0.1, "top_p": 0.05},
        "condescending": {"temp": 0.1, "rep_pen": 0.05, "top_p": -0.05},
        "dismissive": {"temp": 0.05, "rep_pen": 0.0, "top_p": -0.1},
        "contemptuous": {"temp": 0.15, "rep_pen": 0.1, "top_p": -0.05},
        "sympathetic": {"temp": -0.1, "rep_pen": -0.05, "top_p": 0.0},
        "apologetic": {"temp": -0.15, "rep_pen": -0.05, "top_p": -0.05},
        "pleading": {"temp": 0.1, "rep_pen": 0.1, "top_p": 0.0},
        "begging": {"temp": 0.2, "rep_pen": 0.15, "top_p": 0.05},
        
        # Confidence spectrum
        "confident": {"temp": -0.1, "rep_pen": -0.05, "top_p": 0.0},
        "arrogant": {"temp": 0.0, "rep_pen": 0.0, "top_p": -0.1},
        "smug": {"temp": 0.05, "rep_pen": 0.0, "top_p": -0.05},
        "insecure": {"temp": 0.1, "rep_pen": 0.1, "top_p": 0.0},
        "hesitant": {"temp": 0.15, "rep_pen": 0.15, "top_p": 0.0},
        "timid": {"temp": -0.2, "rep_pen": 0.1, "top_p": -0.1},
        "defeated": {"temp": -0.25, "rep_pen": -0.1, "top_p": -0.1},
        "hopeless": {"temp": -0.3, "rep_pen": -0.15, "top_p": -0.15},
        
        # Energy levels
        "excited": {"temp": 0.4, "rep_pen": 0.15, "top_p": 0.1},
        "enthusiastic": {"temp": 0.3, "rep_pen": 0.1, "top_p": 0.05},
        "energetic": {"temp": 0.35, "rep_pen": 0.15, "top_p": 0.1},
        "bored": {"temp": -0.2, "rep_pen": -0.1, "top_p": -0.1},
        "tired": {"temp": -0.25, "rep_pen": -0.1, "top_p": -0.1},
        "exhausted": {"temp": -0.3, "rep_pen": -0.15, "top_p": -0.15},
        "lethargic": {"temp": -0.35, "rep_pen": -0.15, "top_p": -0.15},
        "calm": {"temp": -0.3, "rep_pen": -0.1, "top_p": -0.15},
        
        # Stress/Tension
        "nervous": {"temp": 0.2, "rep_pen": 0.2, "top_p": 0.05},
        "panicked": {"temp": 0.45, "rep_pen": 0.25, "top_p": 0.1},
        "frantic": {"temp": 0.5, "rep_pen": 0.3, "top_p": 0.15},
        "stressed": {"temp": 0.25, "rep_pen": 0.15, "top_p": 0.05},
        "overwhelmed": {"temp": 0.3, "rep_pen": 0.2, "top_p": 0.05},
        "relaxed": {"temp": -0.25, "rep_pen": -0.1, "top_p": -0.1},
        "serene": {"temp": -0.35, "rep_pen": -0.15, "top_p": -0.15},
        
        # Attitude/Mood
        "playful": {"temp": 0.25, "rep_pen": 0.1, "top_p": 0.05},
        "teasing": {"temp": 0.2, "rep_pen": 0.1, "top_p": 0.05},
        "flirtatious": {"temp": 0.15, "rep_pen": 0.05, "top_p": 0.05},
        "seductive": {"temp": 0.1, "rep_pen": 0.0, "top_p": 0.0},
        "mysterious": {"temp": -0.1, "rep_pen": 0.0, "top_p": -0.05},
        "ominous": {"temp": -0.15, "rep_pen": 0.05, "top_p": -0.1},
        "menacing": {"temp": 0.1, "rep_pen": 0.1, "top_p": -0.05},
        "threatening": {"temp": 0.2, "rep_pen": 0.15, "top_p": 0.0},
        "suspicious": {"temp": 0.1, "rep_pen": 0.1, "top_p": 0.0},
        "paranoid": {"temp": 0.25, "rep_pen": 0.2, "top_p": 0.05},
        
        # Physical states
        "drunk": {"temp": 0.4, "rep_pen": 0.25, "top_p": 0.1},
        "breathless": {"temp": 0.3, "rep_pen": 0.2, "top_p": 0.1},
        "whispering": {"temp": -0.2, "rep_pen": -0.05, "top_p": -0.1},
        "shouting": {"temp": 0.4, "rep_pen": 0.2, "top_p": 0.1},
        "in_pain": {"temp": 0.3, "rep_pen": 0.2, "top_p": 0.05},
        "sick": {"temp": -0.2, "rep_pen": -0.05, "top_p": -0.1},
        
        # Complex emotions (pre-mixed)
        "bitter": {"temp": 0.1, "rep_pen": 0.05, "top_p": -0.05},  # sad + angry
        "manic": {"temp": 0.5, "rep_pen": 0.3, "top_p": 0.15},  # excited + nervous
        "reluctant": {"temp": -0.1, "rep_pen": 0.05, "top_p": -0.05},  # defeated + nervous
        "desperate": {"temp": 0.3, "rep_pen": 0.2, "top_p": 0.05},  # fearful + pleading
        "jealous": {"temp": 0.2, "rep_pen": 0.1, "top_p": 0.0},  # angry + sad
        "frustrated": {"temp": 0.25, "rep_pen": 0.1, "top_p": 0.0},  # angry + defeated
        "relieved": {"temp": -0.15, "rep_pen": -0.05, "top_p": 0.0},  # happy + calm
        "guilty": {"temp": -0.1, "rep_pen": 0.05, "top_p": -0.05},  # sad + nervous
        "ashamed": {"temp": -0.2, "rep_pen": -0.05, "top_p": -0.1},  # defeated + sad
        "proud": {"temp": 0.0, "rep_pen": 0.0, "top_p": 0.0},  # confident + happy
        "vengeful": {"temp": 0.3, "rep_pen": 0.15, "top_p": 0.05},  # angry + confident
        "remorseful": {"temp": -0.15, "rep_pen": -0.05, "top_p": -0.05},  # sad + apologetic
        
        # Character-specific
        "rick_like": {"temp": 0.3, "rep_pen": 0.15, "top_p": 0.05},  # condescending + drunk + dismissive
        "jerry_like": {"temp": 0.1, "rep_pen": 0.15, "top_p": 0.0},  # nervous + insecure + apologetic
        "morty_like": {"temp": 0.25, "rep_pen": 0.25, "top_p": 0.05},  # anxious + overwhelmed
    }
    
    @classmethod
    def INPUT_TYPES(cls):
        # Get all emotion names sorted
        emotions = sorted(cls.EMOTION_PRESETS.keys())
        
        return {
            "required": {
                "text": ("STRING", {
                    "multiline": True,
                    "default": "This is a test of emotional voice cloning.",
                    "placeholder": "Text to synthesize"
                }),
                "ref_audio": ("AUDIO", {
                    "tooltip": "Reference audio to clone voice from"
                }),
                "primary_emotion": (emotions, {
                    "default": "neutral",
                    "tooltip": "Main emotion for this line"
                }),
                "ref_audio_transcript": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "placeholder": "What the reference audio says (for x_vector_only=False)"
                }),
                "model_choice": (["0.6B", "1.7B"], {"default": "1.7B"}),
                "device": (["auto", "cuda", "mps", "cpu"], {"default": "auto"}),
                "precision": (["bf16", "fp32"], {"default": "fp32"}),
                "language": (DEMO_LANGUAGES, {"default": "English"}),
            },
            "optional": {
                # Multi-emotion mixing
                "secondary_emotion": (["none"] + emotions, {
                    "default": "none",
                    "tooltip": "Optional: blend with primary emotion"
                }),
                "tertiary_emotion": (["none"] + emotions, {
                    "default": "none",
                    "tooltip": "Optional: add third emotion to mix"
                }),
                "emotion_intensity": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.1,
                    "tooltip": "Multiplier for emotion strength (0=neutral, 2=extreme)"
                }),
                "x_vector_only": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Fast mode (True) vs Accurate mode (False, needs transcript)"
                }),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "max_new_tokens": ("INT", {"default": 2048, "min": 512, "max": 4096, "step": 256}),
                "temperature": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 2.0, "step": 0.1}),
                "top_p": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.05}),
                "top_k": ("INT", {"default": 20, "min": 0, "max": 100, "step": 1}),
                "repetition_penalty": ("FLOAT", {"default": 1.05, "min": 1.0, "max": 2.0, "step": 0.05}),
            }
        }
    
    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "generate_emotional_clone"
    CATEGORY = "Qwen3-TTS"
    DESCRIPTION = "Ultimate voice cloning: 70+ emotions + multi-emotion mixing"

    def _audio_tensor_to_tuple(self, audio_tensor: Dict[str, Any]) -> Tuple[np.ndarray, int]:
        """Convert ComfyUI audio format to (waveform, sample_rate) tuple"""
        waveform = None
        sr = None
        
        try:
            if isinstance(audio_tensor, dict):
                if "waveform" in audio_tensor:
                    waveform = audio_tensor.get("waveform")
                    sr = audio_tensor.get("sample_rate") or audio_tensor.get("sr")
        except Exception:
            pass
        
        if isinstance(waveform, torch.Tensor):
            if waveform.dim() > 1:
                waveform = waveform.squeeze()
                if waveform.dim() > 1:
                    waveform = torch.mean(waveform, dim=0)
            waveform = waveform.cpu().numpy()

        if isinstance(waveform, np.ndarray):
            if waveform.ndim > 1:
                waveform = np.squeeze(waveform)
                if waveform.ndim > 1:
                    waveform = np.mean(waveform, axis=0)
            waveform = waveform.astype(np.float32)

        if waveform is None or not isinstance(waveform, np.ndarray) or waveform.size == 0:
            raise RuntimeError("Failed to parse reference audio waveform")
        
        min_samples = 1024
        if waveform.size < min_samples:
            pad_amount = min_samples - waveform.size
            waveform = np.concatenate([waveform, np.zeros(pad_amount, dtype=np.float32)])

        return (waveform, int(sr))

    def _save_batch_audio(self, audio_data, index, row):
        output_dir = os.path.join(
            folder_paths.get_output_directory(),
            OUTPUT_DIR_NAME
        )
        os.makedirs(output_dir, exist_ok=True)

        def clean(value):
            return re.sub(r"[^a-zA-Z0-9_-]+", "_", value)

        filename = (
            f"{index:03d}_"
            f"{clean(row['primary_emotion'])}_"
            f"{clean(row['secondary_emotion'])}_"
            f"{clean(row['tertiary_emotion'])}_"
            f"{row['emotion_intensity']:.1f}.wav"
        )

        path = os.path.join(output_dir, filename)

        waveform = audio_data["waveform"].detach().cpu()

        # ComfyUI/Qwen output is normally [1, 1, samples]
        waveform = waveform.squeeze()

        # Convert float audio [-1, 1] to 16-bit PCM
        waveform = waveform.clamp(-1.0, 1.0)
        pcm = (waveform.numpy() * 32767).astype(np.int16)

        with wave.open(path, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(int(audio_data["sample_rate"]))
            wav_file.writeframes(pcm.tobytes())

        print(f"🎧 Saved: {path}")

        return path
    
    # custom helper to assist with textfile input
    def _read_emotional_lines(self):
        if not os.path.exists(INPUT_FILE):
            raise RuntimeError(
                f"Input file not found: {INPUT_FILE}"
            )
        rows = []
        with open(INPUT_FILE, "r", encoding="utf-8") as f:
            for line_number, raw_line in enumerate(f, 1):
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = [p.strip() for p in line.split("|")]
                if len(parts) != 5:
                    raise ValueError(
                        f"Line {line_number}: expected "
                        f"text|primary|secondary|tertiary|intensity"
                    )
                text, primary, secondary, tertiary, intensity = parts
                try:
                    intensity = float(intensity)
                except ValueError:
                    raise ValueError(
                        f"Line {line_number}: intensity must be a number"
                    )
                if primary not in self.EMOTION_PRESETS:
                    raise ValueError(
                        f"Line {line_number}: unknown primary emotion '{primary}'"
                    )
                if secondary != "none" and secondary not in self.EMOTION_PRESETS:
                    raise ValueError(
                        f"Line {line_number}: unknown secondary emotion '{secondary}'"
                    )
                if tertiary != "none" and tertiary not in self.EMOTION_PRESETS:
                    raise ValueError(
                        f"Line {line_number}: unknown tertiary emotion '{tertiary}'"
                    )
                if not 0.0 <= intensity <= 2.0:
                    raise ValueError(
                        f"Line {line_number}: intensity must be between 0.0 and 2.0"
                    )
                rows.append({
                    "text": text,
                    "primary_emotion": primary,
                    "secondary_emotion": secondary,
                    "tertiary_emotion": tertiary,
                    "emotion_intensity": intensity,
                })
        if not rows:
            raise RuntimeError(
                f"No usable lines found in {INPUT_FILE}"
            )
        return rows

    # custom, based on text inputes
    def generate_emotional_clone(
        self,
        text: str,
        ref_audio: Dict[str, Any],
        primary_emotion: str,
        ref_audio_transcript: str,
        model_choice: str,
        device: str,
        precision: str,
        language: str,
        secondary_emotion: str = "none",
        tertiary_emotion: str = "none",
        emotion_intensity: float = 1.0,
        x_vector_only: bool = True,
        seed: int = 0,
        max_new_tokens: int = 2048,
        temperature: float = 1.0,
        top_p: float = 0.8,
        top_k: int = 20,
        repetition_penalty: float = 1.05,
    ):
        rows = self._read_emotional_lines()

        last_audio = None

        for index, row in enumerate(rows, 1):
            print(
                f"\n🎭 Emotional batch {index}/{len(rows)}: "
                f"{row['primary_emotion']} + "
                f"{row['secondary_emotion']} + "
                f"{row['tertiary_emotion']} "
                f"@ {row['emotion_intensity']}x"
            )

            result = self._generate_one(
                text=row["text"],
                ref_audio=ref_audio,
                primary_emotion=row["primary_emotion"],
                ref_audio_transcript=ref_audio_transcript,
                model_choice=model_choice,
                device=device,
                precision=precision,
                language=language,
                secondary_emotion=row["secondary_emotion"],
                tertiary_emotion=row["tertiary_emotion"],
                emotion_intensity=row["emotion_intensity"],
                x_vector_only=x_vector_only,
                seed=seed,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
            )

            # _generate_one() returns (audio_data,)
            audio_data = result[0]

            self._save_batch_audio(
                audio_data,
                index,
                row,
            )

            last_audio = audio_data

        # ComfyUI AUDIO output
        return (last_audio,)

    # standard, relegated to helper now below
    def _generate_one(
        self,
        text: str,
        ref_audio: Dict[str, Any],
        primary_emotion: str,
        ref_audio_transcript: str,
        model_choice: str,
        device: str,
        precision: str,
        language: str,
        secondary_emotion: str = "none",
        tertiary_emotion: str = "none",
        emotion_intensity: float = 1.0,
        x_vector_only: bool = True,
        seed: int = 0,
        max_new_tokens: int = 2048,
        temperature: float = 1.0,
        top_p: float = 0.8,
        top_k: int = 20,
        repetition_penalty: float = 1.05,
    ):
        """
        Generate speech with cloned voice and multi-emotion mixing.
        """
        
        if ref_audio is None:
            raise RuntimeError("Reference audio is required")
        
        model = load_qwen_model("Base", model_choice, device, precision)

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        np.random.seed(seed % (2**32))
        
        audio_tuple = self._audio_tensor_to_tuple(ref_audio)
        
        # Mix emotions by averaging their parameters
        emotions_to_mix = [primary_emotion]
        if secondary_emotion != "none":
            emotions_to_mix.append(secondary_emotion)
        if tertiary_emotion != "none":
            emotions_to_mix.append(tertiary_emotion)
        
        # Calculate average modifiers
        total_temp_mod = 0.0
        total_rep_pen_mod = 0.0
        total_top_p_mod = 0.0
        
        for emotion in emotions_to_mix:
            if emotion in self.EMOTION_PRESETS:
                preset = self.EMOTION_PRESETS[emotion]
                total_temp_mod += preset["temp"]
                total_rep_pen_mod += preset["rep_pen"]
                total_top_p_mod += preset["top_p"]
        
        num_emotions = len(emotions_to_mix)
        avg_temp_mod = (total_temp_mod / num_emotions) * emotion_intensity
        avg_rep_pen_mod = (total_rep_pen_mod / num_emotions) * emotion_intensity
        avg_top_p_mod = (total_top_p_mod / num_emotions) * emotion_intensity
        
        # Apply modifiers
        final_temp = max(0.1, min(2.0, temperature + avg_temp_mod))
        final_rep_pen = max(1.0, min(2.0, repetition_penalty + avg_rep_pen_mod))
        final_top_p = max(0.1, min(1.0, top_p + avg_top_p_mod))
        
        emotion_display = " + ".join(emotions_to_mix)
        
        print(f"\n🎭 [Emotional Voice Clone]")
        print(f"   Emotion Mix: {emotion_display} (intensity: {emotion_intensity:.1f}x)")
        print(f"   Text: {text[:50]}...")
        print(f"   x_vector_only: {x_vector_only}")
        print(f"   Base: temp={temperature:.2f}, top_p={top_p:.2f}, rep_pen={repetition_penalty:.2f}")
        print(f"   Final: temp={final_temp:.2f}, top_p={final_top_p:.2f}, rep_pen={final_rep_pen:.2f}")
        print(f"   Seed: {seed}")
        
        mapped_lang = LANGUAGE_MAP.get(language, "auto")
        
        try:
            wavs, sr = model.generate_voice_clone(
                text=text,
                language=mapped_lang,
                ref_audio=audio_tuple,
                ref_text=ref_audio_transcript if ref_audio_transcript.strip() else None,
                voice_clone_prompt=None,
                x_vector_only_mode=x_vector_only,
                max_new_tokens=max_new_tokens,
                top_p=final_top_p,
                top_k=top_k,
                temperature=final_temp,
                repetition_penalty=final_rep_pen,
            )
            
            print(f"   Model returned {len(wavs)} audio samples")
            
        except Exception as e:
            print(f"❌ Generation error: {e}")
            import traceback
            traceback.print_exc()
            raise RuntimeError(f"Voice clone generation failed: {e}")

        if isinstance(wavs, list) and len(wavs) > 0:
            waveform = torch.from_numpy(wavs[0]).float()
            
            print(f"   Waveform: {waveform.shape}, SR: {sr}, Duration: {waveform.shape[0]/sr:.2f}s")
            
            if waveform.ndim > 1:
                waveform = waveform.squeeze()
            waveform = waveform.unsqueeze(0).unsqueeze(0)
            audio_data = {"waveform": waveform, "sample_rate": sr}
            
            print(f"✅ Audio generated successfully ({waveform.shape[-1]} samples)")
            return (audio_data,)
        
        print(f"❌ No audio generated")
        raise RuntimeError("Failed to generate audio - empty waveform")


NODE_CLASS_MAPPINGS = {
    "FB_Qwen3TTSEmotionalVoiceClone": FB_Qwen3TTSEmotionalVoiceClone,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FB_Qwen3TTSEmotionalVoiceClone": "🎭 Qwen3-TTS Emotional Voice Clone (Ultimate)",
}


