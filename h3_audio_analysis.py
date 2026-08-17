"""Optional local audio analysis helpers for H3 prompt creation.

- faster-whisper: speech / lyric transcription with timestamps
- librosa: music/audio feature analysis
- ffmpeg: optional extraction of embedded audio from reference VIDEO files

All imports are optional so the node can still load in environments where the
extra packages are not installed. The node reports missing capabilities rather
than silently inventing audio information.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import threading
import wave
from typing import Any, Dict, Optional, Tuple

import numpy as np

try:
    import librosa  # type: ignore
except Exception:
    librosa = None

try:
    from faster_whisper import WhisperModel  # type: ignore
except Exception:
    WhisperModel = None

_MODEL_CACHE: Dict[Tuple[str, str], Any] = {}
_MODEL_LOCK = threading.Lock()


def _audio_numpy(audio: Any) -> Tuple[Optional[np.ndarray], int]:
    if not isinstance(audio, dict):
        return None, 0
    waveform = audio.get("waveform")
    sr = int(audio.get("sample_rate", 0) or 0)
    if waveform is None or sr <= 0:
        return None, sr
    try:
        arr = waveform.detach().cpu().numpy() if hasattr(waveform, "detach") else np.asarray(waveform)
        arr = np.asarray(arr, dtype=np.float32)
        if arr.ndim == 3:  # [B,C,T]
            arr = arr[0].mean(axis=0)
        elif arr.ndim == 2:  # [C,T]
            arr = arr.mean(axis=0)
        elif arr.ndim != 1:
            arr = arr.reshape(-1)
        if arr.size == 0:
            return None, sr
        peak = float(np.max(np.abs(arr)))
        if peak > 1.5:
            arr = arr / 32768.0
        return np.ascontiguousarray(arr, dtype=np.float32), sr
    except Exception:
        return None, sr


def _load_whisper(model_size: str, device: str, compute_type: str = "auto"):
    if WhisperModel is None:
        raise RuntimeError(
            "faster-whisper is not installed. Audio analysis is opt-in: run "
            "'pip install -r requirements-audio.txt' inside the H3 Prompt Creator "
            "custom node folder, using your ComfyUI Python environment."
        )
    key = (model_size, device, compute_type)
    with _MODEL_LOCK:
        if key in _MODEL_CACHE:
            return _MODEL_CACHE[key]
        actual_device = "cuda" if device == "auto" else device
        actual_compute = compute_type
        if compute_type == "auto":
            actual_compute = "float16" if actual_device == "cuda" else "int8"
        model = WhisperModel(model_size, device=actual_device, compute_type=actual_compute)
        _MODEL_CACHE[key] = model
        return model


def transcribe_audio(
    audio: Any,
    model_size: str = "small",
    device: str = "auto",
    compute_type: str = "auto",
    language: str = "auto",
) -> Dict[str, Any]:
    arr, sr = _audio_numpy(audio)
    if arr is None:
        return {"available": False, "reason": "No decodable ComfyUI AUDIO waveform."}
    model = _load_whisper(model_size, device, compute_type)
    lang = None if language == "auto" else language
    segments, info = model.transcribe(
        arr,
        language=lang,
        beam_size=5,
        vad_filter=True,
        word_timestamps=False,
        condition_on_previous_text=True,
    )
    segs = []
    parts = []
    for seg in segments:
        txt = (seg.text or "").strip()
        if not txt:
            continue
        parts.append(txt)
        segs.append({"start": round(float(seg.start), 3), "end": round(float(seg.end), 3), "text": txt})
    return {
        "available": True,
        "language": getattr(info, "language", None),
        "language_probability": round(float(getattr(info, "language_probability", 0.0) or 0.0), 4),
        "text": " ".join(parts),
        "segments": segs,
        "model": model_size,
    }


def analyze_audio_features(audio: Any) -> Dict[str, Any]:
    arr, sr = _audio_numpy(audio)
    if arr is None:
        return {"available": False, "reason": "No decodable ComfyUI AUDIO waveform."}
    result: Dict[str, Any] = {"available": True}
    duration = arr.size / sr if sr else 0.0
    result["duration_seconds"] = round(float(duration), 3)
    result["sample_rate"] = sr
    result["rms"] = round(float(np.sqrt(np.mean(arr * arr))), 5)
    result["peak"] = round(float(np.max(np.abs(arr))), 5)

    if librosa is None:
        result["analysis_backend"] = "numpy-fallback"
        return result

    result["analysis_backend"] = "librosa"
    try:
        onset_env = librosa.onset.onset_strength(y=arr, sr=sr)
        tempo = float(librosa.feature.tempo(onset_envelope=onset_env, sr=sr, aggregate=np.mean)[0]) if onset_env.size else 0.0
        result["tempo_bpm_estimate"] = round(tempo, 2) if tempo else None
    except Exception:
        result["tempo_bpm_estimate"] = None
    try:
        centroid = librosa.feature.spectral_centroid(y=arr, sr=sr)
        rolloff = librosa.feature.spectral_rolloff(y=arr, sr=sr, roll_percent=0.85)
        zcr = librosa.feature.zero_crossing_rate(arr)
        result["spectral_centroid_hz"] = round(float(np.mean(centroid)), 2)
        result["spectral_rolloff_hz"] = round(float(np.mean(rolloff)), 2)
        result["zero_crossing_rate"] = round(float(np.mean(zcr)), 5)
    except Exception:
        pass
    try:
        rms = librosa.feature.rms(y=arr)
        result["dynamic_rms_mean"] = round(float(np.mean(rms)), 5)
        result["dynamic_rms_std"] = round(float(np.std(rms)), 5)
    except Exception:
        pass
    return result


def extract_audio_from_video_source(source: Any) -> Optional[Dict[str, Any]]:
    """Use ffmpeg to extract embedded video audio into a temporary WAV and return
    it in ComfyUI-like AUDIO structure. Returns None when ffmpeg/audio is unavailable.
    """
    if source is None:
        return None
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return None
    path = source if isinstance(source, str) else None
    if not path or not os.path.exists(path):
        return None
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    try:
        cmd = [ffmpeg, "-y", "-i", path, "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", tmp.name]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if proc.returncode != 0 or not os.path.exists(tmp.name) or os.path.getsize(tmp.name) < 128:
            return None
        with wave.open(tmp.name, "rb") as wf:
            sr = wf.getframerate()
            n = wf.getnframes()
            raw = wf.readframes(n)
        arr = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        return {"waveform": arr[None, None, :], "sample_rate": sr, "_extracted_from_video": True}
    except Exception:
        return None
    finally:
        try:
            os.remove(tmp.name)
        except Exception:
            pass
