"""Audio loading and conversion utilities."""
from __future__ import annotations

import subprocess
import tempfile

import numpy as np
import soundfile as sf
import torch


def convert_to_wav16k(path: str) -> str:
    """Transcode any ffmpeg-readable audio to a 16 kHz mono WAV temp file.

    Skips the transcode only if the input is already a 16 kHz mono WAV —
    a `.wav` extension alone is not enough (a 24 kHz stereo wav must still
    be resampled).
    """
    if path.lower().endswith(".wav"):
        try:
            info = sf.info(path)
            if info.samplerate == 16000 and info.channels == 1:
                return path
        except Exception:
            pass  # unreadable header → fall through to ffmpeg
    out = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    out.close()
    subprocess.check_call(
        ["ffmpeg", "-y", "-i", path, "-ar", "16000", "-ac", "1", out.name],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return out.name


def load_audio(path: str) -> np.ndarray:
    """Return a 1-D float32 numpy waveform in the file's native sample rate (mono)."""
    audio, sr = sf.read(path)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return audio.astype(np.float32), int(sr)


def load_audio_16k_mono(path: str) -> torch.Tensor:
    """Read any audio file and return a 1-D float32 torch tensor at 16 kHz mono."""
    wav_path = convert_to_wav16k(path)
    audio, sr = load_audio(wav_path)
    assert sr == 16000, f"convert_to_wav16k produced sr={sr}"
    return torch.from_numpy(audio)
