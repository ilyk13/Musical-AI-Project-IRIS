"""Audio I/O utilities — load, save, and query audio files."""

import numpy as np
import librosa
import soundfile as sf

TARGET_SR = 16000  # NanoPitch expects 16 kHz


def load(path: str, sr: int = TARGET_SR, mono: bool = True) -> tuple[np.ndarray, int]:
    """Load an audio file, resampling and converting to mono as needed.

    Args:
        path: path to audio file (wav, mp3, flac, etc.)
        sr:   target sample rate (default 16 kHz for NanoPitch compatibility)
        mono: if True, mix down to mono

    Returns:
        audio: (T,) float32 array, range roughly [-1, 1]
        sr:    the actual sample rate (equals the requested sr)
    """
    audio, actual_sr = librosa.load(path, sr=sr, mono=mono)
    return audio.astype(np.float32), actual_sr


def save(path: str, audio: np.ndarray, sr: int = TARGET_SR) -> None:
    """Save a float32 audio array to a WAV file."""
    sf.write(path, audio, sr)


def duration(path: str) -> float:
    """Return the duration of an audio file in seconds without decoding it."""
    info = sf.info(path)
    return info.duration


def sample_rate(path: str) -> int:
    """Return the native sample rate of an audio file."""
    info = sf.info(path)
    return info.samplerate
