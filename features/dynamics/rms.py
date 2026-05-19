"""RMS (Root Mean Square) loudness computation.

RMS measures the average power of an audio signal in a short window.
It's expressed in dBFS (decibels relative to full scale), where 0 dBFS
is the maximum possible level and typical singing ranges from about
-40 to -6 dBFS depending on recording level.
"""

import numpy as np


def compute_rms(audio: np.ndarray, frame_length: int = 400, hop_length: int = 160) -> np.ndarray:
    """Compute per-frame RMS energy of an audio signal.

    Args:
        audio:        (T,) float32 mono audio
        frame_length: samples per analysis window (default 400 = 25 ms at 16 kHz)
        hop_length:   samples between frames (default 160 = 10 ms at 16 kHz)

    Returns:
        rms: (frames,) float32 — RMS amplitude per frame, range [0, ~1]
    """
    # Pad so we get a value for every hop position
    pad = frame_length // 2
    padded = np.pad(audio, (pad, pad), mode='reflect')

    n_frames = 1 + (len(audio) - 1) // hop_length
    rms = np.zeros(n_frames, dtype=np.float32)

    for i in range(n_frames):
        start = i * hop_length
        frame = padded[start: start + frame_length]
        rms[i] = np.sqrt(np.mean(frame ** 2))

    return rms


def rms_to_dbfs(rms: np.ndarray, floor_db: float = -80.0) -> np.ndarray:
    """Convert linear RMS values to dBFS.

    Args:
        rms:      (T,) RMS amplitudes (from compute_rms)
        floor_db: silence floor — values below this are clipped

    Returns:
        dbfs: (T,) float32 — level in dBFS, clipped at floor_db
    """
    with np.errstate(divide='ignore'):
        dbfs = 20.0 * np.log10(np.maximum(rms, 1e-10))
    return np.maximum(dbfs, floor_db).astype(np.float32)
