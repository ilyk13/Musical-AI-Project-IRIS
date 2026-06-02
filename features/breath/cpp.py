"""Cepstral Peak Prominence (CPP) — a measure of breathiness / voice clarity.

CPP measures how prominent the periodic component of the voice is relative
to the overall cepstral noise floor. A strong, clear voice has a high CPP
(> ~15 dB). A breathy or airy voice has a lower CPP because the harmonic
structure is weaker relative to the noise.

Reference: Hillenbrand et al. (1994) — "Acoustic correlates of breathy
vocal quality: Dysphonic voices and continuous speech."
"""

import numpy as np
from scipy import signal


def compute_cpp(
    audio_frame: np.ndarray,
    sr: int = 16000,
    f0_min: float = 60.0,
    f0_max: float = 500.0,
    quefrency_floor: float = 0.001,
) -> float:
    """Compute CPP for a single audio frame.

    CPP = (cepstral peak value) - (value of a linear regression line at that
          quefrency), measured in dB.

    Args:
        audio_frame:     short audio segment (typically 25–50 ms)
        sr:              sample rate
        f0_min / f0_max: pitch range to search for the cepstral peak (Hz)
        quefrency_floor: minimum quefrency to include in regression (seconds)

    Returns:
        cpp: float in dB (higher = clearer / less breathy)
    """
    # Power spectrum → cepstrum via log + IFFT.
    #
    # Important: CPP is defined in dB (peak relative to a baseline), but the
    # cepstrum here is computed from log-power. If we use natural log, CPP
    # values end up in "nats" and look tiny (~0.1). Compute log-power in dB so
    # CPP lands in a human-meaningful dB range (~5–25 for typical singing).
    n = len(audio_frame)
    windowed = audio_frame * np.hanning(n)
    spectrum = np.fft.rfft(windowed, n=n)
    power = np.abs(spectrum) ** 2
    log_power_db = 10.0 * np.log10(power + 1e-12)
    cepstrum = np.fft.irfft(log_power_db)[:n // 2]

    quefrencies = np.arange(len(cepstrum)) / sr  # seconds

    # Restrict to the quefrency range corresponding to f0_min–f0_max
    q_low = 1.0 / f0_max
    q_high = 1.0 / f0_min
    valid = (quefrencies >= max(q_low, quefrency_floor)) & (quefrencies <= q_high)

    if not valid.any():
        return 0.0

    peak_idx = np.argmax(cepstrum[valid])
    peak_q = quefrencies[valid][peak_idx]
    peak_val = cepstrum[valid][peak_idx]

    # Fit a regression line to smooth out the cepstrum and find the baseline
    reg_mask = quefrencies >= quefrency_floor
    q_reg = quefrencies[reg_mask]
    c_reg = cepstrum[reg_mask]
    if len(q_reg) < 2:
        return 0.0

    coeffs = np.polyfit(q_reg, c_reg, 1)
    baseline_at_peak = np.polyval(coeffs, peak_q)

    # Empirical scale: this implementation's cepstral units are smaller than
    # classic CPP dB ranges; rescale so typical singing lands ~5–25.
    cpp = (peak_val - baseline_at_peak) * 12.0
    return float(max(cpp, 0.0))


def compute_cpp_sequence(
    audio: np.ndarray,
    sr: int = 16000,
    hop_length: int = 160,
    frame_length: int = 800,   # 50 ms — longer window gives better cepstral resolution
    f0_min: float = 60.0,
    f0_max: float = 500.0,
) -> np.ndarray:
    """Compute CPP for each frame of an audio signal.

    Args:
        audio:        (T,) float32 mono audio
        sr:           sample rate
        hop_length:   samples between frames (10 ms at 16 kHz)
        frame_length: analysis window in samples (50 ms at 16 kHz)
        f0_min:       minimum expected f0 (Hz)
        f0_max:       maximum expected f0 (Hz)

    Returns:
        cpp: (frames,) float32 — CPP per frame in dB
    """
    pad = frame_length // 2
    padded = np.pad(audio, (pad, pad), mode='reflect')

    n_frames = 1 + (len(audio) - 1) // hop_length
    cpp_vals = np.zeros(n_frames, dtype=np.float32)

    for i in range(n_frames):
        start = i * hop_length
        frame = padded[start: start + frame_length]
        if len(frame) < frame_length:
            frame = np.pad(frame, (0, frame_length - len(frame)))
        cpp_vals[i] = compute_cpp(frame, sr=sr, f0_min=f0_min, f0_max=f0_max)

    return cpp_vals
