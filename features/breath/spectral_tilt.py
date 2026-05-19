"""Spectral tilt — a secondary measure of breathiness and vocal effort.

Spectral tilt describes how quickly the energy in the voice drops off
from low to high frequencies. A breathy voice has steeper tilt (more
energy concentrated in low frequencies) because the high harmonics are
weakened by incomplete glottal closure. A pressed or loud voice has a
flatter tilt because the harmonics are stronger.

We measure this as:
  1. The slope (in dB/octave) of a linear regression on the log power spectrum.
  2. The H1–H2 difference: energy in the first harmonic minus the second,
     which is a simpler correlate of breathiness when f0 is known.
"""

import numpy as np
from scipy import signal as sp_signal


def compute_spectral_tilt_slope(
    audio_frame: np.ndarray,
    sr: int = 16000,
    f_low: float = 50.0,
    f_high: float = 4000.0,
) -> float:
    """Compute the spectral tilt slope in dB/octave for a single frame.

    A more negative value means steeper tilt (breathier).
    A value near zero means flat spectrum (bright / pressed voice).

    Args:
        audio_frame: short audio segment
        sr:          sample rate
        f_low:       lower frequency bound for regression (Hz)
        f_high:      upper frequency bound for regression (Hz)

    Returns:
        slope: float in dB/octave (typically negative, e.g. -6 to -18)
    """
    n = len(audio_frame)
    windowed = audio_frame * np.hanning(n)
    spectrum = np.fft.rfft(windowed, n=n)
    freqs = np.fft.rfftfreq(n, d=1.0 / sr)
    power_db = 20.0 * np.log10(np.abs(spectrum) + 1e-10)

    mask = (freqs >= f_low) & (freqs <= f_high)
    if mask.sum() < 2:
        return 0.0

    # Regress power_db against log2(freq) — this gives slope in dB/octave
    log_freqs = np.log2(freqs[mask])
    coeffs = np.polyfit(log_freqs, power_db[mask], 1)
    return float(coeffs[0])


def compute_h1_h2(
    audio_frame: np.ndarray,
    f0_hz: float,
    sr: int = 16000,
    search_range_cents: float = 30.0,
) -> float:
    """Compute H1–H2: energy at 1st harmonic minus energy at 2nd harmonic.

    H1 is the energy at f0, H2 at 2*f0. H1 >> H2 indicates breathiness;
    H1 ≈ H2 or H1 < H2 suggests a clearer, more modal voice.

    Requires a valid f0 estimate. Returns NaN if f0 is invalid.

    Args:
        audio_frame:       short audio frame
        f0_hz:             fundamental frequency in Hz
        sr:                sample rate
        search_range_cents: ±search range around each harmonic (cents)

    Returns:
        h1_minus_h2: float in dB (positive = breathy, negative = clear)
    """
    if f0_hz <= 0 or np.isnan(f0_hz):
        return float('nan')

    n = len(audio_frame)
    windowed = audio_frame * np.hanning(n)
    spectrum = np.fft.rfft(windowed, n=n)
    freqs = np.fft.rfftfreq(n, d=1.0 / sr)
    power_db = 20.0 * np.log10(np.abs(spectrum) + 1e-10)

    def peak_energy(target_hz: float) -> float:
        ratio = 2.0 ** (search_range_cents / 1200.0)
        mask = (freqs >= target_hz / ratio) & (freqs <= target_hz * ratio)
        if not mask.any():
            return float('nan')
        return float(power_db[mask].max())

    h1 = peak_energy(f0_hz)
    h2 = peak_energy(2.0 * f0_hz)

    if np.isnan(h1) or np.isnan(h2):
        return float('nan')
    return h1 - h2


def compute_spectral_tilt_sequence(
    audio: np.ndarray,
    sr: int = 16000,
    hop_length: int = 160,
    frame_length: int = 400,
    f0_hz: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Compute spectral tilt metrics for each frame.

    Args:
        audio:        (T,) float32 mono audio
        sr:           sample rate
        hop_length:   samples between frames
        frame_length: analysis window in samples
        f0_hz:        (frames,) optional f0 for H1–H2 computation

    Returns:
        dict with keys:
          'slope':  (frames,) spectral tilt slope in dB/octave
          'h1_h2':  (frames,) H1–H2 in dB (NaN where f0 unavailable)
    """
    pad = frame_length // 2
    padded = np.pad(audio, (pad, pad), mode='reflect')

    n_frames = 1 + (len(audio) - 1) // hop_length
    slopes = np.zeros(n_frames, dtype=np.float32)
    h1_h2s = np.full(n_frames, np.nan, dtype=np.float32)

    for i in range(n_frames):
        start = i * hop_length
        frame = padded[start: start + frame_length]
        if len(frame) < frame_length:
            frame = np.pad(frame, (0, frame_length - len(frame)))

        slopes[i] = compute_spectral_tilt_slope(frame, sr=sr)

        if f0_hz is not None and i < len(f0_hz):
            h1_h2s[i] = compute_h1_h2(frame, float(f0_hz[i]), sr=sr)

    return {"slope": slopes, "h1_h2": h1_h2s}
