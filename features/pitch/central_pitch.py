"""Central pitch — a smoothed f0 reference line for scoring pitch accuracy.

The raw f0 from NanoPitch fluctuates with natural vibrato and micro-variations.
"Central pitch" removes those short-term fluctuations to reveal the intended
note, which is what we compare against equal temperament for scoring.

Two smoothing strategies:
  - median filter: robust to outliers, preserves note onsets
  - low-pass filter: smoother transitions, better for continuous glides
"""

import numpy as np
from scipy import signal, ndimage


def compute_central_pitch(
    f0_hz: np.ndarray,
    method: str = "median",
    median_ms: float = 150.0,
    lowpass_hz: float = 5.0,
    frame_hop_ms: float = 10.0,
) -> np.ndarray:
    """Compute a smoothed central pitch from a raw f0 track.

    Unvoiced frames (f0 == 0) are excluded from smoothing and returned as NaN
    in the output so downstream code can distinguish "no pitch" from "pitch=0".

    Args:
        f0_hz:       (T,) raw f0 in Hz from NanoPitch (0 = unvoiced)
        method:      'median' or 'lowpass'
        median_ms:   half-window length for median filter (milliseconds)
        lowpass_hz:  cutoff frequency for low-pass filter (Hz)
        frame_hop_ms: duration of each frame in ms (default 10 ms)

    Returns:
        central: (T,) float32 — smoothed f0 in Hz, NaN where unvoiced
    """
    f0 = f0_hz.astype(np.float64).copy()
    voiced = f0 > 0

    if not voiced.any():
        return np.full_like(f0, np.nan, dtype=np.float32)

    # Work in log-Hz (cent-like scale) so smoothing is perceptually uniform
    log_f0 = np.full_like(f0, np.nan)
    log_f0[voiced] = np.log2(f0[voiced])

    # Fill gaps for filtering (linear interpolation across unvoiced regions)
    log_f0_filled = _fill_nan_linear(log_f0)

    if method == "median":
        half_frames = max(1, int(round((median_ms / 2) / frame_hop_ms)))
        kernel_size = 2 * half_frames + 1
        smoothed = ndimage.median_filter(log_f0_filled, size=kernel_size, mode='nearest')
    elif method == "lowpass":
        fs = 1.0 / (frame_hop_ms / 1000.0)  # frame rate in Hz
        nyq = fs / 2.0
        if lowpass_hz >= nyq:
            smoothed = log_f0_filled
        else:
            sos = signal.butter(4, lowpass_hz / nyq, btype='low', output='sos')
            smoothed = signal.sosfiltfilt(sos, log_f0_filled)
    else:
        raise ValueError(f"Unknown method '{method}'. Use 'median' or 'lowpass'.")

    # Convert back to Hz and mask out unvoiced frames
    central = np.where(voiced, 2.0 ** smoothed, np.nan)
    return central.astype(np.float32)


def hz_to_cents(f0_hz: np.ndarray, reference_hz: float = 440.0) -> np.ndarray:
    """Convert Hz to cents relative to a reference frequency.

    Cents are a perceptually uniform pitch unit: 100 cents = 1 semitone.
    A4 (440 Hz) is used as the default reference.
    """
    f0 = np.asarray(f0_hz, dtype=np.float64)
    with np.errstate(divide='ignore', invalid='ignore'):
        cents = 1200.0 * np.log2(f0 / reference_hz)
    return np.where(f0 > 0, cents, np.nan).astype(np.float32)


def nearest_equal_temperament(f0_hz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Find the nearest equal-temperament note and deviation for each frame.

    Equal temperament divides each octave into 12 semitones of exactly
    100 cents each. Any pitch that lands more than 50 cents from the nearest
    semitone is considered "out of tune."

    Args:
        f0_hz: (T,) f0 in Hz (NaN or 0 for unvoiced)

    Returns:
        nearest_hz:    (T,) Hz of the nearest ET note (NaN for unvoiced)
        deviation_cents: (T,) signed deviation in cents from that note
                          positive = sharp, negative = flat
    """
    f0 = np.asarray(f0_hz, dtype=np.float64)
    voiced = (f0 > 0) & ~np.isnan(f0)

    nearest_hz = np.full_like(f0, np.nan)
    deviation_cents = np.full_like(f0, np.nan)

    if voiced.any():
        # Semitone number relative to A4 (440 Hz)
        semitones = 12.0 * np.log2(f0[voiced] / 440.0)
        nearest_semitone = np.round(semitones)
        nearest_hz[voiced] = 440.0 * 2.0 ** (nearest_semitone / 12.0)
        deviation_cents[voiced] = (semitones - nearest_semitone) * 100.0

    return nearest_hz.astype(np.float32), deviation_cents.astype(np.float32)


def _fill_nan_linear(x: np.ndarray) -> np.ndarray:
    """Fill NaN gaps with linear interpolation (extrapolates at edges)."""
    x = x.copy()
    nans = np.isnan(x)
    if not nans.any():
        return x
    idx = np.arange(len(x))
    x[nans] = np.interp(idx[nans], idx[~nans], x[~nans])
    return x
