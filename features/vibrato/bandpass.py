"""Vibrato isolation via band-pass filtering of the f0 track.

Vibrato is a periodic variation in pitch typically at 4.5–8 Hz with
a depth of 20–100 cents. To isolate it, we:
  1. Convert f0 to cents (log scale, perceptually uniform)
  2. Remove the slowly-varying central pitch trend (high-pass effect)
  3. Apply a band-pass filter in the vibrato frequency range

The result is the vibrato deviation signal — how many cents above/below
the central pitch the voice is at each moment.
"""

import numpy as np
from scipy import signal


VIBRATO_FREQ_MIN = 3.0   # Hz — below this is slow wobble / drift
VIBRATO_FREQ_MAX = 9.0   # Hz — above this is flutter / tremolo


def compute_vibrato_deviation(
    f0_hz: np.ndarray,
    central_hz: np.ndarray,
) -> np.ndarray:
    """Compute the deviation of f0 from central pitch in cents.

    This is the raw (unfiltered) pitch deviation — a combination of vibrato,
    drift, and measurement noise. Use `bandpass_vibrato` to isolate just
    the vibrato component.

    Args:
        f0_hz:      (T,) raw f0 in Hz (0 or NaN = unvoiced)
        central_hz: (T,) smoothed central pitch in Hz (NaN = unvoiced)

    Returns:
        deviation: (T,) float32 in cents (NaN where unvoiced)
    """
    voiced = (f0_hz > 0) & ~np.isnan(central_hz) & (central_hz > 0)
    deviation = np.full(len(f0_hz), np.nan, dtype=np.float32)

    with np.errstate(divide='ignore', invalid='ignore'):
        deviation[voiced] = (
            1200.0 * np.log2(f0_hz[voiced] / central_hz[voiced])
        ).astype(np.float32)

    return deviation


def bandpass_vibrato(
    f0_hz: np.ndarray,
    central_hz: np.ndarray | None = None,
    frame_rate_hz: float = 100.0,
    freq_min: float = VIBRATO_FREQ_MIN,
    freq_max: float = VIBRATO_FREQ_MAX,
) -> np.ndarray:
    """Isolate vibrato as cents above/below central pitch, band-passed in rate."""
    f0_hz = np.asarray(f0_hz, dtype=np.float64)
    voiced = f0_hz > 0
    if voiced.sum() < 10:
        return np.full(len(f0_hz), np.nan, dtype=np.float32)

    if central_hz is not None:
        central_hz = np.asarray(central_hz, dtype=np.float64)
        dev = compute_vibrato_deviation(f0_hz.astype(np.float32), central_hz.astype(np.float32))
        signal_in = _fill_nan_linear(dev.astype(np.float64))
    else:
        cents = np.full(len(f0_hz), np.nan, dtype=np.float64)
        cents[voiced] = 1200.0 * np.log2(f0_hz[voiced] / 440.0)
        signal_in = _fill_nan_linear(cents)

    nyq = frame_rate_hz / 2.0
    lo = freq_min / nyq
    hi = min(freq_max / nyq, 0.99)

    if lo >= hi:
        return np.full(len(f0_hz), np.nan, dtype=np.float32)

    sos = signal.butter(4, [lo, hi], btype='bandpass', output='sos')
    filtered = signal.sosfiltfilt(sos, signal_in)

    result = np.where(voiced, filtered, np.nan)
    return result.astype(np.float32)


def _fill_nan_linear(x: np.ndarray) -> np.ndarray:
    """Fill NaN values with linear interpolation."""
    x = x.copy()
    nans = np.isnan(x)
    if not nans.any():
        return x
    idx = np.arange(len(x))
    x[nans] = np.interp(idx[nans], idx[~nans], x[~nans])
    return x
