"""Vibrato parameter extraction — rate, depth, and consistency.

Given the band-pass filtered vibrato signal, we extract the key
perceptual parameters:

  - Rate (Hz): how many oscillations per second (typically 4.5–8 Hz)
  - Depth (cents): peak-to-peak amplitude (typically 20–100 cents)
  - Consistency: how stable the vibrato is across a phrase (0–1)

These are used both for visualization and for scoring.
"""

import numpy as np
from dataclasses import dataclass, field
from scipy import signal


# Reference ranges for "good" professional vibrato
IDEAL_RATE_MIN = 5.0   # Hz
IDEAL_RATE_MAX = 7.0   # Hz
IDEAL_DEPTH_MIN = 25.0  # cents (peak-to-peak / 2 = amplitude in cents)
IDEAL_DEPTH_MAX = 75.0  # cents


@dataclass
class VibratoParams:
    """Extracted vibrato parameters for a voiced segment.

    Attributes:
        start_frame:   first frame of the segment
        end_frame:     last frame (exclusive)
        duration_s:    segment length in seconds
        rate_hz:       dominant vibrato rate (Hz), NaN if no vibrato
        depth_cents:   peak-to-peak amplitude (cents), NaN if no vibrato
        consistency:   0–1, ratio of energy in the dominant rate bin
        has_vibrato:   True if vibrato energy is strong enough to classify
        rate_score:    0–100 based on how close rate is to ideal range
        depth_score:   0–100 based on how close depth is to ideal range
    """
    start_frame: int
    end_frame: int
    duration_s: float
    rate_hz: float
    depth_cents: float
    consistency: float
    has_vibrato: bool
    rate_score: float
    depth_score: float


def extract_vibrato_params(
    vibrato_signal: np.ndarray,
    frame_rate_hz: float = 100.0,
    min_duration_s: float = 0.3,
    vibrato_energy_threshold: float = 10.0,
) -> VibratoParams | None:
    """Extract vibrato parameters from a band-pass filtered vibrato signal.

    Args:
        vibrato_signal:          (T,) vibrato deviation in cents (NaN = unvoiced)
        frame_rate_hz:           frames per second
        min_duration_s:          skip segments shorter than this
        vibrato_energy_threshold: minimum RMS amplitude (cents) to classify as vibrato

    Returns:
        VibratoParams, or None if the segment is too short
    """
    valid = ~np.isnan(vibrato_signal)
    n_valid = valid.sum()
    n_frames = len(vibrato_signal)
    duration_s = n_frames / frame_rate_hz

    if duration_s < min_duration_s or n_valid < 10:
        return None

    sig = vibrato_signal[valid]

    # RMS amplitude — if too weak, it's not really vibrato
    rms_cents = float(np.sqrt(np.mean(sig ** 2)))
    has_vibrato = rms_cents >= vibrato_energy_threshold

    if not has_vibrato:
        return VibratoParams(
            start_frame=0, end_frame=n_frames, duration_s=duration_s,
            rate_hz=float('nan'), depth_cents=float('nan'),
            consistency=0.0, has_vibrato=False,
            rate_score=0.0, depth_score=0.0,
        )

    # FFT to find dominant rate
    n_fft = max(256, 2 ** int(np.ceil(np.log2(len(sig)))))
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / frame_rate_hz)
    spectrum = np.abs(np.fft.rfft(sig, n=n_fft)) ** 2

    # Restrict to vibrato frequency range
    vib_mask = (freqs >= 4.0) & (freqs <= 9.0)
    if not vib_mask.any():
        rate_hz = float('nan')
        consistency = 0.0
    else:
        peak_idx = np.argmax(spectrum[vib_mask])
        rate_hz = float(freqs[vib_mask][peak_idx])
        consistency = float(spectrum[vib_mask][peak_idx] / (spectrum.sum() + 1e-10))

    # Depth: peak-to-peak amplitude in cents
    depth_cents = float(np.max(sig) - np.min(sig))

    rate_score = _range_score(rate_hz, IDEAL_RATE_MIN, IDEAL_RATE_MAX)
    depth_score = _range_score(depth_cents / 2.0, IDEAL_DEPTH_MIN / 2.0, IDEAL_DEPTH_MAX / 2.0)

    return VibratoParams(
        start_frame=0, end_frame=n_frames, duration_s=duration_s,
        rate_hz=rate_hz, depth_cents=depth_cents,
        consistency=consistency, has_vibrato=True,
        rate_score=rate_score, depth_score=depth_score,
    )


def _range_score(value: float, lo: float, hi: float) -> float:
    """Score 0–100: 100 if value is in [lo, hi], decays outside."""
    if np.isnan(value):
        return 0.0
    if lo <= value <= hi:
        return 100.0
    if value < lo:
        deficit = lo - value
        scale = lo
    else:
        deficit = value - hi
        scale = hi
    return float(max(0.0, 100.0 * (1.0 - deficit / scale)))
