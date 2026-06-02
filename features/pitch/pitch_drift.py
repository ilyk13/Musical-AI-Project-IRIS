"""Pitch drift analysis — score how well a singer holds their pitch.

"Drift" means the gradual movement of pitch away from the intended note
over the course of a sustained tone. A singer who starts a note on pitch
but slowly goes flat (or sharp) has high drift; one who stays centered
throughout has low drift.

We analyze each voiced segment between note onsets / breath events,
using the central pitch as the "intended" reference.
"""

import numpy as np
from dataclasses import dataclass, field


@dataclass
class DriftResult:
    """Drift analysis for one sustained note segment.

    Attributes:
        start_frame:    first frame of the segment
        end_frame:      last frame (exclusive)
        duration_s:     segment length in seconds
        mean_dev_cents: mean absolute deviation from central pitch (cents)
        max_dev_cents:  maximum absolute deviation within the segment (cents)
        trend_cents_s:  linear drift slope in cents/second
                        (positive = drifting sharp, negative = flat)
        score:          0–100, higher is better (less drift)
    """
    start_frame: int
    end_frame: int
    duration_s: float
    mean_dev_cents: float
    max_dev_cents: float
    trend_cents_s: float
    score: float


def analyze_drift(
    f0_hz: np.ndarray,
    central_hz: np.ndarray,
    vad: np.ndarray | None = None,
    scored_mask: np.ndarray | None = None,
    frame_hop_s: float = 0.010,
    min_duration_s: float = 0.15,
    max_score_cents: float = 55.0,
) -> list[DriftResult]:
    """Analyse pitch drift across voiced segments (optionally steady frames only).

    Segments shorter than `min_duration_s` are skipped (too brief to score).

    Args:
        f0_hz:         (T,) raw f0 from NanoPitch (0 = unvoiced)
        central_hz:    (T,) smoothed central pitch (NaN = unvoiced)
        vad:           (T,) optional VAD probabilities; if None, derived from f0_hz
        scored_mask:   (T,) optional bool — restrict to steady/scored frames
        frame_hop_s:   seconds per frame (default 10 ms)
        min_duration_s: minimum segment length to score
        max_score_cents: deviation (cents) that maps to a score of 0

    Returns:
        list of DriftResult, one per scoreable voiced segment
    """
    voiced = (f0_hz > 0) & ~np.isnan(central_hz)
    if vad is not None:
        voiced &= vad > 0.3
    if scored_mask is not None:
        voiced &= np.asarray(scored_mask, dtype=bool)

    segments = _get_voiced_segments(voiced)
    min_frames = int(round(min_duration_s / frame_hop_s))

    results = []
    for start, end in segments:
        if (end - start) < min_frames:
            continue

        f0_seg = f0_hz[start:end]
        central_seg = central_hz[start:end]
        valid = (f0_seg > 0) & ~np.isnan(central_seg)

        if valid.sum() < 3:
            continue

        # Convert to cents
        with np.errstate(divide='ignore', invalid='ignore'):
            f0_cents = 1200.0 * np.log2(f0_seg[valid] / central_seg[valid])

        mean_dev = float(np.mean(np.abs(f0_cents)))
        max_dev = float(np.max(np.abs(f0_cents)))

        # Linear trend: slope = drift direction and rate
        t = np.where(valid)[0].astype(np.float64) * frame_hop_s
        if len(t) >= 2:
            coeffs = np.polyfit(t, f0_cents, 1)
            trend = float(coeffs[0])  # cents per second
        else:
            trend = 0.0

        duration = (end - start) * frame_hop_s
        score = max(0.0, 100.0 * (1.0 - mean_dev / max_score_cents))

        results.append(DriftResult(
            start_frame=start,
            end_frame=end,
            duration_s=duration,
            mean_dev_cents=mean_dev,
            max_dev_cents=max_dev,
            trend_cents_s=trend,
            score=score,
        ))

    return results


def aggregate_drift_score(results: list[DriftResult]) -> float:
    """Compute a single pitch-drift score (0–100) across all segments.

    Weighted by segment duration so longer notes contribute more.
    Returns 100.0 if there are no scoreable segments.
    """
    if not results:
        return 100.0

    total_weight = sum(r.duration_s for r in results)
    if total_weight == 0:
        return 100.0

    weighted_score = sum(r.score * r.duration_s for r in results)
    return float(weighted_score / total_weight)


def _get_voiced_segments(voiced: np.ndarray) -> list[tuple[int, int]]:
    """Return (start, end) index pairs for each contiguous voiced run."""
    segments = []
    in_segment = False
    start = 0
    for i, v in enumerate(voiced):
        if v and not in_segment:
            start = i
            in_segment = True
        elif not v and in_segment:
            segments.append((start, i))
            in_segment = False
    if in_segment:
        segments.append((start, len(voiced)))
    return segments
