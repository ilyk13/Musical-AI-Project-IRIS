"""Pitch accuracy scoring shared by live phrase feedback and session analysis."""

from __future__ import annotations

import numpy as np

from features.pitch.gesture import GESTURE_GLISSANDO, GESTURE_TRANSITION


def _scoring_mask(f0: np.ndarray, scored: np.ndarray) -> np.ndarray:
    """Steady frames for ET/bias; avoid mixing in transition/gliss when steady exist."""
    voiced = f0 > 0
    steady = scored & voiced
    if int(steady.sum()) >= 3:
        return steady
    return voiced


def aggregate_pitch_deviation_cents(
    et_dev: np.ndarray,
    f0: np.ndarray,
    scored: np.ndarray,
) -> float:
    """Typical ET deviation (¢) — median-heavy so occasional spikes don't dominate."""
    mask = _scoring_mask(f0, scored)
    if int(mask.sum()) < 3:
        return 50.0
    abs_d = np.abs(et_dev[mask])
    return float(
        0.45 * np.median(abs_d)
        + 0.40 * np.mean(abs_d)
        + 0.15 * np.percentile(abs_d, 90)
    )


def systematic_bias_cents(et_dev: np.ndarray, f0: np.ndarray, scored: np.ndarray) -> float:
    """Signed median ET deviation — consistent flat/sharp (not occasional spikes)."""
    mask = _scoring_mask(f0, scored)
    if int(mask.sum()) < 3:
        return 0.0
    return float(np.median(et_dev[mask]))


def pitch_bias_score(bias_cents: float) -> float:
    """0–100; penalizes sustained sharp/flat (tracking noise below ~12¢ is ignored)."""
    b = abs(float(bias_cents))
    if b <= 12.0:
        return float(96.0 + 0.33 * b)
    b = min(b, 45.0)
    t = min((b - 12.0) / 30.0, 1.0)
    return float(max(0.0, 100.0 * (1.0 - t ** 1.15)))


def f0_instability_cents(f0: np.ndarray, mask: np.ndarray) -> float:
    """85th percentile of frame-to-frame pitch steps (¢) — catches drift / wandering."""
    idx = np.where(mask & (f0 > 0))[0]
    if len(idx) < 4:
        return 0.0
    logf = np.log2(f0[idx].astype(np.float64) + 1e-12)
    steps = np.abs(np.diff(logf)) * 1200.0
    return float(np.percentile(steps, 85))


def pitch_accuracy_score(mean_dev_cents: float) -> float:
    """0–100 from typical cents off the nearest note."""
    d = min(max(float(mean_dev_cents), 0.0), 50.0)
    t = min(d / 54.0, 1.0)
    return float(max(0.0, 50.0 + 50.0 * (1.0 - t ** 1.08)))


def instability_score(step_p85_cents: float) -> float:
    """0–100 from frame-to-frame pitch motion (lower when pitch is jumping around)."""
    j = min(max(float(step_p85_cents), 0.0), 55.0)
    return float(max(0.0, 100.0 - j * 1.65))


def slide_wander_cents(f0: np.ndarray, gestures: np.ndarray) -> float:
    """How choppy gliss/transition motion is (¢ scale) — not ET distance from a note."""
    g = gestures.astype(np.int8)
    slide = (g == GESTURE_GLISSANDO) | (g == GESTURE_TRANSITION)
    idx = np.where(slide & (f0 > 0))[0]
    if len(idx) < 4:
        return 0.0
    logf = np.log2(f0[idx].astype(np.float64) + 1e-12)
    steps = np.diff(logf) * 1200.0
    abs_steps = np.abs(steps)
    if len(abs_steps) < 2:
        return float(abs_steps[0]) if len(abs_steps) else 0.0
    signed = steps[np.abs(steps) > 0.5]
    rev = 0.0
    if len(signed) >= 2:
        sgn = np.sign(signed)
        rev = float(np.mean(sgn[1:] * sgn[:-1] < 0))
    step_p85 = float(np.percentile(abs_steps, 85))
    mean_abs = float(np.mean(abs_steps)) + 1e-6
    choppy = float(np.std(abs_steps)) / mean_abs
    return float(0.50 * step_p85 + 0.30 * min(choppy * 18.0, 38.0) + 0.20 * rev * 40.0)


def slide_wander_from_steps(steps: list[float]) -> float:
    """Live phrase tracker — step sizes (¢) on gliss/transition frames only."""
    if len(steps) < 3:
        return 0.0
    a = np.asarray(steps, dtype=np.float64)
    step_p85 = float(np.percentile(a, 85))
    mean_abs = float(np.mean(a)) + 1e-6
    choppy = float(np.std(a)) / mean_abs
    return float(0.62 * step_p85 + 0.38 * min(choppy * 18.0, 38.0))


def _slide_fraction(f0: np.ndarray, gestures: np.ndarray) -> float:
    g = gestures.astype(np.int8)
    voiced = f0 > 0
    n_voiced = int(voiced.sum())
    if n_voiced < 4:
        return 0.0
    slide = ((g == GESTURE_GLISSANDO) | (g == GESTURE_TRANSITION)) & voiced
    return float(slide.sum()) / float(n_voiced)


def slide_control_score(wander_cents: float) -> float:
    """0–100; penalizes messy gliss/transition (smooth slides stay high)."""
    w = min(max(float(wander_cents), 0.0), 85.0)
    if w <= 30.0:
        return 100.0
    t = min((w - 30.0) / 45.0, 1.0)
    return float(max(55.0, 100.0 - 38.0 * (t ** 0.95)))


def combined_pitch_score(
    et_dev: np.ndarray,
    f0: np.ndarray,
    scored: np.ndarray,
    drift_score: float,
    gestures: np.ndarray | None = None,
) -> float:
    """Typical on-pitch accuracy + bias + drift + slide control on gliss/transition."""
    steady = scored & (f0 > 0)
    inst_mask = steady if int(steady.sum()) >= 8 else (f0 > 0)

    et_metric = aggregate_pitch_deviation_cents(et_dev, f0, scored)
    et_score = pitch_accuracy_score(et_metric)
    bias_score = pitch_bias_score(systematic_bias_cents(et_dev, f0, scored))
    jitter = f0_instability_cents(f0, inst_mask)
    inst_score = instability_score(jitter)

    drift_s = float(np.clip(drift_score, 0.0, 100.0))
    slide_sc = 100.0
    slide_w = 0.0
    if gestures is not None:
        slide_frac = _slide_fraction(f0, gestures)
        wander = slide_wander_cents(f0, gestures)
        if slide_frac >= 0.08 and wander > 0.0:
            slide_sc = slide_control_score(wander)
            # Only weight slides when they are a real part of the phrase.
            slide_w = 0.12 * min(max((slide_frac - 0.06) / 0.16, 0.0), 1.0)

    core_w = 1.0 - slide_w - 0.01
    core_scale = core_w / 0.87  # et + bias + drift base weights sum to 0.87
    return float(
        core_scale * (0.52 * et_score + 0.20 * bias_score + 0.15 * drift_s)
        + slide_w * slide_sc
        + 0.01 * inst_score
    )


def phrase_pitch_score_from_deviations(
    deviations_cents: list[float],
    slide_steps: list[float] | None = None,
) -> float:
    """Live phrase end — steady ET accuracy + optional gliss/transition slide control."""
    if len(deviations_cents) < 3:
        return 70.0
    d = np.asarray(deviations_cents, dtype=np.float64)
    abs_d = np.abs(d)
    et_metric = float(
        0.45 * np.median(abs_d) + 0.40 * np.mean(abs_d) + 0.15 * np.percentile(abs_d, 90)
    )
    et_score = pitch_accuracy_score(et_metric)
    bias_score = pitch_bias_score(float(np.median(d)))
    core = float(0.68 * et_score + 0.32 * bias_score)
    n_slide = len(slide_steps) if slide_steps else 0
    n_steady = len(deviations_cents)
    slide_frac = n_slide / max(n_slide + n_steady, 1)
    if slide_steps and n_slide >= 4 and slide_frac >= 0.08:
        wander = slide_wander_from_steps(slide_steps)
        if wander > 0.0:
            slide_sc = slide_control_score(wander)
            slide_w = 0.12 * min(max((slide_frac - 0.06) / 0.16, 0.0), 1.0)
            return float((1.0 - slide_w) * core + slide_w * slide_sc)
    return core
