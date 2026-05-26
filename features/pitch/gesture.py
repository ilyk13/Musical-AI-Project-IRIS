"""Live gesture classification for gesture-aware pitch scoring.

Classifies each 10 ms frame as steady, vibrato, glissando, or transition.
Uses f0-track heuristics by default (no trained model required). When a
NanoPitchPlus checkpoint is loaded in the app, model predictions override
heuristics frame-by-frame.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, filtfilt

GESTURE_VOCAB = ["steady", "vibrato", "glissando", "transition"]
GESTURE_STEADY = 0
GESTURE_VIBRATO = 1
GESTURE_GLISSANDO = 2
GESTURE_TRANSITION = 3

# Frames with these gestures are excluded from equal-temperament accuracy.
SCORED_GESTURES = frozenset({GESTURE_STEADY})


def gesture_name(idx: int) -> str:
    if 0 <= idx < len(GESTURE_VOCAB):
        return GESTURE_VOCAB[idx]
    return "unknown"


def is_scored_gesture(idx: int) -> bool:
    return int(idx) in SCORED_GESTURES


def _interpolate_unvoiced(log_f0: np.ndarray, voiced: np.ndarray) -> np.ndarray:
    out = log_f0.copy()
    if voiced.sum() < 2:
        return out
    idx = np.arange(len(log_f0))
    out[~voiced] = np.interp(idx[~voiced], idx[voiced], log_f0[voiced])
    return out


def detect_vibrato_frames(f0: np.ndarray, min_voiced: int = 25) -> np.ndarray:
    """3–9 Hz modulation on the f0 track (live singing + deliberate oscillation)."""
    mask = np.zeros(len(f0), dtype=bool)
    voiced = f0 > 0
    if voiced.sum() < min_voiced:
        return mask

    log_f0 = np.zeros_like(f0, dtype=np.float64)
    log_f0[voiced] = np.log2(f0[voiced])
    log_f0 = _interpolate_unvoiced(log_f0, voiced)

    sr_frame = 100.0
    nyq = sr_frame / 2.0
    energy = np.zeros(len(f0), dtype=np.float64)
    for lo_hz, hi_hz in ((3.0, 9.0), (4.5, 8.0)):
        lo, hi = lo_hz / nyq, min(hi_hz / nyq, 0.99)
        if lo >= hi:
            continue
        b, a = butter(2, [lo, hi], btype="band")
        try:
            mod = filtfilt(b, a, log_f0)
            energy = np.maximum(energy, np.abs(mod))
        except ValueError:
            continue

    if energy.max() <= 0:
        return mask

    thr = np.percentile(energy[voiced], 40)
    depth_thr = 0.012  # ~14 cents in log2 domain
    return (energy > max(thr, depth_thr)) & voiced


def detect_transition_frames(f0: np.ndarray, jump_cents: float = 90.0) -> np.ndarray:
    """Wide regions around rapid pitch jumps (note changes / scoops)."""
    n = len(f0)
    mask = np.zeros(n, dtype=bool)
    for t in range(1, n):
        if f0[t] <= 0 or f0[t - 1] <= 0:
            continue
        jump = abs(1200.0 * np.log2(f0[t] / (f0[t - 1] + 1e-10) + 1e-10))
        if jump >= jump_cents:
            lo = max(0, t - 3)
            hi = min(n, t + 4)
            mask[lo:hi] = True
    return mask


def detect_glissando_frames(
    f0: np.ndarray,
    transition_mask: np.ndarray,
    min_slope_cents: float = 6.0,
    min_run: int = 4,
) -> np.ndarray:
    """Sustained monotonic f0 motion (slides between notes)."""
    n = len(f0)
    mask = np.zeros(n, dtype=bool)
    if n < min_run:
        return mask

    log_f0 = np.zeros(n, dtype=np.float64)
    voiced = f0 > 0
    log_f0[voiced] = np.log2(f0[voiced] + 1e-10)
    log_f0 = _interpolate_unvoiced(log_f0, voiced)
    slope = np.diff(log_f0, prepend=log_f0[0]) * 1200.0

    run = 0
    sign = 0
    for t in range(n):
        if not voiced[t] or transition_mask[t]:
            run = 0
            sign = 0
            continue
        s = slope[t]
        if abs(s) < min_slope_cents:
            run = 0
            sign = 0
            continue
        cur_sign = 1 if s > 0 else -1
        if cur_sign == sign:
            run += 1
        else:
            run = 1
            sign = cur_sign
        if run >= min_run:
            mask[t - min_run + 1: t + 1] = True
    return mask


def classify_gestures_live(f0: np.ndarray) -> np.ndarray:
    """Per-frame gesture labels from an f0 track (T,) in Hz."""
    f0 = np.asarray(f0, dtype=np.float64)
    n = len(f0)
    labels = np.full(n, GESTURE_STEADY, dtype=np.int8)

    transition = detect_transition_frames(f0)
    vibrato = detect_vibrato_frames(f0.astype(np.float32))
    glissando = detect_glissando_frames(f0, transition)

    labels[glissando & ~transition] = GESTURE_GLISSANDO
    labels[vibrato & ~transition & ~glissando] = GESTURE_VIBRATO
    labels[transition] = GESTURE_TRANSITION
    return labels


def provisional_f0_from_posteriors(posteriorgram: np.ndarray) -> np.ndarray:
    """Argmax decode for gesture estimation before Viterbi refines f0."""
    from model.nanopitch import bin_to_f0

    if len(posteriorgram) == 0:
        return np.zeros(0, dtype=np.float32)
    bins = posteriorgram.argmax(axis=1)
    max_p = posteriorgram.max(axis=1)
    f0 = bin_to_f0(bins.astype(np.float64)).astype(np.float32)
    f0[max_p < 0.08] = 0.0
    return f0


def merge_gesture_predictions(
    heuristic: np.ndarray,
    model_logits: np.ndarray | None,
    model_confidence: float = 0.40,
) -> np.ndarray:
    """Blend model + f0 heuristics — non-steady heuristic wins over model steady."""
    if model_logits is None or len(model_logits) == 0:
        return heuristic

    import torch
    logits = torch.from_numpy(np.asarray(model_logits, dtype=np.float32))
    probs = torch.softmax(logits, dim=-1).numpy()
    pred = probs.argmax(axis=-1).astype(np.int8)
    conf = probs.max(axis=-1)

    out = heuristic.copy()
    for i in range(len(out)):
        h, m, c = int(heuristic[i]), int(pred[i]), float(conf[i])
        vib_conf = float(probs[i, GESTURE_VIBRATO]) if probs.shape[1] > GESTURE_VIBRATO else 0.0
        # Heuristic vibrato/glissando/transition beats model "steady".
        if h != GESTURE_STEADY and m == GESTURE_STEADY:
            out[i] = h
        elif m == GESTURE_VIBRATO and vib_conf >= 0.30:
            out[i] = GESTURE_VIBRATO
        elif m != GESTURE_STEADY and c >= model_confidence:
            out[i] = m
        elif c >= 0.60:
            out[i] = m
    return out


def overlay_vibrato_from_deviation(
    gesture: np.ndarray,
    f0: np.ndarray,
    vib_cents: np.ndarray,
    threshold: float = 6.0,
    win: int = 25,
    std_thr: float = 4.0,
) -> np.ndarray:
    """Promote steady → vibrato when band-pass deviation matches the vibrato chart."""
    out = gesture.copy()
    n = min(len(out), len(f0), len(vib_cents))
    for i in range(n):
        if f0[i] <= 0 or out[i] == GESTURE_TRANSITION:
            continue
        lo = max(0, i - win + 1)
        seg = vib_cents[lo:i + 1]
        valid = ~np.isnan(seg)
        if valid.sum() < 8:
            continue
        seg = seg[valid]
        if np.max(np.abs(seg)) >= threshold or np.std(seg) >= std_thr:
            out[i] = GESTURE_VIBRATO
    return out


def smooth_gesture_label(
    recent: np.ndarray,
    vib_recent: np.ndarray | None = None,
    vib_threshold: float = 6.0,
) -> int:
    """Stable UI label — uses vibrato deviation when available."""
    if vib_recent is not None and len(vib_recent) > 0:
        vib = np.asarray(vib_recent, dtype=np.float64)
        valid = ~np.isnan(vib)
        if valid.sum() >= 12:
            active = valid & (np.abs(vib) >= vib_threshold)
            if active.sum() >= max(4, int(valid.sum() * 0.08)):
                return GESTURE_VIBRATO
            seg = vib[valid]
            if len(seg) >= 20 and np.std(seg) >= 4.0:
                return GESTURE_VIBRATO

    if len(recent) == 0:
        return GESTURE_STEADY
    n_vib = int(np.sum(recent == GESTURE_VIBRATO))
    if n_vib >= max(3, int(len(recent) * 0.06)):
        return GESTURE_VIBRATO
    vals, counts = np.unique(recent, return_counts=True)
    return int(vals[counts.argmax()])
