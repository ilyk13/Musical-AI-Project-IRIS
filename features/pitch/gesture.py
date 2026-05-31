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
    """4.5–8 Hz modulation on the f0 track (genuine vibrato band, matches training labels)."""
    mask = np.zeros(len(f0), dtype=bool)
    voiced = f0 > 0
    if voiced.sum() < min_voiced:
        return mask

    log_f0 = np.zeros_like(f0, dtype=np.float64)
    log_f0[voiced] = np.log2(f0[voiced])
    log_f0 = _interpolate_unvoiced(log_f0, voiced)

    sr_frame = 100.0
    nyq = sr_frame / 2.0
    lo, hi = 4.5 / nyq, 8.0 / nyq
    b, a = butter(2, [lo, hi], btype="band")
    try:
        mod = filtfilt(b, a, log_f0)
    except ValueError:
        return mask

    energy = np.abs(mod)
    if energy.max() <= 0:
        return mask

    # 65th percentile matches training label generation more closely than 40th
    thr = np.percentile(energy[voiced], 65)
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


def pitch_posterior_entropy(posterior: np.ndarray) -> np.ndarray:
    """Normalized pitch posterior entropy — high value means model is uncertain.

    A confident pitched note produces a narrow peak (low entropy); a frame
    mid-transition produces a flat or bimodal posterior (high entropy). This
    bridges the training/runtime gap: the model learns from expert CSV
    transition annotations, and entropy spikes serve as a proxy at runtime.

    Returns (T,) float array in [0, 1], normalized by log(num_bins).
    """
    p = posterior / (posterior.sum(axis=-1, keepdims=True) + 1e-10)
    raw = -np.sum(p * np.log(p + 1e-10), axis=-1)
    return raw / np.log(max(posterior.shape[-1], 2))


# Minimum credible run length per gesture class (frames at 10 ms/frame).
# Runs shorter than these are collapsed into their surrounding context.
_MIN_DURATION_FRAMES: dict[int, int] = {
    GESTURE_VIBRATO: 10,     # < 100 ms is noise — half a vibrato cycle at 5 Hz
    GESTURE_TRANSITION: 3,   # allow brief 30 ms transitions
    GESTURE_GLISSANDO: 6,    # at least 60 ms of sustained slide
    GESTURE_STEADY: 5,       # prevent single-frame dropouts from steady
}


def enforce_min_duration(labels: np.ndarray) -> np.ndarray:
    """Collapse runs shorter than the per-gesture minimum into their neighbors.

    Scans the label sequence for runs that are too short to be credible (e.g.
    a 2-frame "vibrato" burst, a single-frame "transition") and replaces them
    with the label of the preceding frame (or the following frame at the start).
    """
    out = labels.copy()
    i = 0
    while i < len(out):
        g = int(out[i])
        j = i + 1
        while j < len(out) and int(out[j]) == g:
            j += 1
        run_len = j - i
        if run_len < _MIN_DURATION_FRAMES.get(g, 0):
            fill = int(out[i - 1]) if i > 0 else (int(out[j]) if j < len(out) else GESTURE_STEADY)
            out[i:j] = fill
        i = j
    return out


def classify_gestures_live(
    f0: np.ndarray,
    posteriorgram: np.ndarray | None = None,
    entropy_threshold: float = 0.62,
) -> np.ndarray:
    """Per-frame gesture labels from an f0 track (T,) in Hz.

    Args:
        f0: (T,) f0 track in Hz, 0 = unvoiced.
        posteriorgram: optional (K, bins) pitch posterior for the last K frames
            of f0.  When provided, high-entropy frames are added to the
            transition mask — bridging the CSV-annotation gap at runtime.
        entropy_threshold: normalized entropy above which a voiced frame is
            treated as a transition (0–1 scale; 0.62 ≈ ~10× uniform over
            narrow peak).
    """
    f0 = np.asarray(f0, dtype=np.float64)
    n = len(f0)
    labels = np.full(n, GESTURE_STEADY, dtype=np.int8)

    transition = detect_transition_frames(f0)

    if posteriorgram is not None and len(posteriorgram) > 0:
        post_arr = np.asarray(posteriorgram, dtype=np.float32)
        k = min(len(post_arr), n)
        entropy = pitch_posterior_entropy(post_arr[:k])
        voiced_tail = f0[n - k:] > 0
        transition[n - k:] |= (entropy > entropy_threshold) & voiced_tail

    vibrato = detect_vibrato_frames(f0.astype(np.float32))
    glissando = detect_glissando_frames(f0, transition)

    labels[glissando & ~transition] = GESTURE_GLISSANDO
    labels[vibrato & ~transition & ~glissando] = GESTURE_VIBRATO
    labels[transition] = GESTURE_TRANSITION

    return enforce_min_duration(labels)


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
