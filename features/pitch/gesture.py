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


def is_phrase_pitch_frame(gesture: int) -> bool:
    """Steady voiced frames only — same gate as live ET accuracy scoring."""
    return int(gesture) in SCORED_GESTURES


def _interpolate_unvoiced(log_f0: np.ndarray, voiced: np.ndarray) -> np.ndarray:
    out = log_f0.copy()
    if voiced.sum() < 2:
        return out
    idx = np.arange(len(log_f0))
    out[~voiced] = np.interp(idx[~voiced], idx[voiced], log_f0[voiced])
    return out


def detect_vibrato_frames(
    f0: np.ndarray,
    min_voiced: int = 25,
    *,
    percentile: float = 78.0,
    depth_thr: float = 0.015,
) -> np.ndarray:
    """4.5–8 Hz modulation on the f0 track (genuine vibrato band)."""
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

    thr = np.percentile(energy[voiced], percentile)
    return (energy > max(thr, depth_thr)) & voiced


def detect_transition_frames(f0: np.ndarray, jump_cents: float = 110.0) -> np.ndarray:
    """Wide regions around rapid pitch jumps (note changes / scoops)."""
    n = len(f0)
    mask = np.zeros(n, dtype=bool)
    for t in range(1, n):
        if f0[t] <= 0 or f0[t - 1] <= 0:
            continue
        jump = abs(1200.0 * np.log2(f0[t] / (f0[t - 1] + 1e-10) + 1e-10))
        if jump >= jump_cents:
            lo = max(0, t - 2)
            hi = min(n, t + 3)
            mask[lo:hi] = True
    return mask


def _validate_monotonic_run(
    log_f0: np.ndarray,
    run_start: int,
    run_end: int,
    min_total_cents: float,
    *,
    max_residual_std: float = 18.0,
    max_flip_ratio: float = 0.28,
) -> bool:
    """Reject wobbly runs — vibrato drift is not a slide."""
    if run_end <= run_start:
        return False
    seg = log_f0[run_start:run_end + 1]
    span_c = abs(seg[-1] - seg[0]) * 1200.0
    if span_c < min_total_cents:
        return False
    if len(seg) < 4:
        return True
    slopes = np.diff(seg) * 1200.0
    sig = slopes[np.abs(slopes) > 3.0]
    if len(sig) >= 3:
        signs = np.sign(sig)
        flips = int(np.sum(signs[1:] != signs[:-1]))
        if flips / max(len(signs) - 1, 1) > max_flip_ratio:
            return False
    t = np.linspace(0.0, 1.0, len(seg))
    trend = seg[0] + (seg[-1] - seg[0]) * t
    residual_std = float(np.std((seg - trend) * 1200.0))
    if residual_std > max(min_total_cents * 0.38, max_residual_std):
        return False
    return True


def detect_glissando_frames(
    f0: np.ndarray,
    transition_mask: np.ndarray,
    min_slope_cents: float = 6.0,
    min_run: int = 4,
    min_total_cents: float = 35.0,
    vibrato_mask: np.ndarray | None = None,
    validate_monotonic: bool = False,
) -> np.ndarray:
    """Sustained monotonic f0 motion (slides between notes).

    Excludes vibrato-modulated regions — oscillation is not a slide.
    """
    n = len(f0)
    mask = np.zeros(n, dtype=bool)
    if n < min_run:
        return mask

    log_f0 = np.zeros(n, dtype=np.float64)
    voiced = f0 > 0
    log_f0[voiced] = np.log2(f0[voiced] + 1e-10)
    log_f0 = _interpolate_unvoiced(log_f0, voiced)
    slope = np.diff(log_f0, prepend=log_f0[0]) * 1200.0

    vib = np.zeros(n, dtype=bool) if vibrato_mask is None else vibrato_mask[:n]

    def _commit_run(run_start: int, run_end: int) -> None:
        if run_end < run_start:
            return
        run_len = run_end - run_start + 1
        if run_len < min_run:
            return
        span = log_f0[run_end] - log_f0[run_start]
        if abs(span) * 1200.0 < min_total_cents:
            return
        if validate_monotonic and not _validate_monotonic_run(
            log_f0, run_start, run_end, min_total_cents,
        ):
            return
        mask[run_start:run_end + 1] = True

    run = 0
    sign = 0
    run_start = 0
    for t in range(n):
        if not voiced[t] or transition_mask[t] or vib[t]:
            if run >= min_run:
                _commit_run(run_start, t - 1)
            run = 0
            sign = 0
            continue
        s = slope[t]
        if abs(s) < min_slope_cents:
            if run >= min_run:
                _commit_run(run_start, t - 1)
            run = 0
            sign = 0
            continue
        cur_sign = 1 if s > 0 else -1
        if cur_sign == sign:
            run += 1
        else:
            run_start = t
            run = 1
            sign = cur_sign
        if run >= min_run:
            _commit_run(run_start, t)
    return mask


def detect_glissando_net_ramp(
    f0: np.ndarray,
    *,
    win: int = 18,
    min_net_cents: float = 32.0,
    min_directedness: float = 0.50,
    vibrato_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Net displacement over a sliding window — works on Viterbi and raw f0."""
    n = len(f0)
    mask = np.zeros(n, dtype=bool)
    voiced = f0 > 0
    if voiced.sum() < max(5, win // 2):
        return mask

    log_f0 = np.zeros(n, dtype=np.float64)
    log_f0[voiced] = np.log2(f0[voiced] + 1e-10)
    log_f0 = _interpolate_unvoiced(log_f0, voiced)
    vib = np.zeros(n, dtype=bool) if vibrato_mask is None else vibrato_mask[:n]

    for i in range(n):
        if not voiced[i] or vib[i]:
            continue
        lo = max(0, i - win + 1)
        seg = log_f0[lo:i + 1]
        if np.sum(seg > 0) < max(5, win // 3):
            continue
        deltas = np.diff(seg) * 1200.0
        if len(deltas) < 4:
            continue
        net = (seg[-1] - seg[0]) * 1200.0
        path = float(np.sum(np.abs(deltas)))
        if path <= 0:
            continue
        if abs(net) >= min_net_cents and abs(net) / path >= min_directedness:
            mask[lo:i + 1] = True
    return mask


def boost_glissando_labels(
    labels: np.ndarray,
    f0: np.ndarray,
    vibrato_mask: np.ndarray | None = None,
    *,
    net_kw: dict | None = None,
) -> np.ndarray:
    """Promote steady/transition frames when net f0 displacement looks like a slide."""
    out = labels.copy()
    vib = (
        (out == GESTURE_VIBRATO)
        if vibrato_mask is None
        else np.asarray(vibrato_mask[:len(out)], dtype=bool)
    )
    kw = net_kw if net_kw is not None else _LIVE_GLISS_NET
    ramp = detect_glissando_net_ramp(f0, vibrato_mask=vib, **kw)
    for i in range(len(out)):
        if ramp[i] and out[i] in (GESTURE_STEADY, GESTURE_TRANSITION) and not vib[i]:
            out[i] = GESTURE_GLISSANDO
    return out


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
    GESTURE_VIBRATO: 15,
    GESTURE_TRANSITION: 5,
    GESTURE_GLISSANDO: 18,
    GESTURE_STEADY: 5,
}

# Looser min-runs for offline CSV labels (noisy f0); live uses _MIN_DURATION_FRAMES.
_MIN_DURATION_TRAIN: dict[int, int] = {
    GESTURE_VIBRATO: 8,
    GESTURE_TRANSITION: 4,
    GESTURE_GLISSANDO: 8,
    GESTURE_STEADY: 5,
}

# Live inference — slow slides ~2 ¢/frame; monotonic check blocks vibrato drift.
_LIVE_VIBRATO = dict(percentile=72.0, depth_thr=0.012)
_LIVE_GLISS_VIB_BLOCK = dict(percentile=58.0, depth_thr=0.009)
_LIVE_GLISS = dict(
    min_slope_cents=3.5, min_run=10, min_total_cents=55.0, validate_monotonic=True,
)
_LIVE_GLISS_NET = dict(win=35, min_net_cents=55.0, min_directedness=0.62)

# Offline training labels — vibrato before glissando; tighter glissando gate.
_TRAIN_VIBRATO = dict(percentile=62.0, depth_thr=0.010)
_TRAIN_GLISS = dict(min_slope_cents=8.0, min_run=6, min_total_cents=30.0)


def _compose_gesture_labels(
    f0: np.ndarray,
    transition: np.ndarray,
    *,
    vibrato_kw: dict,
    glissando_kw: dict,
    min_duration: dict[int, int],
) -> np.ndarray:
    """Priority: transition > vibrato > glissando > steady."""
    f0 = np.asarray(f0, dtype=np.float64)
    n = len(f0)
    vibrato = detect_vibrato_frames(f0.astype(np.float32), **vibrato_kw)
    glissando = detect_glissando_frames(
        f0, transition, vibrato_mask=vibrato, **glissando_kw,
    )

    labels = np.full(n, GESTURE_STEADY, dtype=np.int8)
    labels[glissando & ~transition & ~vibrato] = GESTURE_GLISSANDO
    labels[vibrato & ~transition] = GESTURE_VIBRATO
    labels[transition] = GESTURE_TRANSITION
    return enforce_min_duration(labels, min_duration)


def enforce_min_duration(
    labels: np.ndarray,
    min_frames: dict[int, int] | None = None,
) -> np.ndarray:
    """Collapse runs shorter than the per-gesture minimum into their neighbors."""
    mins = min_frames if min_frames is not None else _MIN_DURATION_FRAMES
    out = labels.copy()
    i = 0
    while i < len(out):
        g = int(out[i])
        j = i + 1
        while j < len(out) and int(out[j]) == g:
            j += 1
        run_len = j - i
        if run_len < mins.get(g, 0):
            fill = int(out[i - 1]) if i > 0 else (int(out[j]) if j < len(out) else GESTURE_STEADY)
            out[i:j] = fill
        i = j
    return out


def classify_gestures_live(
    f0: np.ndarray,
    posteriorgram: np.ndarray | None = None,
    entropy_threshold: float = 0.78,
) -> np.ndarray:
    """Per-frame gesture labels from an f0 track (T,) in Hz.

    Args:
        f0: (T,) f0 track in Hz, 0 = unvoiced.
        posteriorgram: optional (K, bins) pitch posterior for the last K frames
            of f0.  When provided, high-entropy frames are added to the
            transition mask — bridging the CSV-annotation gap at runtime.
        entropy_threshold: normalized entropy above which a voiced frame is
            treated as a transition (0–1 scale; 0.78 ≈ very flat posterior).
    """
    f0 = np.asarray(f0, dtype=np.float64)
    n = len(f0)
    jump_t = detect_transition_frames(f0)
    vibrato = detect_vibrato_frames(f0.astype(np.float32), **_LIVE_VIBRATO)
    vib_block = detect_vibrato_frames(f0.astype(np.float32), **_LIVE_GLISS_VIB_BLOCK)
    # Slides often coincide with pitch jumps and flat posteriors — detect
    # glissando without blocking on those markers (same as offline labels).
    vib_excl = vibrato | vib_block
    glissando = detect_glissando_frames(
        f0, np.zeros(n, dtype=bool), vibrato_mask=vib_excl, **_LIVE_GLISS,
    )
    glissando |= detect_glissando_net_ramp(f0, vibrato_mask=vib_excl, **_LIVE_GLISS_NET)

    transition = jump_t.copy()
    if posteriorgram is not None and len(posteriorgram) > 0:
        post_arr = np.asarray(posteriorgram, dtype=np.float32)
        k = min(len(post_arr), n)
        entropy = pitch_posterior_entropy(post_arr[:k])
        voiced_tail = f0[n - k:] > 0
        transition[n - k:] |= (entropy > entropy_threshold) & voiced_tail

    transition &= ~glissando & ~vibrato

    labels = np.full(n, GESTURE_STEADY, dtype=np.int8)
    labels[glissando & ~transition & ~vibrato] = GESTURE_GLISSANDO
    labels[vibrato & ~transition] = GESTURE_VIBRATO
    labels[transition] = GESTURE_TRANSITION
    labels = coalesce_glissando_labels(labels)
    return enforce_min_duration(labels, _MIN_DURATION_FRAMES)


def coalesce_glissando_labels(labels: np.ndarray, max_bridge: int = 8) -> np.ndarray:
    """Merge short transition runs beside glissando into the slide."""
    out = labels.copy()
    i = 0
    while i < len(out):
        if int(out[i]) != GESTURE_TRANSITION:
            i += 1
            continue
        j = i + 1
        while j < len(out) and int(out[j]) == GESTURE_TRANSITION:
            j += 1
        run_len = j - i
        prev_g = int(out[i - 1]) if i > 0 else GESTURE_STEADY
        next_g = int(out[j]) if j < len(out) else GESTURE_STEADY
        if run_len <= max_bridge and (
            prev_g == GESTURE_GLISSANDO or next_g == GESTURE_GLISSANDO
        ):
            out[i:j] = GESTURE_GLISSANDO
        i = j
    return out


def label_gestures_from_f0(
    f0: np.ndarray,
    csv_transition: np.ndarray | None = None,
) -> np.ndarray:
    """Offline gesture labels from f0 (+ optional CSV transition flags).

    Training uses looser glissando detection on pitch ramps. Expert CSV
    transitions are excluded from glissando search, but f0 jump regions can
    still become glissando (scoops between notes) unless marked transition.
    """
    f0 = np.asarray(f0, dtype=np.float64)
    n = len(f0)
    csv_t = np.zeros(n, dtype=bool) if csv_transition is None else np.asarray(
        csv_transition[:n], dtype=bool,
    )
    jump_t = detect_transition_frames(f0)
    vibrato = detect_vibrato_frames(f0.astype(np.float32), **_TRAIN_VIBRATO)
    # Slides often overlap pitch jumps — only CSV transition blocks glissando search.
    glissando = detect_glissando_frames(
        f0, csv_t, vibrato_mask=vibrato, **_TRAIN_GLISS,
    )
    transition = csv_t | (jump_t & ~glissando & ~vibrato)

    labels = np.full(n, GESTURE_STEADY, dtype=np.int8)
    labels[glissando & ~transition & ~vibrato] = GESTURE_GLISSANDO
    labels[vibrato & ~transition] = GESTURE_VIBRATO
    labels[transition] = GESTURE_TRANSITION
    return enforce_min_duration(labels, _MIN_DURATION_TRAIN)


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
    vibrato_prob_min: float = 0.72,
    transition_conf_min: float = 0.70,
    steady_vibrato_override: float = 0.78,
) -> np.ndarray:
    """Blend model logits with f0 heuristics — heuristics win for motion classes.

    v3 val vibrato precision ~27%%; model may only override steady when very
    confident. Glissando/transition from heuristics are preserved unless the
    model strongly predicts transition.
    """
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
        vib_p = float(probs[i, GESTURE_VIBRATO])
        trans_p = float(probs[i, GESTURE_TRANSITION])
        gliss_p = float(probs[i, GESTURE_GLISSANDO])

        if h == GESTURE_TRANSITION:
            out[i] = GESTURE_TRANSITION
        elif h == GESTURE_GLISSANDO:
            if m == GESTURE_TRANSITION and trans_p >= transition_conf_min:
                out[i] = GESTURE_TRANSITION
            else:
                out[i] = GESTURE_GLISSANDO
        elif h == GESTURE_VIBRATO:
            if m == GESTURE_TRANSITION and trans_p >= transition_conf_min:
                out[i] = GESTURE_TRANSITION
            else:
                out[i] = GESTURE_VIBRATO
        elif h == GESTURE_STEADY:
            if m == GESTURE_TRANSITION and trans_p >= transition_conf_min:
                out[i] = GESTURE_TRANSITION
            elif m == GESTURE_GLISSANDO and gliss_p >= 0.52:
                out[i] = GESTURE_GLISSANDO
            else:
                # Never promote steady→vibrato from model alone (v3 precision ~27%).
                # Genuine vibrato comes from heuristics + band-pass overlay.
                out[i] = GESTURE_STEADY
        elif m != GESTURE_STEADY and c >= model_confidence:
            out[i] = m
        elif c >= 0.65:
            out[i] = m
    return out


def reconcile_false_vibrato(
    gesture: np.ndarray,
    f0: np.ndarray,
    vib_cents: np.ndarray,
    *,
    win: int = 25,
    amp_thr: float = 8.0,
    std_thr: float = 3.5,
) -> np.ndarray:
    """Downgrade vibrato labels when the band-pass deviation is too weak."""
    out = gesture.copy()
    n = min(len(out), len(f0), len(vib_cents))
    for i in range(n):
        if out[i] != GESTURE_VIBRATO or f0[i] <= 0:
            continue
        lo = max(0, i - win + 1)
        seg = vib_cents[lo:i + 1]
        valid = ~np.isnan(seg)
        if valid.sum() < 8:
            continue
        seg = seg[valid]
        if np.max(np.abs(seg)) < amp_thr and np.std(seg) < std_thr:
            out[i] = GESTURE_STEADY
    return out


def reconcile_false_glissando(
    gesture: np.ndarray,
    f0: np.ndarray,
    vib_cents: np.ndarray,
    *,
    win: int = 25,
    peak_thr: float = 10.0,
    std_thr: float = 5.0,
) -> np.ndarray:
    """Downgrade glissando when band-pass deviation oscillates (vibrato)."""
    out = gesture.copy()
    n = min(len(out), len(f0), len(vib_cents))
    for i in range(n):
        if out[i] != GESTURE_GLISSANDO or f0[i] <= 0:
            continue
        lo = max(0, i - win + 1)
        seg = vib_cents[lo:i + 1]
        valid = ~np.isnan(seg)
        if valid.sum() < 10:
            continue
        seg = seg[valid]
        if np.max(np.abs(seg)) >= peak_thr and np.std(seg) >= std_thr:
            out[i] = GESTURE_VIBRATO
    return out


def overlay_vibrato_from_deviation(
    gesture: np.ndarray,
    f0: np.ndarray,
    vib_cents: np.ndarray,
    threshold: float = 13.0,
    win: int = 25,
    std_thr: float = 6.0,
) -> np.ndarray:
    """Promote steady/glissando → vibrato when band-pass deviation matches chart."""
    out = gesture.copy()
    n = min(len(out), len(f0), len(vib_cents))
    for i in range(n):
        if f0[i] <= 0 or out[i] not in (GESTURE_STEADY, GESTURE_GLISSANDO):
            continue
        lo = max(0, i - win + 1)
        seg = vib_cents[lo:i + 1]
        valid = ~np.isnan(seg)
        if valid.sum() < 10:
            continue
        seg = seg[valid]
        peak = float(np.max(np.abs(seg)))
        std = float(np.std(seg))
        if peak >= threshold and std >= std_thr:
            out[i] = GESTURE_VIBRATO
    return out


def smooth_gesture_label(
    recent: np.ndarray,
    vib_recent: np.ndarray | None = None,
    vib_threshold: float = 10.0,
) -> int:
    """Stable UI label — vibrato wins over weak glissando; slides need a strong run."""
    if vib_recent is not None and len(vib_recent) > 0:
        vib = np.asarray(vib_recent, dtype=np.float64)
        valid = ~np.isnan(vib)
        if valid.sum() >= 15:
            active = valid & (np.abs(vib) >= vib_threshold)
            if active.sum() >= max(8, int(valid.sum() * 0.18)):
                return GESTURE_VIBRATO
            seg = vib[valid]
            if len(seg) >= 25 and np.std(seg) >= 6.0 and np.max(np.abs(seg)) >= vib_threshold:
                return GESTURE_VIBRATO

    if len(recent) > 0:
        win = len(recent)
        n_vib = int(np.sum(recent == GESTURE_VIBRATO))
        n_gliss = int(np.sum(recent == GESTURE_GLISSANDO))
        n_trans = int(np.sum(recent == GESTURE_TRANSITION))
        if n_vib >= max(6, int(win * 0.12)):
            return GESTURE_VIBRATO
        if n_gliss >= max(8, int(win * 0.14)) and n_gliss >= n_trans:
            return GESTURE_GLISSANDO
        if n_trans >= max(4, int(win * 0.10)):
            return GESTURE_TRANSITION
        if n_gliss >= max(8, int(win * 0.14)):
            return GESTURE_GLISSANDO

    if len(recent) == 0:
        return GESTURE_STEADY
    n_vib = int(np.sum(recent == GESTURE_VIBRATO))
    if n_vib >= max(8, int(len(recent) * 0.15)):
        return GESTURE_VIBRATO
    vals, counts = np.unique(recent, return_counts=True)
    return int(vals[counts.argmax()])
