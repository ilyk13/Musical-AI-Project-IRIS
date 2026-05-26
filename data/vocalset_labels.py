"""Label vocabularies and mappings for VocalSet + Annotated-VocalSet training.

Annotated-VocalSet per-frame CSV columns (raw/ folders):
  Time, F0, Amplitude, onset, offset, Transition

Gesture classes match the AI Vocal Coach plan:
  steady, vibrato, glissando, transition

Register and dynamics are derived from VocalSet technique metadata and
Annotated-VocalSet amplitude contours respectively.
"""

from __future__ import annotations

import re
from typing import Iterable

import numpy as np

# ── Gesture ────────────────────────────────────────────────────────────
GESTURE_VOCAB = ["steady", "vibrato", "glissando", "transition"]
GESTURE_STEADY = 0
GESTURE_VIBRATO = 1
GESTURE_GLISSANDO = 2
GESTURE_TRANSITION = 3

# ── Register ───────────────────────────────────────────────────────────
REGISTER_VOCAB = ["chest", "mixed", "head", "falsetto"]
REGISTER_UNKNOWN = -1

# VocalSet 1.2 technique folder names → register weak labels (file-level).
TECHNIQUE_TO_REGISTER: dict[str, int] = {
    "belt": 0,
    "forte": 0,
    "fast_forte": 0,
    "loud": 0,
    "mixed": 1,
    "messa_di_voce": 1,
    "messa": 1,
    "vibrato": 1,
    "soft": 2,
    "slow_piano": 2,
    "slow_forte": 2,
    "pp": 2,
    "fast_piano": 2,
    "breathy": 2,
    "straight": 2,
    "buzzy": 2,
    "trill": 2,
    "trillo": 2,
    "trill_harmonic": 3,
    "lip_trill": 3,
    "vocal_fry": 0,
    "messy": REGISTER_UNKNOWN,
    "spoken": REGISTER_UNKNOWN,
}

# Good starter set for gesture + register training (~1–2 GB audio, not full 8 GB).
MINIMAL_TRAINING_TECHNIQUES = (
    "straight",     # steady / neutral phonation
    "vibrato",      # vibrato gesture
    "belt",         # chest register
    "slow_piano",   # soft / head register (no "soft" folder in VocalSet 1.2)
    "messa",        # mixed register (messa di voce)
)

# Techniques where vibrato is expected across much of the clip.
VIBRATO_TECHNIQUES = {"vibrato", "trill", "trillo", "trill_harmonic", "messa_di_voce"}

# ── Dynamics (reuse app vocabulary) ────────────────────────────────────
DYNAMIC_VOCAB = ["pp", "p", "mp", "mf", "f", "ff"]
DYNAMIC_SILENCE = -1

# Annotated-VocalSet amplitude is normalised 0–1 within each clip.
AMPLITUDE_TO_DYNAMIC = (0.08, 0.18, 0.30, 0.45, 0.62, 0.78)


def _norm_technique(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_")


def technique_from_path(path: str) -> str | None:
    """Extract VocalSet technique from a path segment (e.g. .../vibrato/foo.wav)."""
    parts = path.replace("\\", "/").split("/")
    for part in reversed(parts):
        key = _norm_technique(part)
        if key in TECHNIQUE_TO_REGISTER or key in VIBRATO_TECHNIQUES:
            return key
    return None


def register_from_technique(technique: str | None) -> int:
    if not technique:
        return REGISTER_UNKNOWN
    return TECHNIQUE_TO_REGISTER.get(_norm_technique(technique), REGISTER_UNKNOWN)


def amplitude_to_dynamic(amp: float) -> int:
    """Map Annotated-VocalSet amplitude (0–1) to dynamic class index."""
    if amp < AMPLITUDE_TO_DYNAMIC[0]:
        return DYNAMIC_SILENCE
    label = 0
    for idx, thr in enumerate(AMPLITUDE_TO_DYNAMIC):
        if amp >= thr:
            label = idx
    return label


def _is_true(val) -> bool:
    if val is None:
        return False
    s = str(val).strip().lower()
    return s in {"true", "1", "yes", "t"}


def detect_vibrato_frames(f0: np.ndarray, min_voiced: int = 30) -> np.ndarray:
    """Mark frames with 4.5–8 Hz f0 modulation (same band as live vibrato analysis)."""
    from scipy.signal import butter, filtfilt

    mask = np.zeros(len(f0), dtype=bool)
    voiced = f0 > 0
    if voiced.sum() < min_voiced:
        return mask

    log_f0 = np.zeros_like(f0, dtype=np.float64)
    log_f0[voiced] = np.log2(f0[voiced])
    log_f0[~voiced] = np.interp(
        np.flatnonzero(~voiced),
        np.flatnonzero(voiced),
        log_f0[voiced],
    )

    sr_frame = 100.0  # 10 ms hop
    b, a = butter(2, [4.5 / (sr_frame / 2), 8.0 / (sr_frame / 2)], btype="band")
    try:
        mod = filtfilt(b, a, log_f0)
    except ValueError:
        return mask

    # Normalise modulation energy per clip.
    energy = np.abs(mod)
    thr = np.percentile(energy[voiced], 75) if voiced.any() else 0.0
    if thr <= 0:
        return mask
    mask = (energy > thr) & voiced
    return mask


def detect_glissando_frames(
    f0: np.ndarray,
    transition_mask: np.ndarray,
    min_slope_cents: float = 8.0,
) -> np.ndarray:
    """Mark transition frames with sustained monotonic f0 motion."""
    mask = np.zeros(len(f0), dtype=bool)
    if len(f0) < 3:
        return mask

    log_f0 = np.zeros_like(f0, dtype=np.float64)
    voiced = f0 > 0
    log_f0[voiced] = np.log2(f0[voiced] + 1e-10)
    slope = np.diff(log_f0, prepend=log_f0[0]) * 1200.0  # cents / frame

    for t in range(1, len(f0) - 1):
        if not transition_mask[t] or not voiced[t]:
            continue
        seg = slope[max(0, t - 2): min(len(f0), t + 3)]
        if seg.size < 2:
            continue
        same_sign = np.all(seg > 0) or np.all(seg < 0)
        if same_sign and np.mean(np.abs(seg)) >= min_slope_cents:
            mask[t] = True
    return mask


def label_gestures(
    f0: np.ndarray,
    transition_col: Iterable,
    technique: str | None = None,
) -> np.ndarray:
    """Build per-frame gesture labels from Annotated-VocalSet columns."""
    transition_col = list(transition_col)
    n = len(f0)
    labels = np.full(n, GESTURE_STEADY, dtype=np.int8)

    transition_mask = np.array([_is_true(v) for v in transition_col[:n]], dtype=bool)
    labels[transition_mask] = GESTURE_TRANSITION

    vibrato_mask = detect_vibrato_frames(f0)
    if technique and _norm_technique(technique) in VIBRATO_TECHNIQUES:
        voiced = f0 > 0
        vibrato_mask |= voiced & ~transition_mask

    gliss_mask = detect_glissando_frames(f0, transition_mask)

    # Priority: transition > glissando > vibrato > steady
    labels[gliss_mask & ~transition_mask] = GESTURE_GLISSANDO
    labels[vibrato_mask & ~transition_mask & ~gliss_mask] = GESTURE_VIBRATO
    return labels


def detect_breath_frames(
    audio: np.ndarray,
    f0: np.ndarray,
    hop: int,
    *,
    rms_threshold: float = 0.015,
    min_gap_frames: int = 5,
) -> np.ndarray:
    """Heuristic breath labels: low-energy unvoiced gaps between phrases."""
    n_frames = int(np.ceil(len(audio) / hop))
    breath = np.zeros(n_frames, dtype=np.float32)

    frame_rms = np.array([
        np.sqrt(np.mean(audio[i * hop: (i + 1) * hop] ** 2) + 1e-12)
        for i in range(n_frames)
    ])

    unvoiced = f0[:n_frames] <= 0 if len(f0) >= n_frames else np.pad(f0 <= 0, (0, n_frames - len(f0)))
    low_energy = frame_rms < rms_threshold

    candidate = unvoiced & low_energy
    start = None
    for t, flag in enumerate(candidate):
        if flag and start is None:
            start = t
        elif not flag and start is not None:
            if t - start >= min_gap_frames:
                breath[start:t] = 1.0
            start = None
    if start is not None and n_frames - start >= min_gap_frames:
        breath[start:n_frames] = 1.0

    return breath
