"""3-class / 4-class gesture label mapping for GestureTCN training and inference."""

from __future__ import annotations

import numpy as np

GESTURE_VOCAB_3 = ["steady", "vibrato", "transition"]
NUM_GESTURES_3 = 3
NUM_GESTURES_4 = 4

# VocalSet npz uses 4-class indices
_GLISS = 2
_TRANSITION_4 = 3


def remap_gesture_labels_4_to_n(gesture: np.ndarray, n_classes: int) -> np.ndarray:
    """Collapse glissando → steady; for 3-class, transition index 3 → 2."""
    g = np.asarray(gesture, dtype=np.int64)
    if n_classes >= 4:
        return g.clip(0, 3)
    out = g.copy()
    out[g == _GLISS] = 0
    out[g == _TRANSITION_4] = 2
    return out.clip(0, 2)


def expand_logits_to_4class(logits: np.ndarray) -> np.ndarray:
    """Map (T, 3) steady/vibrato/transition logits to (T, 4); gliss logit suppressed."""
    logits = np.asarray(logits, dtype=np.float32)
    if logits.ndim == 1:
        logits = logits.reshape(1, -1)
    if logits.shape[-1] == 4:
        return logits
    if logits.shape[-1] != 3:
        raise ValueError(f"expected 3 or 4 gesture logits, got shape {logits.shape}")
    out = np.full((*logits.shape[:-1], 4), -1e4, dtype=np.float32)
    out[..., 0] = logits[..., 0]
    out[..., 1] = logits[..., 1]
    out[..., 3] = logits[..., 2]
    return out


def compute_tcn_class_weights(
    gesture: np.ndarray,
    n_classes: int,
    *,
    max_ratio: float = 2.5,
    transition_cap: float = 5.0,
) -> np.ndarray:
    """Inverse-frequency class weights on remapped gesture labels."""
    g = remap_gesture_labels_4_to_n(gesture, n_classes)
    counts = np.bincount(g.clip(0, n_classes - 1), minlength=n_classes).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    weights = g.size / (n_classes * counts)
    weights /= weights.min()
    caps: dict[int, float] = {}
    if n_classes == 3:
        caps[2] = transition_cap
    else:
        caps[3] = transition_cap
    for i in range(n_classes):
        cap = caps.get(i, max_ratio)
        weights[i] = min(weights[i], max(1.0, float(cap)))
    weights /= weights.sum()
    weights *= n_classes
    return weights.astype(np.float32)


def gesture_vocab(n_classes: int) -> list[str]:
    if n_classes == 3:
        return list(GESTURE_VOCAB_3)
    from data.vocalset_labels import GESTURE_VOCAB
    return list(GESTURE_VOCAB)


def metrics_from_cm(cm: np.ndarray, n_classes: int) -> dict:
    """Per-class recall/precision plus coaching-oriented aggregates."""
    names = gesture_vocab(n_classes)
    recalls, precisions = [], []
    per_recall, per_precision = {}, {}
    for c in range(n_classes):
        true_n = int(cm[c, :].sum())
        pred_n = int(cm[:, c].sum())
        if true_n == 0:
            per_recall[names[c]] = None
        else:
            rec = float(cm[c, c] / true_n)
            recalls.append(rec)
            per_recall[names[c]] = rec
        if pred_n == 0:
            per_precision[names[c]] = None
        else:
            prec = float(cm[c, c] / pred_n)
            precisions.append(prec)
            per_precision[names[c]] = prec

    trans_name = "transition"
    trans_rec = per_recall.get(trans_name) or 0.0
    vib_prec = per_precision.get("vibrato")
    vib_rec = per_recall.get("vibrato")
    return {
        "steady_recall": per_recall.get("steady") or 0.0,
        "vibrato_precision": vib_prec if vib_prec is not None else 0.0,
        "vibrato_recall": vib_rec if vib_rec is not None else 0.0,
        "transition_recall": trans_rec if trans_rec is not None else 0.0,
        "balanced_acc": float(np.mean(recalls)) if recalls else 0.0,
        "per_class_recall": per_recall,
        "per_class_precision": per_precision,
    }
