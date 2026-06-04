"""F0-trajectory features for the standalone GestureTCN model."""

from __future__ import annotations

import math

import numpy as np
import torch

N_GESTURE_FEATURES = 8


def _log_f0_hz(f0_hz: np.ndarray) -> np.ndarray:
    f0 = np.asarray(f0_hz, dtype=np.float64)
    out = np.zeros_like(f0)
    voiced = f0 > 0
    out[voiced] = np.log2(f0[voiced] + 1e-10)
    return out


def build_gesture_features_np(
    f0_hz: np.ndarray,
    vad: np.ndarray | None = None,
    *,
    slope_win: int = 7,
    std_win: int = 11,
    range_win: int = 21,
) -> np.ndarray:
    """Per-frame features (T, N_GESTURE_FEATURES) from an f0 track in Hz."""
    f0 = np.asarray(f0_hz, dtype=np.float32)
    n = len(f0)
    if vad is None:
        voiced = (f0 > 0).astype(np.float32)
    else:
        voiced = (np.asarray(vad, dtype=np.float32) > 0.5).astype(np.float32)
        voiced = np.maximum(voiced, (f0 > 0).astype(np.float32))

    logf = _log_f0_hz(f0).astype(np.float32)
    log_norm = (logf / 10.0) * voiced

    delta = np.zeros(n, dtype=np.float32)
    if n > 1:
        delta[1:] = 1200.0 * (logf[1:] - logf[:-1])
    delta = (np.clip(delta, -200.0, 200.0) / 200.0) * voiced

    delta2 = np.zeros(n, dtype=np.float32)
    if n > 2:
        delta2[2:] = delta[2:] - delta[1:-1]
    delta2 = np.clip(delta2, -1.0, 1.0) * voiced

    abs_slope = np.abs(delta) * voiced

    local_std = np.zeros(n, dtype=np.float32)
    local_range = np.zeros(n, dtype=np.float32)
    half_s = std_win // 2
    half_r = range_win // 2
    for i in range(n):
        if voiced[i] < 0.5:
            continue
        s0, s1 = max(0, i - half_s), min(n, i + half_s + 1)
        seg = logf[s0:s1]
        vseg = voiced[s0:s1] > 0.5
        if vseg.sum() >= 3:
            local_std[i] = min(float(np.std(seg[vseg]) * 4.0), 1.0)
        r0, r1 = max(0, i - half_r), min(n, i + half_r + 1)
        rseg = logf[r0:r1]
        rv = voiced[r0:r1] > 0.5
        if rv.sum() >= 4:
            local_range[i] = min(
                float((np.max(rseg[rv]) - np.min(rseg[rv])) * 1200.0 / 200.0),
                1.0,
            )

    slope_win = max(3, int(slope_win))
    net_slope = np.zeros(n, dtype=np.float32)
    half = slope_win // 2
    for i in range(n):
        if voiced[i] < 0.5:
            continue
        j0, j1 = max(0, i - half), min(n, i + half + 1)
        seg = logf[j0:j1]
        vv = voiced[j0:j1] > 0.5
        if vv.sum() >= 3:
            t = np.arange(j1 - j0, dtype=np.float32)[vv]
            y = seg[vv]
            if len(t) >= 3:
                coeff = np.polyfit(t, y, 1)[0]
                net_slope[i] = np.clip(coeff * 1200.0 / 40.0, -1.0, 1.0)

    return np.stack(
        [log_norm, delta, delta2, voiced, abs_slope, local_std, local_range, net_slope],
        axis=-1,
    ).astype(np.float32)


def build_gesture_features_torch(
    f0_hz: torch.Tensor,
    vad: torch.Tensor | None = None,
) -> torch.Tensor:
    """Batch (B, T) f0 → (B, T, N_GESTURE_FEATURES). Uses numpy per row for now."""
    f0_np = f0_hz.detach().cpu().numpy()
    b, t = f0_np.shape
    out = np.zeros((b, t, N_GESTURE_FEATURES), dtype=np.float32)
    for i in range(b):
        v = None if vad is None else vad[i].detach().cpu().numpy()
        out[i] = build_gesture_features_np(f0_np[i], v)
    return torch.from_numpy(out).to(device=f0_hz.device, dtype=f0_hz.dtype)


class GestureFeatureStream:
    """Rolling f0 buffer for causal GestureTCN inference."""

    def __init__(self, context: int = 64):
        self.context = max(16, int(context))
        self._f0: list[float] = []
        self._vad: list[float] = []

    def reset(self) -> None:
        self._f0.clear()
        self._vad.clear()

    def push(self, f0_hz: float, voiced: float = 1.0) -> np.ndarray:
        self._f0.append(float(f0_hz))
        self._vad.append(float(voiced))
        if len(self._f0) > self.context:
            self._f0.pop(0)
            self._vad.pop(0)
        f0_arr = np.asarray(self._f0, dtype=np.float32)
        vad_arr = np.asarray(self._vad, dtype=np.float32)
        return build_gesture_features_np(f0_arr, vad_arr)
