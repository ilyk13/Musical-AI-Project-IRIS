"""
NanoPitch Model — a lightweight neural network for real-time pitch tracking.

=== What This Model Does ===

Given a short chunk of audio (represented as a mel spectrogram), the model
predicts two things at every 10ms time step:

  1. VAD (Voice Activity Detection): Is someone singing/speaking right now?
     Output: a probability between 0 and 1.

  2. Pitch Posteriorgram: What pitch is being sung?
     Output: 360 probabilities, one for each possible pitch bin.
     The bins cover 6 octaves (B0 through ~B6) at 20-cent resolution.
     A "cent" is 1/100 of a semitone — so 20 cents ≈ 1/5 of a semitone.

=== Architecture (adapted from RNNoise) ===

The model uses GRUs (Gated Recurrent Units), which are a type of recurrent
neural network well-suited for sequential data like audio. GRUs can "remember"
what they heard in previous frames, which helps track pitch continuously.

Signal flow:
    40 mel bands (input)
      │
      ▼
    Conv1d(40 → 64, kernel=3) + tanh    ← extract local patterns
    Conv1d(64 → 96, kernel=3) + tanh    ← combine into features
      │
      ▼
    GRU layer 1 (96 units)              ← track patterns over time
    GRU layer 2 (96 units)              ← deeper temporal modeling
    GRU layer 3 (96 units)              ← even deeper
      │
      ▼
    Concatenate [conv_out, gru1, gru2, gru3] = 384 features
      │
      ├──→ Dense(384 → 1)   + sigmoid  → VAD probability
      └──→ Dense(384 → 360) + sigmoid  → pitch posteriorgram

Total: ~333K parameters — small enough to run on a laptop CPU or in a browser.

=== Why This Design? ===

- Conv layers act as a learned feature extractor (like a smarter mel filterbank)
- GRU layers capture temporal context (pitch is continuous, not independent per frame)
- Multiple GRU layers at different depths capture different time scales
- Concatenating all layers gives the output heads access to both low-level
  and high-level features (a "skip connection" pattern from RNNoise)
- Sigmoid outputs ensure values are in [0, 1], interpretable as probabilities
"""

import math

import torch
from torch import nn


# ═══════════════════════════════════════════════════════════════════════
# Pitch Posteriorgram Constants
#
# We represent pitch as a probability distribution over 360 bins.
# Each bin is 20 cents wide. There are 1200 cents in an octave
# (12 semitones × 100 cents), so 360 bins = 6 octaves.
#
# Bin 0 ≈ B0 (31.7 Hz), just below C1 on a standard piano
# Bin 359 ≈ B6 (~2006 Hz), above typical soprano range
# ═══════════════════════════════════════════════════════════════════════

PITCH_BINS = 360
PITCH_FMIN = 31.7          # Hz — ~B0 (see bin_to_f0 / nanopitch.c)
PITCH_CENTS_PER_BIN = 20   # resolution in cents

N_MELS = 40  # number of mel spectrogram bands

# Maximum layer size supported by the C/WASM inference engine.
# cond_size and gru_size must not exceed this, or the exported model
# will crash in the browser. Matches NC_MAX_LAYER_SIZE in nanopitch.h.
MAX_LAYER_SIZE = 512


# ═══════════════════════════════════════════════════════════════════════
# Pitch Conversion Utilities
# ═══════════════════════════════════════════════════════════════════════

def f0_to_bin(f0_hz):
    """Convert fundamental frequency (Hz) to pitch bin index.

    The formula uses the logarithmic relationship between frequency and
    musical pitch: going up one octave doubles the frequency, and there
    are 1200 cents per octave.

        bin = 1200 * log2(f0 / f_min) / cents_per_bin

    Returns -1 for unvoiced frames (f0 <= 0).
    """
    import numpy as np
    f0_hz = np.asarray(f0_hz, dtype=np.float64)
    result = np.full_like(f0_hz, -1.0)
    voiced = f0_hz > 0
    result[voiced] = 1200.0 * np.log2(f0_hz[voiced] / PITCH_FMIN) / PITCH_CENTS_PER_BIN
    return result


def bin_to_f0(bins):
    """Convert pitch bin index back to Hz. Inverse of f0_to_bin."""
    import numpy as np
    bins = np.asarray(bins, dtype=np.float64)
    return PITCH_FMIN * 2.0 ** (bins * PITCH_CENTS_PER_BIN / 1200.0)


def f0_to_posteriorgram(f0_hz, n_frames=None, sigma_bins=1.2):
    """Create a Gaussian-blurred pitch posteriorgram from f0 values.

    For each voiced frame, we place a Gaussian bump centered at the true
    pitch bin. This is a "soft" label — instead of a hard one-hot vector,
    nearby bins also get some probability mass. This helps the model learn
    because pitch is continuous, not discrete.

    Args:
        f0_hz: (T,) array of f0 in Hz (0 = unvoiced)
        sigma_bins: width of the Gaussian in bins (1.2 ≈ 24 cents)

    Returns:
        (T, 360) float32 array — one probability distribution per frame
    """
    import numpy as np
    if n_frames is None:
        n_frames = len(f0_hz)

    f0_hz = np.asarray(f0_hz[:n_frames], dtype=np.float64)
    bins = f0_to_bin(f0_hz)

    posteriorgram = np.zeros((n_frames, PITCH_BINS), dtype=np.float32)
    bin_indices = np.arange(PITCH_BINS, dtype=np.float64)

    for t in range(n_frames):
        if bins[t] < 0:
            continue  # unvoiced frame — all zeros (no pitch)
        # Gaussian centered at the true pitch bin
        dist = bin_indices - bins[t]
        posteriorgram[t] = np.exp(-0.5 * (dist / sigma_bins) ** 2)

    return posteriorgram


def viterbi_decode(posteriorgram, transition_width=12, voicing_threshold=0.3,
                   onset_penalty=2.0):
    """Decode a pitch posteriorgram into a smooth f0 track using Viterbi.

    The Viterbi algorithm finds the most likely sequence of pitch states
    over time, given:
      - Observation probabilities: the model's pitch posteriorgram
      - Transition constraints: pitch can't jump more than ±12 bins per frame
      - Voicing model: an unvoiced state with onset/offset penalties

    This is a proper dynamic programming algorithm (not just argmax), so
    it produces smoother pitch tracks and handles brief dropouts better.

    State space: 360 voiced bins + 1 unvoiced state = 361 total.
    Transition: voiced states can reach neighbors within ±transition_width.
    Observation: log(posteriorgram) for voiced, log(1 - max_post) for unvoiced.

    Vectorized implementation using numpy strided windows for speed.

    Args:
        posteriorgram: (T, 360) pitch probabilities from the model
        transition_width: max pitch change per frame in bins (12 = 240 cents)
        voicing_threshold: min confidence to initialize as voiced
        onset_penalty: log-domain cost for voiced↔unvoiced transitions

    Returns:
        f0_hz: (T,) float32 array of decoded f0 in Hz (0 = unvoiced)
    """
    import numpy as np

    T, N = posteriorgram.shape
    if T == 0:
        return np.zeros(0, dtype=np.float32)

    tw = int(transition_width)
    W = 2 * tw + 1  # window size for transition neighborhood
    log_obs = np.log(posteriorgram + 1e-10)

    # Viterbi tables: V[t] = best log-probability ending in each state
    # States 0..N-1 = pitched, state N = unvoiced
    V = np.full((T, N + 1), -np.inf, dtype=np.float64)
    bp = np.zeros((T, N + 1), dtype=np.int32)  # backpointers

    # ── Initialize frame 0 ──
    max_post = posteriorgram[0].max()
    if max_post > voicing_threshold:
        V[0, :N] = log_obs[0]
    V[0, N] = np.log(1.0 - max_post + 1e-10)

    # ── Forward pass (vectorized per frame) ──
    for t in range(1, T):
        max_post_t = posteriorgram[t].max()
        prev = V[t - 1, :N]  # (N,) previous voiced scores

        # Find best predecessor within ±tw bins for each state.
        # Pad prev with -inf, then use as_strided to get all windows at once.
        padded = np.pad(prev, (tw, tw), constant_values=-np.inf)
        # windows[s, k] = padded[s + k] = prev[s + k - tw] for k in 0..W-1
        windows = np.lib.stride_tricks.as_strided(
            padded, shape=(N, W),
            strides=(padded.strides[0], padded.strides[0]))
        # Best within each window
        best_k = np.argmax(windows, axis=1)          # (N,) offset within window
        best_val = windows[np.arange(N), best_k]     # (N,) best score
        best_from_voiced = np.clip(np.arange(N) - tw + best_k, 0, N - 1)

        # Option: come from unvoiced state (onset penalty)
        from_unvoiced = V[t - 1, N] - onset_penalty

        # Choose best predecessor for each voiced state
        use_voiced = best_val >= from_unvoiced
        V[t, :N] = np.where(use_voiced, best_val, from_unvoiced) + log_obs[t]
        bp[t, :N] = np.where(use_voiced, best_from_voiced, N)

        # Unvoiced state: from best voiced (offset penalty) or stay unvoiced
        best_voiced_score = prev.max()
        best_voiced_idx = prev.argmax()
        from_voiced = best_voiced_score - onset_penalty
        stay_uv = V[t - 1, N]
        uv_obs = np.log(1.0 - max_post_t + 1e-10)

        if stay_uv >= from_voiced:
            V[t, N] = stay_uv + uv_obs
            bp[t, N] = N
        else:
            V[t, N] = from_voiced + uv_obs
            bp[t, N] = best_voiced_idx

    # ── Backtrace ──
    path = np.zeros(T, dtype=np.int32)
    path[T - 1] = np.argmax(V[T - 1])
    for t in range(T - 2, -1, -1):
        path[t] = bp[t + 1, path[t + 1]]

    # ── Convert to f0 ──
    f0_hz = np.zeros(T, dtype=np.float32)
    voiced_mask = path < N
    if voiced_mask.any():
        f0_hz[voiced_mask] = bin_to_f0(path[voiced_mask].astype(np.float64))

    return f0_hz


def viterbi_stream(posteriorgram, state=None,
                   transition_width=12, voicing_threshold=0.3, onset_penalty=2.0):
    """Streaming Viterbi for continuous real-time inference across chunk boundaries.

    Identical transition model to viterbi_decode_realtime, but accepts and returns
    the prev-score vector so state persists across calls. This is the correct way
    to run Viterbi when audio arrives in chunks rather than as a complete sequence —
    it matches the C/WASM deployment exactly at chunk boundaries.

    Args:
        posteriorgram: (T, 360) pitch probabilities for the current chunk
        state:         (361,) float64 prev-score vector from the previous call,
                       or None to initialize fresh on the first chunk.
        transition_width, voicing_threshold, onset_penalty:
                       same semantics as viterbi_decode_realtime

    Returns:
        f0_hz: (T,) float32 decoded f0 in Hz (0 = unvoiced)
        state: (361,) float64 prev-score vector to pass into the next call
    """
    import numpy as np

    T, N = posteriorgram.shape
    if T == 0:
        s = state if state is not None else np.full(N + 1, -10.0, dtype=np.float64)
        return np.zeros(0, dtype=np.float32), s

    tw = int(transition_width)
    W = 2 * tw + 1

    # Match NanoPitch C exactly: initialise all states to -10.0, not -inf.
    # The C code does: st->viterbi_prev[i] = -10.0f  (for all N+1 states).
    # A voicing_threshold gate on frame 0 traps the tracker in all-unvoiced
    # (-inf) and it can never recover; the flat -10.0 prior lets the
    # observation model decide voicing immediately on every frame.
    prev = state if state is not None else np.full(N + 1, -10.0, dtype=np.float64)
    f0_hz = np.zeros(T, dtype=np.float32)

    for t in range(T):
        max_post_t = posteriorgram[t].max()
        log_obs_t = np.log(posteriorgram[t] + 1e-10)
        uv_obs = np.log(1.0 - max_post_t + 1e-10)

        curr = np.full(N + 1, -np.inf, dtype=np.float64)

        prev_voiced = prev[:N]
        padded = np.pad(prev_voiced, (tw, tw), constant_values=-np.inf)
        windows = np.lib.stride_tricks.as_strided(
            padded, shape=(N, W),
            strides=(padded.strides[0], padded.strides[0]))
        best_val = np.max(windows, axis=1)
        from_uv = prev[N] - onset_penalty
        curr[:N] = np.maximum(best_val, from_uv) + log_obs_t
        from_voiced = prev_voiced.max() - onset_penalty
        curr[N] = max(prev[N], from_voiced) + uv_obs

        best_state = np.argmax(curr)
        if best_state < N:
            f0_hz[t] = bin_to_f0(float(best_state))

        prev = curr

    return f0_hz, prev


def viterbi_decode_realtime(posteriorgram, transition_width=12,
                            voicing_threshold=0.3, onset_penalty=2.0):
    """Realtime (greedy) Viterbi — matches the C/WASM deployment exactly.

    Unlike the offline version, this processes frames left-to-right and
    emits the best state immediately at each frame, without backtracing.
    This is what runs in real-time in the browser.

    The tradeoff: realtime Viterbi can't "change its mind" about earlier
    frames when it sees future evidence. In practice, the difference is
    small for well-trained models, but it can occasionally miss brief
    voiced segments or produce slightly less smooth pitch tracks.

    Same transition model as offline:
      - 360 voiced bins + 1 unvoiced state
      - ±transition_width bin transitions allowed per frame
      - onset/offset penalty for voiced↔unvoiced switches

    Args / Returns: same as viterbi_decode.
    """
    import numpy as np

    T, N = posteriorgram.shape
    if T == 0:
        return np.zeros(0, dtype=np.float32)

    tw = int(transition_width)
    W = 2 * tw + 1

    # Only keep the current column of scores (no backpointer storage)
    prev = np.full(N + 1, -np.inf, dtype=np.float64)
    f0_hz = np.zeros(T, dtype=np.float32)

    for t in range(T):
        max_post_t = posteriorgram[t].max()
        log_obs_t = np.log(posteriorgram[t] + 1e-10)
        uv_obs = np.log(1.0 - max_post_t + 1e-10)

        curr = np.full(N + 1, -np.inf, dtype=np.float64)

        if t == 0:
            # Initialize
            if max_post_t > voicing_threshold:
                curr[:N] = log_obs_t
            curr[N] = uv_obs
        else:
            prev_voiced = prev[:N]

            # Best predecessor within ±tw for each voiced state (vectorized)
            padded = np.pad(prev_voiced, (tw, tw), constant_values=-np.inf)
            windows = np.lib.stride_tricks.as_strided(
                padded, shape=(N, W),
                strides=(padded.strides[0], padded.strides[0]))
            best_val = np.max(windows, axis=1)

            # From unvoiced (onset penalty)
            from_uv = prev[N] - onset_penalty

            # Voiced states: best of (from voiced neighbor, from unvoiced)
            curr[:N] = np.maximum(best_val, from_uv) + log_obs_t

            # Unvoiced state: best of (stay unvoiced, from any voiced)
            from_voiced = prev_voiced.max() - onset_penalty
            curr[N] = max(prev[N], from_voiced) + uv_obs

        # Emit: pick best current state (greedy, no backtrace)
        best_state = np.argmax(curr)
        if best_state < N:
            f0_hz[t] = bin_to_f0(float(best_state))

        prev = curr

    return f0_hz


# ═══════════════════════════════════════════════════════════════════════
# The Neural Network
# ═══════════════════════════════════════════════════════════════════════

class NanoPitch(nn.Module):
    """Lightweight GRU network for real-time pitch tracking and VAD.

    Args:
        n_mels: number of mel spectrogram input bands (40)
        cond_size: width of the first conv layer (64)
        gru_size: number of units in each GRU layer (96)
    """

    def __init__(self, n_mels=N_MELS, cond_size=64, gru_size=96):
        super().__init__()
        if cond_size > MAX_LAYER_SIZE or gru_size > MAX_LAYER_SIZE:
            raise ValueError(
                f"cond_size={cond_size} or gru_size={gru_size} exceeds "
                f"MAX_LAYER_SIZE={MAX_LAYER_SIZE}. The C/WASM engine "
                f"cannot run models larger than this. Increase "
                f"NC_MAX_LAYER_SIZE in nanopitch.h if you really need it.")
        self.n_mels = n_mels
        self.cond_size = cond_size
        self.gru_size = gru_size

        # ── Causal convolutional feature extractor ──
        # Conv1d slides a small kernel (size 3) across the time axis.
        # "Causal" means we only look at the current and past frames, never
        # the future — essential for real-time streaming. We achieve this by
        # padding 2 frames on the LEFT only, so:
        #   output[t] = f(input[t-2], input[t-1], input[t])
        # Two stacked causal conv layers look at 5 frames of past context
        # total (50ms at our 10ms hop rate), with zero latency.
        self.conv1 = nn.Conv1d(n_mels, cond_size, kernel_size=3, padding=0)
        self.conv2 = nn.Conv1d(cond_size, gru_size, kernel_size=3, padding=0)

        # ── Recurrent layers ──
        self.gru1 = nn.GRU(gru_size, gru_size, batch_first=True)
        self.gru2 = nn.GRU(gru_size, gru_size, batch_first=True)
        self.gru3 = nn.GRU(gru_size, gru_size, batch_first=True)

        # ── Output heads ──
        cat_size = gru_size * 4
        self.dense_vad = nn.Linear(cat_size, 1)
        self.dense_pitch = nn.Linear(cat_size, PITCH_BINS)

        self._init_weights()
        n_params = sum(p.numel() for p in self.parameters())
        print(f"NanoPitch: {n_params:,} parameters "
              f"(cond={cond_size}, gru={gru_size})")

    def _init_weights(self):
        for name, module in self.named_modules():
            if isinstance(module, nn.GRU):
                for pname, p in module.named_parameters():
                    if 'weight_hh' in pname:
                        nn.init.orthogonal_(p)

    def forward(self, mel, states=None, return_logits=False):
        """Run the model on a batch of mel spectrograms.

        Args:
            mel: (batch, time, 40) — log-mel spectrogram input
            states: optional GRU hidden states (for continuing a stream)
            return_logits: if True, return raw pre-sigmoid scores

        Returns:
            vad:    (batch, time, 1)   — voice activity probability
            pitch:  (batch, time, 360) — pitch posteriorgram
            states: list of 3 GRU hidden states
        """
        B = mel.size(0)
        device = mel.device

        if states is None:
            h1 = torch.zeros(1, B, self.gru_size, device=device)
            h2 = torch.zeros(1, B, self.gru_size, device=device)
            h3 = torch.zeros(1, B, self.gru_size, device=device)
        else:
            h1, h2, h3 = states

        x = mel.permute(0, 2, 1)                         # (B, 40, T)
        x = torch.nn.functional.pad(x, (2, 0))
        x = torch.tanh(self.conv1(x))                     # (B, 64, T)
        x = torch.nn.functional.pad(x, (2, 0))
        x = torch.tanh(self.conv2(x))                     # (B, 96, T)
        x = x.permute(0, 2, 1)                            # (B, T, 96)

        g1, h1 = self.gru1(x, h1)
        g2, h2 = self.gru2(g1, h2)
        g3, h3 = self.gru3(g2, h3)

        cat = torch.cat([x, g1, g2, g3], dim=-1)          # (B, T, 384)

        vad_logits = self.dense_vad(cat)
        pitch_logits = self.dense_pitch(cat)
        if return_logits:
            return vad_logits, pitch_logits, [h1, h2, h3]
        vad = torch.sigmoid(vad_logits)
        pitch = torch.sigmoid(pitch_logits)

        return vad, pitch, [h1, h2, h3]

    def forward_single_frame(self, mel_frame, states, return_logits=False):
        """Process one frame at a time (for real-time/streaming use)."""
        conv1_buf = states['conv1_buf']
        conv2_buf = states['conv2_buf']
        gru_states = states['gru_states']

        conv1_in = torch.cat([conv1_buf, mel_frame], dim=1)
        x = torch.tanh(self.conv1(conv1_in.permute(0, 2, 1)))
        conv1_buf = conv1_in[:, 1:, :]

        x_t = x.permute(0, 2, 1)
        conv2_in = torch.cat([conv2_buf, x_t], dim=1)
        x = torch.tanh(self.conv2(conv2_in.permute(0, 2, 1)))
        conv2_buf = conv2_in[:, 1:, :]
        x = x.permute(0, 2, 1)

        h1, h2, h3 = gru_states
        g1, h1 = self.gru1(x, h1)
        g2, h2 = self.gru2(g1, h2)
        g3, h3 = self.gru3(g2, h3)

        cat = torch.cat([x, g1, g2, g3], dim=-1)
        vad_logits = self.dense_vad(cat)
        pitch_logits = self.dense_pitch(cat)
        if return_logits:
            vad, pitch = vad_logits, pitch_logits
        else:
            vad = torch.sigmoid(vad_logits)
            pitch = torch.sigmoid(pitch_logits)

        new_states = {
            'conv1_buf': conv1_buf,
            'conv2_buf': conv2_buf,
            'gru_states': [h1, h2, h3],
        }
        return vad, pitch, new_states

    def init_streaming_state(self, device='cpu'):
        """Create initial state for streaming inference (all zeros)."""
        return {
            'conv1_buf': torch.zeros(1, 2, self.n_mels, device=device),
            'conv2_buf': torch.zeros(1, 2, self.cond_size, device=device),
            'gru_states': [
                torch.zeros(1, 1, self.gru_size, device=device),
                torch.zeros(1, 1, self.gru_size, device=device),
                torch.zeros(1, 1, self.gru_size, device=device),
            ],
        }


# ═══════════════════════════════════════════════════════════════════════
# NanoPitch+ — multi-task extension for VocalSet training
# ═══════════════════════════════════════════════════════════════════════

NUM_GESTURES = 4    # steady, vibrato, glissando, transition
NUM_REGISTERS = 4   # chest, mixed, head, falsetto
NUM_DYNAMICS = 6    # pp … ff

# Gesture indices — must match data/vocalset_labels.py
GESTURE_STEADY = 0
GESTURE_VIBRATO = 1
GESTURE_GLISSANDO = 2
GESTURE_TRANSITION = 3

# Per-gesture Viterbi transition width (bins). Wider widths allow expected motion.
GESTURE_TRANSITION_WIDTH = {
    GESTURE_STEADY: 12,
    GESTURE_VIBRATO: 30,
    GESTURE_GLISSANDO: 24,
    GESTURE_TRANSITION: 36,
}

GESTURE_F0_FEAT_DIM = 4  # log-f0, frame delta (¢), pitch entropy, VAD prob


def build_gesture_f0_features(
    vad_logits: torch.Tensor,
    pitch_logits: torch.Tensor,
    *,
    prev_log_f0: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Trajectory features for the gesture head (matches train + streaming).

    Built from pitch/VAD logits so training matches live inference. Returns
    (B, T, GESTURE_F0_FEAT_DIM) and per-batch previous log-f0 for streaming.
    """
    vad = torch.sigmoid(vad_logits.squeeze(-1))
    pitch = torch.sigmoid(pitch_logits)
    bins = pitch.argmax(dim=-1).float()
    log_f0 = torch.log2(
        PITCH_FMIN * (2.0 ** (bins * PITCH_CENTS_PER_BIN / 1200.0)) + 1e-10
    )
    voiced = vad > 0.3
    log_f0 = log_f0 * voiced

    log_norm = (log_f0 / 10.0).unsqueeze(-1)
    B, T = log_f0.shape
    device, dtype = log_f0.device, log_f0.dtype

    if prev_log_f0 is None:
        delta = torch.zeros(B, T, device=device, dtype=dtype)
        if T > 1:
            delta[:, 1:] = 1200.0 * (log_f0[:, 1:] - log_f0[:, :-1])
        last_prev = log_f0[:, -1].detach()
    else:
        delta0 = 1200.0 * (log_f0[:, 0] - prev_log_f0.squeeze(-1))
        delta = torch.zeros(B, T, device=device, dtype=dtype)
        delta[:, 0] = delta0
        if T > 1:
            delta[:, 1:] = 1200.0 * (log_f0[:, 1:] - log_f0[:, :-1])
        last_prev = log_f0[:, -1].detach()
    delta = (delta.clamp(-200.0, 200.0) / 200.0 * voiced).unsqueeze(-1)

    p = pitch + 1e-8
    ent = (-(p * torch.log(p)).sum(dim=-1) / math.log(PITCH_BINS)).unsqueeze(-1)
    vad_f = vad.unsqueeze(-1)
    feats = torch.cat([log_norm, delta, ent, vad_f], dim=-1)
    return feats, last_prev


class GestureHead(nn.Module):
    """IRIS-only gesture classifier — MLP on backbone + pitch trajectory."""

    def __init__(
        self,
        cat_size: int,
        n_classes: int = NUM_GESTURES,
        hidden: int = 128,
        f0_feat_dim: int = GESTURE_F0_FEAT_DIM,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.f0_feat_dim = f0_feat_dim
        in_dim = cat_size + f0_feat_dim
        h2 = max(hidden // 2, 32)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, h2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(h2, n_classes),
        )
        n_params = sum(p.numel() for p in self.parameters())
        print(f"  GestureHead: {n_params:,} parameters "
              f"(in={in_dim}, hidden={hidden})")

    def forward(self, cat: torch.Tensor, f0_feat: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([cat, f0_feat], dim=-1))


class NanoPitchPlus(NanoPitch):
    """NanoPitch with gesture, register, and dynamics heads on the 384-d concat."""

    def __init__(
        self,
        n_mels=N_MELS,
        cond_size=64,
        gru_size=96,
        gesture_hidden: int = 128,
        use_f0_gesture_feats: bool = True,
    ):
        super().__init__(n_mels=n_mels, cond_size=cond_size, gru_size=gru_size)
        cat_size = gru_size * 4
        self.use_f0_gesture_feats = use_f0_gesture_feats
        f0_dim = GESTURE_F0_FEAT_DIM if use_f0_gesture_feats else 0
        self.gesture_head = GestureHead(
            cat_size, hidden=gesture_hidden, f0_feat_dim=f0_dim,
        )
        self.dense_register = nn.Linear(cat_size, NUM_REGISTERS)
        self.dense_dynamics = nn.Linear(cat_size, NUM_DYNAMICS)
        n_params = sum(p.numel() for p in self.parameters())
        print(f"NanoPitchPlus: {n_params:,} parameters "
              f"(cond={cond_size}, gru={gru_size}, gesture_hidden={gesture_hidden})")

    def init_streaming_state(self, device='cpu'):
        state = super().init_streaming_state(device)
        state['gesture_prev_log_f0'] = torch.zeros(1, device=device)
        return state

    def _encode(self, mel, states=None):
        """Shared mel → concat features (+ GRU states)."""
        B = mel.size(0)
        device = mel.device
        if states is None:
            h1 = torch.zeros(1, B, self.gru_size, device=device)
            h2 = torch.zeros(1, B, self.gru_size, device=device)
            h3 = torch.zeros(1, B, self.gru_size, device=device)
        else:
            h1, h2, h3 = states

        x = mel.permute(0, 2, 1)
        x = torch.nn.functional.pad(x, (2, 0))
        x = torch.tanh(self.conv1(x))
        x = torch.nn.functional.pad(x, (2, 0))
        x = torch.tanh(self.conv2(x))
        x = x.permute(0, 2, 1)

        g1, h1 = self.gru1(x, h1)
        g2, h2 = self.gru2(g1, h2)
        g3, h3 = self.gru3(g2, h3)
        cat = torch.cat([x, g1, g2, g3], dim=-1)
        return cat, [h1, h2, h3]

    def _heads(
        self,
        cat,
        vad_logits,
        pitch_logits,
        *,
        return_logits=False,
        gesture_prev_log_f0=None,
    ):
        f0_feat = torch.zeros(
            *cat.shape[:-1], 0, device=cat.device, dtype=cat.dtype,
        )
        new_prev = gesture_prev_log_f0
        if self.use_f0_gesture_feats:
            f0_feat, new_prev = build_gesture_f0_features(
                vad_logits, pitch_logits, prev_log_f0=gesture_prev_log_f0,
            )
        gesture_logits = self.gesture_head(cat, f0_feat)
        register_logits = self.dense_register(cat)
        dynamics_logits = self.dense_dynamics(cat)
        if return_logits:
            return (
                vad_logits, pitch_logits, gesture_logits,
                register_logits, dynamics_logits, new_prev,
            )
        return (
            torch.sigmoid(vad_logits),
            torch.sigmoid(pitch_logits),
            gesture_logits,
            register_logits,
            dynamics_logits,
            new_prev,
        )

    def forward(self, mel, states=None, return_logits=False):
        cat, gru_states = self._encode(mel, states)
        vad_logits = self.dense_vad(cat)
        pitch_logits = self.dense_pitch(cat)
        out = self._heads(
            cat, vad_logits, pitch_logits, return_logits=return_logits,
        )
        return (*out[:-1], gru_states)

    def forward_single_frame(self, mel_frame, states, return_logits=False):
        conv1_buf = states['conv1_buf']
        conv2_buf = states['conv2_buf']
        gru_states = states['gru_states']

        conv1_in = torch.cat([conv1_buf, mel_frame], dim=1)
        x = torch.tanh(self.conv1(conv1_in.permute(0, 2, 1)))
        conv1_buf = conv1_in[:, 1:, :]

        x_t = x.permute(0, 2, 1)
        conv2_in = torch.cat([conv2_buf, x_t], dim=1)
        x = torch.tanh(self.conv2(conv2_in.permute(0, 2, 1)))
        conv2_buf = conv2_in[:, 1:, :]
        x = x.permute(0, 2, 1)

        h1, h2, h3 = gru_states
        g1, h1 = self.gru1(x, h1)
        g2, h2 = self.gru2(g1, h2)
        g3, h3 = self.gru3(g2, h3)
        cat = torch.cat([x, g1, g2, g3], dim=-1)

        vad_logits = self.dense_vad(cat)
        pitch_logits = self.dense_pitch(cat)
        prev = states.get('gesture_prev_log_f0')
        out = self._heads(
            cat, vad_logits, pitch_logits,
            return_logits=return_logits,
            gesture_prev_log_f0=prev,
        )
        new_states = {
            'conv1_buf': conv1_buf,
            'conv2_buf': conv2_buf,
            'gru_states': [h1, h2, h3],
            'gesture_prev_log_f0': out[-1],
        }
        return (*out[:-1], new_states)

    @classmethod
    def from_nanopitch_checkpoint(cls, ckpt_path, device='cpu'):
        """Initialise Plus heads randomly but load shared NanoPitch weights."""
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        kwargs = ckpt.get("model_kwargs", {}) if isinstance(ckpt, dict) else {}
        model = cls(
            cond_size=kwargs.get("cond_size", 64),
            gru_size=kwargs.get("gru_size", 96),
            gesture_hidden=kwargs.get("gesture_hidden", 128),
            use_f0_gesture_feats=kwargs.get("use_f0_gesture_feats", True),
        )
        state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            print(f"  NanoPitchPlus: loaded base weights, new heads: {missing}")
        if unexpected:
            print(f"  NanoPitchPlus: ignored keys: {unexpected}")
        return model


def viterbi_decode_gesture(
    posteriorgram,
    gesture_classes,
    transition_width=12,
    voicing_threshold=0.3,
    onset_penalty=2.0,
):
    """Offline Viterbi with per-frame transition width from gesture class.

    Vibrato frames use a wider transition neighbourhood; glissando and
    transition frames allow larger pitch motion before scoring accuracy.
    """
    import numpy as np

    T, N = posteriorgram.shape
    if T == 0:
        return np.zeros(0, dtype=np.float32)

    gesture_classes = np.asarray(gesture_classes[:T], dtype=np.int32)
    widths = np.array([
        GESTURE_TRANSITION_WIDTH.get(int(g), transition_width)
        for g in gesture_classes
    ], dtype=np.int32)

    tw_max = int(widths.max())
    log_obs = np.log(posteriorgram + 1e-10)
    V = np.full((T, N + 1), -np.inf, dtype=np.float64)
    bp = np.zeros((T, N + 1), dtype=np.int32)

    max_post = posteriorgram[0].max()
    if max_post > voicing_threshold:
        V[0, :N] = log_obs[0]
    V[0, N] = np.log(1.0 - max_post + 1e-10)

    for t in range(1, T):
        tw = int(widths[t])
        W = 2 * tw + 1
        max_post_t = posteriorgram[t].max()
        prev = V[t - 1, :N]

        padded = np.pad(prev, (tw, tw), constant_values=-np.inf)
        windows = np.lib.stride_tricks.as_strided(
            padded, shape=(N, W),
            strides=(padded.strides[0], padded.strides[0]))
        best_k = np.argmax(windows, axis=1)
        best_val = windows[np.arange(N), best_k]
        best_from_voiced = np.clip(np.arange(N) - tw + best_k, 0, N - 1)

        from_unvoiced = V[t - 1, N] - onset_penalty
        use_voiced = best_val >= from_unvoiced
        V[t, :N] = np.where(use_voiced, best_val, from_unvoiced) + log_obs[t]
        bp[t, :N] = np.where(use_voiced, best_from_voiced, N)

        best_voiced_score = prev.max()
        best_voiced_idx = prev.argmax()
        from_voiced = best_voiced_score - onset_penalty
        stay_uv = V[t - 1, N]
        uv_obs = np.log(1.0 - max_post_t + 1e-10)
        if stay_uv >= from_voiced:
            V[t, N] = stay_uv + uv_obs
            bp[t, N] = N
        else:
            V[t, N] = from_voiced + uv_obs
            bp[t, N] = best_voiced_idx

    path = np.zeros(T, dtype=np.int32)
    path[T - 1] = np.argmax(V[T - 1])
    for t in range(T - 2, -1, -1):
        path[t] = bp[t + 1, path[t + 1]]

    f0_hz = np.zeros(T, dtype=np.float32)
    voiced_mask = path < N
    if voiced_mask.any():
        f0_hz[voiced_mask] = bin_to_f0(path[voiced_mask].astype(np.float64))
    return f0_hz


def viterbi_stream_gesture(
    posteriorgram,
    gesture_classes,
    state=None,
    transition_width=12,
    onset_penalty=2.0,
):
    """Streaming greedy Viterbi with gesture-dependent transition width."""
    import numpy as np

    T, N = posteriorgram.shape
    if T == 0:
        s = state if state is not None else np.full(N + 1, -10.0, dtype=np.float64)
        return np.zeros(0, dtype=np.float32), s

    gesture_classes = np.asarray(gesture_classes[:T], dtype=np.int32)
    prev = state if state is not None else np.full(N + 1, -10.0, dtype=np.float64)
    f0_hz = np.zeros(T, dtype=np.float32)

    for t in range(T):
        tw = int(GESTURE_TRANSITION_WIDTH.get(int(gesture_classes[t]), transition_width))
        W = 2 * tw + 1
        max_post_t = posteriorgram[t].max()
        log_obs_t = np.log(posteriorgram[t] + 1e-10)
        uv_obs = np.log(1.0 - max_post_t + 1e-10)
        curr = np.full(N + 1, -np.inf, dtype=np.float64)

        prev_voiced = prev[:N]
        padded = np.pad(prev_voiced, (tw, tw), constant_values=-np.inf)
        windows = np.lib.stride_tricks.as_strided(
            padded, shape=(N, W),
            strides=(padded.strides[0], padded.strides[0]))
        best_val = np.max(windows, axis=1)
        from_uv = prev[N] - onset_penalty
        curr[:N] = np.maximum(best_val, from_uv) + log_obs_t
        from_voiced = prev_voiced.max() - onset_penalty
        curr[N] = max(prev[N], from_voiced) + uv_obs

        best_state = np.argmax(curr)
        if best_state < N:
            f0_hz[t] = bin_to_f0(float(best_state))
        prev = curr

    return f0_hz, prev


if __name__ == "__main__":
    model = NanoPitch()

    x = torch.randn(2, 100, N_MELS)
    vad, pitch, states = model(x)
    print(f"Input:  {x.shape}")
    print(f"VAD:    {vad.shape}   range [{vad.min():.3f}, {vad.max():.3f}]")
    print(f"Pitch:  {pitch.shape} range [{pitch.min():.3f}, {pitch.max():.3f}]")

    state = model.init_streaming_state()
    for t in range(10):
        frame = torch.randn(1, 1, N_MELS)
        v, p, state = model.forward_single_frame(frame, state)
    print(f"Streaming OK: vad={v.shape}, pitch={p.shape}")
