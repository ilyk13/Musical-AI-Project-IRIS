"""IRIS — AI Vocal Coach  ·  Real-time web server

Audio is captured in the BROWSER via the Web Audio API and streamed
as raw Float32 PCM over WebSocket.

Pitch: trained NanoPitch GRU model + gesture-aware scoring.
       Steady frames are scored for ET accuracy; glissando/transition frames get
       a gentle slide-control penalty instead of ET scoring; vibrato is excluded.

Run:  python3 app.py
Open: http://localhost:8000
"""

import asyncio
import contextlib
import json
import sys
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import librosa
import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from scipy.signal import savgol_filter

sys.path.insert(0, str(Path(__file__).parent))

from features.pitch.gesture import (
    GESTURE_GLISSANDO,
    GESTURE_STEADY,
    GESTURE_VIBRATO,
    VIB_PERIOD_LIVE,
    classify_gestures_live,
    gesture_name,
    gesture_index,
    is_scored_gesture,
    merge_gesture_predictions,
    demote_glissando_for_coaching,
    overlay_vibrato_from_deviation,
    provisional_f0_from_posteriors,
    coalesce_glissando_labels,
    reconcile_false_glissando,
    reconcile_false_vibrato,
    segment_has_vibrato_period,
    smooth_gesture_label,
)
from features.pitch.nanopitch import (
    NanoPitchExtractor, _mel_frame_from_segment,
    HOP_LENGTH as NP_HOP, WIN_LENGTH as NP_WIN, NC_CONV_CONTEXT, MEL_MIN_TAIL,
)
from features.pitch.accuracy import combined_pitch_score
from features.pitch.central_pitch import compute_central_pitch
from features.breath.phrase import _vib_score_from_params
from features.vibrato.bandpass import bandpass_vibrato
from features.dynamics.dynamic_class import classify_dynamic
from data.vocalset_labels import REGISTER_VOCAB, DYNAMIC_VOCAB
from features.breath.cpp import compute_cpp
from features.breath.phrase import PhraseTracker, breath_window_from_roll, MIN_PHRASE_FRAMES
from features.breath.spectral_tilt import compute_spectral_tilt_slope
from features.vibrato.bandpass import bandpass_vibrato
from features.vibrato.parameters import extract_vibrato_params
from features.pitch.pitch_drift import analyze_drift, aggregate_drift_score
from model.nanopitch import viterbi_stream
from model.breath_cnn import BreathCNN, DEFAULT_WINDOW as BREATH_WINDOW
from model.gesture_classes import expand_logits_to_4class
from model.gesture_tcn import GestureTCN

# ── Constants ──────────────────────────────────────────────────────────
SR             = 16_000
HOP_LENGTH     = NP_HOP          # 160 — 10 ms per NanoPitch frame at 16 kHz
WIN_LENGTH     = NP_WIN          # 400 — 25 ms analysis window
ROLLING_SECS   = 3
ROLLING_SAMPS  = SR * ROLLING_SECS    # 48 000
ROLLING_FRAMES = ROLLING_SECS * 100   # 300

MIN_AUDIO_SAMPS = NP_WIN            # 400 samples = first mel frame
MIN_VIB_FRAMES  = 120
VIB_LIVE_FRAMES = 90   # ~0.9 s window for live has_vibrato (not whole rolling buffer)
# How often to run the slower DSP operations (in chunks)
CPP_EVERY   = 30   # every ~30 chunks ≈ 300 ms — keeps realtime proc under budget
VIB_EVERY   = 30

app = FastAPI()

# ── Model — loaded once at startup ────────────────────────────────────
# Checkpoint search order: env var, then candidates below (first existing wins).
# exp6 is the production NanoPitch checkpoint (WASM parity). exp4-cosine-logits
# outputs logits that never exceed ~0.27 after sigmoid — Viterbi stays unvoiced.
NANOPITCH_CHECKPOINT_CANDIDATES = [
    Path(__file__).parent.parent / "NanoPitchFork" / "training" / "runs" / "exp6" / "checkpoints" / "best.pth",
    Path(__file__).parent / "runs" / "exp1" / "checkpoints" / "best.pth",
    Path(__file__).parent.parent / "NanoPitchFork" / "training" / "runs" / "exp4-cosine-logits" / "checkpoints" / "best.pth",
]
NANOPITCH_PLUS_CANDIDATES = [
    Path(__file__).parent / "runs" / "vocalset_plus_v3" / "checkpoints" / "best.pth",
    Path(__file__).parent / "runs" / "vocalset_plus" / "checkpoints" / "best.pth",
]
BREATH_CHECKPOINT_CANDIDATES = [
    Path(__file__).parent / "runs" / "breath_cnn" / "checkpoints" / "best.pth",
]
GESTURE_TCN_CANDIDATES = [
    Path(__file__).parent / "runs" / "gesture_tcn_3class" / "checkpoints" / "best.pth",
    Path(__file__).parent / "runs" / "gesture_tcn" / "checkpoints" / "best.pth",
]
# GestureTCN 3-class merge — tuned for precision (fewer steady→transition FPs).
TCN_MERGE_KW = dict(
    transition_conf_min=0.86,
    transition_prob_min=0.52,
    transition_margin=0.18,
    vibrato_prob_min=0.74,
    model_confidence=0.55,
)
_extractor: NanoPitchExtractor | None = None
_breath_model: BreathCNN | None = None
_gesture_tcn: GestureTCN | None = None
_breath_window: int = BREATH_WINDOW
_checkpoint_used: Path | None = None
_breath_checkpoint_used: Path | None = None
_gesture_tcn_checkpoint_used: Path | None = None
_gesture_source: str = "heuristic"  # heuristic | tcn | model


def _resolve_plus_checkpoint() -> Path | None:
    import os
    env = os.environ.get("NANOPITCH_PLUS_CHECKPOINT")
    if env:
        p = Path(env).expanduser()
        return p if p.exists() else None
    for p in NANOPITCH_PLUS_CANDIDATES:
        if p.exists():
            return p
    return None


def _resolve_gesture_tcn_checkpoint() -> Path | None:
    import os
    env = os.environ.get("GESTURE_TCN_CHECKPOINT")
    if env:
        p = Path(env).expanduser()
        return p if p.exists() else None
    for p in GESTURE_TCN_CANDIDATES:
        if p.exists():
            return p
    return None


def _resolve_breath_checkpoint() -> Path | None:
    import os
    env = os.environ.get("BREATH_CHECKPOINT")
    if env:
        p = Path(env).expanduser()
        return p if p.exists() else None
    for p in BREATH_CHECKPOINT_CANDIDATES:
        if p.exists():
            return p
    return None


def _load_breath_model(ckpt_path: Path) -> BreathCNN:
    global _breath_window
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    window = int(ckpt.get("window_size", BREATH_WINDOW))
    model = BreathCNN(window_size=window)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    _breath_window = window
    return model


def _resolve_checkpoint() -> Path | None:
    import os
    env = os.environ.get("NANOPITCH_CHECKPOINT")
    if env:
        p = Path(env).expanduser()
        if p.exists():
            return p
        print(f"[WARNING] NANOPITCH_CHECKPOINT not found: {p}")
    for p in NANOPITCH_CHECKPOINT_CANDIDATES:
        if p.exists():
            return p
    return None


def _load_and_warmup(
    pitch_ckpt: str | None,
    plus_ckpt: str | None,
    *,
    use_plus_head: bool = False,
) -> NanoPitchExtractor:
    """Load pitch model; optionally attach Plus gesture head without replacing pitch weights."""
    if use_plus_head and plus_ckpt and pitch_ckpt:
        ext = NanoPitchExtractor.from_hybrid_checkpoints(pitch_ckpt, plus_ckpt)
    elif use_plus_head and plus_ckpt:
        ext = NanoPitchExtractor.from_checkpoint(plus_ckpt, prefer_plus=True)
    elif pitch_ckpt:
        ext = NanoPitchExtractor.from_checkpoint(pitch_ckpt)
    else:
        ext = NanoPitchExtractor.from_pretrained(local_path=None)

    state = ext.model.init_streaming_state()
    dummy_buf = np.zeros(MEL_MIN_TAIL, dtype=np.float32)
    with torch.no_grad():
        for _ in range(NC_CONV_CONTEXT + 1):
            mf = _mel_frame_from_segment(dummy_buf)
            frame_t = torch.from_numpy(mf).unsqueeze(0).unsqueeze(0)
            ext.model.forward_single_frame(frame_t, state)
    return ext


@app.on_event("startup")
async def _startup() -> None:
    global _extractor, _checkpoint_used, _breath_model, _breath_checkpoint_used
    global _gesture_tcn, _gesture_tcn_checkpoint_used, _gesture_source
    import os
    use_plus = os.environ.get("NANOPITCH_USE_PLUS", "").lower() in ("1", "true", "yes")
    use_tcn = os.environ.get("GESTURE_USE_TCN", "").lower() in ("1", "true", "yes")
    plus_ckpt = _resolve_plus_checkpoint() if use_plus else None
    tcn_ckpt = _resolve_gesture_tcn_checkpoint() if use_tcn else None
    if not use_tcn and not os.environ.get("GESTURE_USE_TCN"):
        auto = _resolve_gesture_tcn_checkpoint()
        if auto is not None:
            tcn_ckpt = auto
            use_tcn = True
    pitch_ckpt = _resolve_checkpoint()
    if pitch_ckpt and "exp4-cosine-logits" in str(pitch_ckpt):
        print(
            "[WARNING] exp4-cosine-logits produces sub-threshold pitch posteriors "
            "(no voiced frames). Use exp6 for live pitch, or set NANOPITCH_CHECKPOINT "
            "to .../exp6/checkpoints/best.pth"
        )

    if use_plus and plus_ckpt and pitch_ckpt:
        _checkpoint_used = plus_ckpt
        _extractor = await asyncio.to_thread(
            _load_and_warmup, str(pitch_ckpt), str(plus_ckpt), use_plus_head=True,
        )
        print(
            f"Hybrid ready — pitch: {pitch_ckpt}  "
            f"gesture head: {plus_ckpt} (source: model)"
        )
    elif use_plus and plus_ckpt:
        _checkpoint_used = plus_ckpt
        _extractor = await asyncio.to_thread(
            _load_and_warmup, None, str(plus_ckpt), use_plus_head=True,
        )
        print(f"NanoPitchPlus ready — {plus_ckpt} (gesture head: model)")
    elif pitch_ckpt:
        _checkpoint_used = pitch_ckpt
        _extractor = await asyncio.to_thread(
            _load_and_warmup, str(pitch_ckpt), None,
        )
        print(f"NanoPitch ready — {pitch_ckpt} (gesture: heuristic f0)")
    else:
        _extractor = await asyncio.to_thread(_load_and_warmup, None, None)
        print("NanoPitch ready — random weights (gesture: heuristic f0)")

    breath_ckpt = _resolve_breath_checkpoint()
    if breath_ckpt:
        _breath_checkpoint_used = breath_ckpt
        _breath_model = await asyncio.to_thread(_load_breath_model, breath_ckpt)
        print(f"BreathCNN ready — {breath_ckpt} (live phrase boundaries)")
    else:
        print("BreathCNN not loaded — phrasing uses silence gaps only")

    _gesture_source = "heuristic"
    if tcn_ckpt:
        _gesture_tcn_checkpoint_used = tcn_ckpt
        _gesture_tcn = await asyncio.to_thread(
            GestureTCN.load_checkpoint, str(tcn_ckpt), "cpu",
        )
        _gesture_source = "tcn"
        ncls = _gesture_tcn.n_classes
        print(
            f"GestureTCN ready — {tcn_ckpt} ({ncls}-class, merge: tcn + heuristics)"
        )
    elif use_plus and plus_ckpt:
        _gesture_source = "model"


_NOTE_NAMES = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']

def _buf_index(buf_len: int, samples_rx: int, abs_sample: int) -> int:
    """Map absolute sample index to index in the rolling buffer."""
    if samples_rx <= buf_len:
        return (buf_len - samples_rx) + abs_sample
    return abs_sample - (samples_rx - buf_len)


def _hz_to_et(hz: float) -> tuple[float, str, float]:
    if hz <= 0:
        return 0.0, '—', 0.0
    # Convert to MIDI note number (A4 = 69).  MIDI is the unambiguous
    # standard: note_idx = midi % 12, octave = midi // 12 - 1
    # (C-1 = MIDI 0, C4 = MIDI 60, A4 = MIDI 69, C5 = MIDI 72 …)
    midi     = 69.0 + 12.0 * np.log2(hz / 440.0)
    midi_n   = int(round(midi))
    et_hz    = 440.0 * 2.0 ** ((midi_n - 69) / 12.0)
    deviation = (midi - midi_n) * 100.0
    note_idx  = midi_n % 12
    octave    = midi_n // 12 - 1
    return et_hz, f"{_NOTE_NAMES[note_idx]}{octave}", float(deviation)


def _et_deviation_cents(hz: float) -> float:
    """Absolute cents from nearest equal-temperament note (matches live chart)."""
    if hz <= 0:
        return 0.0
    return abs(_hz_to_et(hz)[2])


# ── Per-connection state ───────────────────────────────────────────────
@dataclass
class ClientState:
    audio_roll:      np.ndarray = field(default_factory=lambda: np.zeros(ROLLING_SAMPS, dtype=np.float32))
    f0_history:      deque      = field(default_factory=lambda: deque(maxlen=ROLLING_FRAMES))
    raw_f0_history:  deque      = field(default_factory=lambda: deque(maxlen=ROLLING_FRAMES))
    gesture_history: deque      = field(default_factory=lambda: deque(maxlen=ROLLING_FRAMES))
    samples_rx:      int        = 0
    elapsed:         float      = 0.0
    client_sr:       int        = 16000
    chunk_n:         int        = 0
    # cached slow-DSP results (updated every N chunks)
    last_cpp:        float      = 0.0
    last_tilt:       float      = 0.0
    last_dbfs:       float      = -80.0
    last_dynamic:    str        = "silent"
    cached_vib:      np.ndarray = field(default_factory=lambda: np.zeros(1, dtype=np.float32))
    sample_pending:  np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    model_frame_n:   int        = 0
    streaming_state: dict | None = field(default=None)
    viterbi_state:   np.ndarray | None = field(default=None)
    # Vibrato parameters (updated every VIB_EVERY chunks)
    last_vib_rate_hz:     float = float('nan')
    last_has_vibrato:     bool = False
    last_vib_depth_cents: float = float('nan')
    last_vib_consistency: float = 0.0
    phrase_tracker:     PhraseTracker = field(default_factory=PhraseTracker)
    phrase_info:        dict = field(default_factory=dict)
    phrase_boundaries:  list = field(default_factory=list)
    gesture_feat_stream: object | None = field(default=None)


def _pitch_dev_cents(hz: float, central_hz: float | None) -> float:
    """Drift from central pitch (cents); falls back to ET deviation when central unknown."""
    if hz <= 0:
        return 0.0
    if central_hz is not None and central_hz > 0 and not np.isnan(central_hz):
        return abs(1200.0 * np.log2(hz / central_hz + 1e-10))
    _, _, et_dev = _hz_to_et(hz)
    return abs(et_dev)


def _track_phrases(
    state: ClientState,
    buf: np.ndarray,
    frame_indices: list[int],
    f0_arr: np.ndarray,
    gesture_arr: np.ndarray,
    hop_s: float,
    central_arr: np.ndarray | None = None,
) -> tuple[list[int], dict, list[dict]]:
    """Update phrase tracker; return per-frame phrase ids, summary, and boundary events."""
    n = len(f0_arr)
    if n == 0:
        return [], state.phrase_tracker.snapshot(), []

    probs = np.zeros(n, dtype=np.float32)
    if _breath_model is not None and len(frame_indices) == n:
        windows = np.stack([
            breath_window_from_roll(
                buf, state.samples_rx, fi,
                window_size=_breath_window, hop=HOP_LENGTH, buf_index=_buf_index,
            )
            for fi in frame_indices
        ])
        with torch.no_grad():
            probs = _breath_model(
                torch.from_numpy(windows).unsqueeze(0),
            )[0, :, 0].cpu().numpy()

    phrase_ids: list[int] = []
    boundary_events: list[dict] = []
    last_info = state.phrase_tracker.snapshot()
    for i in range(n):
        t = state.elapsed - (n - i - 1) * hop_s
        f0 = float(f0_arr[i])
        g = int(gesture_arr[i]) if i < len(gesture_arr) else GESTURE_STEADY
        central = None
        if central_arr is not None and i < len(central_arr):
            c = float(central_arr[i])
            if c > 0 and not np.isnan(c):
                central = c
        _, _, et_dev_signed = _hz_to_et(f0)
        # Phrase metrics: signed ET cents (matches chart; scoring uses median + bias).
        pitch_dev = et_dev_signed
        # Phrase voicing: avoid phantom voicing, but also avoid splitting on
        # short f0 dropouts during note transitions.
        if not hasattr(state, "_last_phrase_voiced_t"):
            state._last_phrase_voiced_t = 0.0
        audible = state.last_dbfs > -55.0
        voiced_raw = (f0 > 0) and audible
        if voiced_raw:
            state._last_phrase_voiced_t = t
        voiced = voiced_raw or (audible and (t - float(state._last_phrase_voiced_t)) < 0.25)
        last_info = state.phrase_tracker.update(
            float(probs[i]),
            voiced,
            t,
            scored=is_scored_gesture(g),
            et_dev_cents=pitch_dev,
            gesture=g,
            cpp=state.last_cpp,
            f0_hz=f0,
            vib_rate_hz=state.last_vib_rate_hz,
            vib_depth_cents=state.last_vib_depth_cents,
            vib_consistency=state.last_vib_consistency,
        )
        phrase_ids.append(last_info["phrase_id"])
        boundary = last_info.get("phrase_boundary")
        if boundary in ("start", "end"):
            ev_out: dict = {
                "event": boundary,
                "phrase_id": last_info.get("phrase_id", 0),
                "t": round(t, 3),
                "samples": int(round(t * SR)),
                "feedback": last_info.get("phrase_feedback"),
            }
            m = state.phrase_tracker.metrics
            if boundary == "start":
                ev_out["vocal_start_t"] = round(m.start_t, 3)
                ev_out["vocal_start_samples"] = int(round(m.start_t * SR))
            else:
                ev_out["vocal_end_t"] = round(m.end_t, 3)
                ev_out["vocal_end_samples"] = int(round(m.end_t * SR))
                ev_out["phrase_start_t"] = round(
                    float(last_info.get("phrase_start_t", m.start_t)), 3,
                )
                ev_out["vocal_start_t"] = round(m.start_t, 3)
                ev_out["vocal_start_samples"] = int(round(m.start_t * SR))
            boundary_events.append(ev_out)

    return phrase_ids, last_info, boundary_events


# ── Feature extraction ─────────────────────────────────────────────────
def _extract(state: ClientState, chunk: np.ndarray,
             audio_snap: np.ndarray | None = None) -> dict:
    """Run all feature extraction for one audio chunk.

    audio_snap is a snapshot of state.audio_roll taken in the event-loop
    thread before dispatching to the thread pool.  Using the snapshot avoids
    a race with the recv task that keeps rolling the buffer.
    """
    buf = audio_snap if audio_snap is not None else state.audio_roll
    hop_s  = HOP_LENGTH / SR
    state.chunk_n += 1

    # ── Energy / dynamics ────────────────────────────────────────────
    rms_lin = float(np.sqrt(np.mean(chunk ** 2)))
    dbfs    = float(20.0 * np.log10(max(rms_lin, 1e-10)))

    state.last_dbfs    = dbfs
    state.last_dynamic = classify_dynamic(dbfs) or "silent"

    # ── Pitch — streaming NanoPitch + gesture-aware Viterbi ─────────
    f0_arr = np.zeros(0, dtype=np.float32)
    gesture_arr = np.zeros(0, dtype=np.int8)
    frame_indices: list[int] = []

    if state.samples_rx >= MIN_AUDIO_SAMPS and _extractor is not None:
        try:
            if state.streaming_state is None:
                state.streaming_state = _extractor.model.init_streaming_state()

            pending = (
                np.concatenate([state.sample_pending, chunk])
                if len(state.sample_pending) else chunk
            )
            n_hops = len(pending) // HOP_LENGTH
            state.sample_pending = pending[n_hops * HOP_LENGTH:].copy()

            base_abs = state.samples_rx - len(pending)

            posteriors: list[np.ndarray] = []
            gesture_logits:  list[np.ndarray] = []
            register_logits: list[np.ndarray] = []
            model_dyn_logits: list[np.ndarray] = []
            frame_ids: list[int] = []
            for i in range(n_hops):
                hop_end = base_abs + (i + 1) * HOP_LENGTH
                if hop_end < WIN_LENGTH:
                    continue
                frame_idx = (hop_end - WIN_LENGTH) // HOP_LENGTH
                abs_start = frame_idx * HOP_LENGTH
                if state.samples_rx < abs_start + MEL_MIN_TAIL:
                    continue
                pos = _buf_index(len(buf), state.samples_rx, abs_start)
                mel_frame = _mel_frame_from_segment(buf[pos:])
                frame_t = torch.from_numpy(mel_frame).unsqueeze(0).unsqueeze(0)
                with torch.no_grad():
                    if _extractor.has_gesture_head:
                        _, pitch_f, gest_l, reg_l, dyn_l, state.streaming_state = \
                            _extractor.model.forward_single_frame(
                                frame_t, state.streaming_state)
                        gesture_logits.append(gest_l[0, 0].cpu().numpy())
                        register_logits.append(reg_l[0, 0].cpu().numpy())
                        model_dyn_logits.append(dyn_l[0, 0].cpu().numpy())
                    else:
                        _, pitch_f, state.streaming_state = \
                            _extractor.model.forward_single_frame(
                                frame_t, state.streaming_state)
                state.model_frame_n += 1
                posteriors.append(pitch_f[0, 0].cpu().numpy())
                frame_ids.append(state.model_frame_n)
                frame_indices.append(frame_idx)

            if posteriors:
                post = np.stack(posteriors)

                # Gesture from raw argmax f0 (pre-Viterbi) — keeps vibrato modulation.
                prov = provisional_f0_from_posteriors(post)
                for pf in prov:
                    state.raw_f0_history.append(float(pf))
                raw_track = np.array(list(state.raw_f0_history), dtype=np.float32)
                heuristic_gest = classify_gestures_live(raw_track, posteriorgram=post)[-len(prov):]

                if _gesture_tcn is not None:
                    if state.gesture_feat_stream is None:
                        state.gesture_feat_stream = _gesture_tcn.init_stream()
                    tcn_logits: list[np.ndarray] = []
                    for pf, prow in zip(prov, post):
                        voiced = 1.0 if (pf > 0 and float(prow.max()) > 0.08) else 0.0
                        feat_win = state.gesture_feat_stream.push(float(pf), voiced)
                        logit = _gesture_tcn.predict_frame(feat_win)
                        tcn_logits.append(logit.numpy())
                    tcn_stack = np.stack(tcn_logits)
                    if _gesture_tcn.n_classes == 3:
                        tcn_stack = expand_logits_to_4class(tcn_stack)
                    gest_arr = merge_gesture_predictions(
                        heuristic_gest,
                        tcn_stack,
                        **TCN_MERGE_KW,
                    )
                elif gesture_logits:
                    gest_arr = merge_gesture_predictions(
                        heuristic_gest,
                        np.stack(gesture_logits),
                    )
                else:
                    gest_arr = heuristic_gest

                # Standard ±12-bin Viterbi for f0 (matches NanoPitchFork WASM).
                # Gesture labels only drive UI/scoring — not pitch decode.
                # Gesture-aware Viterbi widens paths on vibrato/transition and
                # was dropping voiced frames when labels were noisy.
                f0_raw, state.viterbi_state = viterbi_stream(
                    post, state.viterbi_state,
                )
                for i, frame_n in enumerate(frame_ids):
                    if frame_n <= NC_CONV_CONTEXT:
                        f0_raw[i] = 0.0
                f0_arr = f0_raw.astype(np.float32)

                # Keep gesture labels from raw-track heuristics (+ model), not Viterbi f0.
                gesture_arr = gest_arr.astype(np.int8)

                # ── Register + model dynamics (NanoPitch+ only) ───────────
                def _softmax(x: np.ndarray) -> np.ndarray:
                    e = np.exp(x - x.max())
                    return e / e.sum()

                if register_logits:
                    reg_probs = np.stack([_softmax(l) for l in register_logits]).mean(axis=0)
                    state.register_prob_history.append(reg_probs)
                    smoothed_reg = np.stack(state.register_prob_history).mean(axis=0)
                    state.last_register      = REGISTER_VOCAB[int(smoothed_reg.argmax())]
                    state.last_register_conf = float(smoothed_reg.max())

                if model_dyn_logits:
                    dyn_probs = np.stack([_softmax(l) for l in model_dyn_logits]).mean(axis=0)
                    state.model_dyn_prob_history.append(dyn_probs)
                    smoothed_dyn = np.stack(state.model_dyn_prob_history).mean(axis=0)
                    state.last_model_dynamic = DYNAMIC_VOCAB[int(smoothed_dyn.argmax())]

            else:
                gesture_arr = np.zeros(0, dtype=np.int8)
        except Exception as exc:
            import traceback
            print(f"  [NanoPitch inference error] {exc}")
            traceback.print_exc()

    n_new = len(f0_arr) if len(f0_arr) else max(1, len(chunk) // HOP_LENGTH)
    if len(f0_arr) == 0:
        f0_arr = np.zeros(n_new, dtype=np.float32)
    if len(gesture_arr) == 0:
        gesture_arr = np.full(n_new, GESTURE_STEADY, dtype=np.int8)
    elif len(gesture_arr) != n_new:
        gesture_arr = np.resize(gesture_arr, n_new)

    # ── Update rolling f0 history (gestures appended after overlay) ───
    for f in f0_arr:
        state.f0_history.append(float(f))

    # ── Central pitch (per-frame) ──────────────────────────────────
    f0_roll     = np.array(list(state.f0_history), dtype=np.float32)
    central_arr = compute_central_pitch(f0_roll, method='median')
    if len(central_arr) >= n_new:
        central_latest = central_arr[-n_new:]
    else:
        central_latest = np.full(n_new, np.nan)

    # ── CPP + tilt (every CPP_EVERY chunks; gate on audible voicing)
    if state.chunk_n % CPP_EVERY == 0 and state.samples_rx >= 4096:
        voiced_count = int((f0_roll > 0).sum())
        audible = state.last_dbfs > -55.0
        cpp_len = max(len(chunk) * CPP_EVERY, 1600)
        cpp_buf = buf[-cpp_len:].copy()
        if audible and voiced_count >= 20:
            state.last_cpp  = float(compute_cpp(cpp_buf, sr=SR))
            state.last_tilt = float(compute_spectral_tilt_slope(cpp_buf, sr=SR))
        else:
            state.last_cpp = 0.0
            state.last_tilt = 0.0

    # ── Vibrato (every VIB_EVERY chunks) ──────────────────────────
    if state.chunk_n % VIB_EVERY == 0:
        voiced_count = int((f0_roll > 0).sum())
        if voiced_count >= MIN_VIB_FRAMES:
            try:
                vib_full = bandpass_vibrato(
                    f0_roll, central_hz=central_arr, frame_rate_hz=100.0,
                )
                valid    = ~np.isnan(vib_full)
                if valid.sum() > 20:
                    win = min(21, max(5, (valid.sum() // 4) * 2 + 1))
                    vib_full[valid] = savgol_filter(vib_full[valid], win, 3)
                state.cached_vib = np.clip(vib_full, -80, 80).astype(np.float32)
            except Exception:
                pass
        else:
            state.cached_vib = np.zeros(max(len(f0_roll), 1), dtype=np.float32)

    vib_latest = (
        state.cached_vib[-n_new:].astype(np.float32)
        if len(state.cached_vib) >= n_new
        else np.zeros(n_new, dtype=np.float32)
    )

    # Vibrato / transition reconciliation (no glissando boost — FP hurts coaching).
    if len(vib_latest) and np.any(f0_arr > 0):
        gesture_arr = overlay_vibrato_from_deviation(
            gesture_arr, f0_arr, vib_latest,
        ).astype(np.int8)
        gesture_arr = reconcile_false_vibrato(
            gesture_arr, f0_arr, vib_latest,
        ).astype(np.int8)
        gesture_arr = reconcile_false_glissando(
            gesture_arr, f0_arr, vib_latest,
        ).astype(np.int8)
        gesture_arr = demote_glissando_for_coaching(
            gesture_arr, f0_arr, vib_latest,
        ).astype(np.int8)
        gesture_arr = coalesce_glissando_labels(gesture_arr).astype(np.int8)

    for g in gesture_arr:
        state.gesture_history.append(int(g))

    if state.chunk_n % VIB_EVERY == 0 and len(state.cached_vib) > 10:
        try:
            vib_win = state.cached_vib[-VIB_LIVE_FRAMES:]
            vp = extract_vibrato_params(vib_win, frame_rate_hz=100.0, live=True)
            if vp is not None:
                state.last_vib_consistency = vp.consistency
                hist = np.array(list(state.gesture_history)[-45:], dtype=np.int8)
                vib_frac = (
                    float(np.mean(hist == GESTURE_VIBRATO)) if len(hist) else 0.0
                )
                periodic = segment_has_vibrato_period(vib_win, **VIB_PERIOD_LIVE)
                state.last_has_vibrato = bool(
                    vp.has_vibrato and (vib_frac >= 0.08 or periodic),
                )
                if state.last_has_vibrato:
                    state.last_vib_rate_hz     = vp.rate_hz
                    state.last_vib_depth_cents = vp.depth_cents
                else:
                    state.last_vib_rate_hz     = float("nan")
                    state.last_vib_depth_cents = float("nan")
        except Exception:
            pass

    hist_gate = np.array(list(state.gesture_history)[-45:], dtype=np.int8)
    vib_recent = state.cached_vib[-min(45, len(state.cached_vib)):]
    periodic_now = (
        segment_has_vibrato_period(vib_recent, **VIB_PERIOD_LIVE)
        if len(vib_recent) > 15 else False
    )
    if len(hist_gate) and float(np.mean(hist_gate == GESTURE_VIBRATO)) < 0.06:
        if not periodic_now:
            state.last_has_vibrato = False

    # Phrase scoring uses final gestures + central-relative pitch drift.
    phrase_ids, state.phrase_info, phrase_boundary_events = _track_phrases(
        state, buf, frame_indices, f0_arr, gesture_arr, hop_s,
        central_arr=central_latest,
    )
    if len(phrase_ids) < n_new:
        phrase_ids = phrase_ids + [0] * (n_new - len(phrase_ids))
    elif len(phrase_ids) > n_new:
        phrase_ids = phrase_ids[:n_new]

    # ── ET deviation + gesture display ───────────────────────────────
    # Use the last voiced frame in this batch for summary (not the trailing
    # unvoiced tail), so brief decode gaps don't wipe the live readout.
    f0_now = 0.0
    gesture_now = GESTURE_STEADY
    for i in range(len(f0_arr) - 1, -1, -1):
        if f0_arr[i] > 0:
            f0_now = float(f0_arr[i])
            gesture_now = int(gesture_arr[i]) if i < len(gesture_arr) else GESTURE_STEADY
            break
    vib_display_win     = min(45, len(state.cached_vib))
    gesture_hist        = np.array(list(state.gesture_history)[-vib_display_win:], dtype=np.int8)
    gesture_display     = smooth_gesture_label(
        gesture_hist,
        vib_recent=state.cached_vib[-vib_display_win:],
    )
    recent_slice        = gesture_hist[-45:]
    win_len             = max(len(recent_slice), 1)
    glissando_active    = (
        int(np.sum(recent_slice == GESTURE_GLISSANDO)) >= max(18, int(win_len * 0.30))
    )
    n_vib_recent = int(np.sum(recent_slice == GESTURE_VIBRATO))
    vibrato_active      = (
        n_vib_recent >= max(7, int(win_len * 0.12))
        and (state.last_has_vibrato or periodic_now)
    )
    _, et_note, et_dev = _hz_to_et(f0_now)

    # ── Build per-frame list ───────────────────────────────────────
    frames = []
    for i, f in enumerate(f0_arr):
        c = float(central_latest[i]) if i < len(central_latest) and not np.isnan(central_latest[i]) else None
        g = int(gesture_arr[i]) if i < len(gesture_arr) else GESTURE_STEADY
        vi = float(vib_latest[i]) if i < len(vib_latest) else np.nan
        frames.append({
            "t":       round(state.elapsed - (n_new - i - 1) * hop_s, 3),
            "f0":      round(float(f), 2)  if float(f) > 0     else None,
            "central": round(c, 2)          if c and c > 0      else None,
            "vib":     round(vi, 2)
                       if float(f) > 0 and not np.isnan(vi) else None,
            "gesture": gesture_name(g),
            "scored":  is_scored_gesture(g),
            "phrase_id": int(phrase_ids[i]) if i < len(phrase_ids) else 0,
        })

    pi = state.phrase_info or state.phrase_tracker.snapshot()
    pf = pi.get("phrase_feedback")
    boundary = pi.get("phrase_boundary")

    phrase_events_batch: list[dict] = []
    for be in phrase_boundary_events:
        _bt = float(be.get("t", state.elapsed))
        ev = {
            "event": be["event"],
            "phrase_id": be.get("phrase_id", 0),
            "t": _bt,
            "samples": int(be.get("samples", round(_bt * SR))),
            "feedback": be.get("feedback") if be["event"] == "end" else None,
        }
        if be["event"] == "start":
            if "vocal_start_t" in be:
                ev["vocal_start_t"] = be["vocal_start_t"]
            if "vocal_start_samples" in be:
                ev["vocal_start_samples"] = be["vocal_start_samples"]
        elif be["event"] == "end":
            if "vocal_end_t" in be:
                ev["vocal_end_t"] = be["vocal_end_t"]
            if "vocal_end_samples" in be:
                ev["vocal_end_samples"] = be["vocal_end_samples"]
            if "phrase_start_t" in be:
                ev["phrase_start_t"] = be["phrase_start_t"]
            if "vocal_start_t" in be:
                ev["vocal_start_t"] = be["vocal_start_t"]
            if "vocal_start_samples" in be:
                ev["vocal_start_samples"] = be["vocal_start_samples"]
        phrase_events_batch.append(ev)
        state.phrase_boundaries.append(ev)

    phrase_event = phrase_events_batch[-1] if phrase_events_batch else None
    if phrase_event is None and boundary in ("start", "end"):
        _t = (
            pi.get("phrase_start_t", 0.0) if boundary == "start"
            else state.elapsed
        )
        phrase_event = {
            "event": boundary,
            "phrase_id": pi.get("phrase_id", 0),
            "t": round(_t, 3),
            "samples": int(round(float(_t) * SR)),
            "feedback": pf if boundary == "end" else None,
        }
        m = state.phrase_tracker.metrics
        if boundary == "start":
            phrase_event["vocal_start_t"] = round(m.start_t, 3)
            phrase_event["vocal_start_samples"] = int(round(m.start_t * SR))
        elif boundary == "end":
            phrase_event["vocal_end_t"] = round(m.end_t, 3)
            phrase_event["vocal_end_samples"] = int(round(m.end_t * SR))
        state.phrase_boundaries.append(phrase_event)

    return {
        "frames": frames,
        "phrase_events": phrase_events_batch,
        "phrase_event": phrase_event,
        "summary": {
            "dbfs":             round(state.last_dbfs, 1),
            "dynamic":          state.last_dynamic,
            "cpp":              round(state.last_cpp, 2),
            "tilt":             round(state.last_tilt, 1),
            "et_note":          et_note,
            "et_dev_cents":     round(et_dev, 1),
            "gesture":          gesture_name(gesture_display),
            "gesture_raw":      gesture_name(gesture_now),
            "glissando_active": glissando_active,
            "vibrato_active":   vibrato_active,
            "gesture_source":   _gesture_source,
            "vib_rate_hz":      None if np.isnan(state.last_vib_rate_hz)     else round(state.last_vib_rate_hz, 2),
            "has_vibrato":      state.last_has_vibrato,
            "vib_depth_cents":  None if np.isnan(state.last_vib_depth_cents) else round(state.last_vib_depth_cents, 1),
            "vib_consistency":  round(state.last_vib_consistency, 3),
            "phrase_id":        pi.get("phrase_id", 0),
            "phrase_start_t":   pi.get("phrase_start_t", 0.0),
            "phrase_boundary":  boundary,
            "breath_prob":      pi.get("breath_prob", 0.0),
            "in_phrase":        pi.get("in_phrase", False),
            "pending_phrase":   pi.get("pending_phrase", False),
            "phrase_singing":   bool(
                pi.get("in_phrase") or pi.get("pending_phrase")
            ) and state.phrase_tracker.phrase_voiced_frames > 0,
            "phrase_feedback":  pf,
            "phrase_boundaries": state.phrase_boundaries[-30:],
        },
    }


# ── WebSocket handler ──────────────────────────────────────────────────
@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket) -> None:
    await ws.accept()
    print("Browser connected")
    state   = ClientState()
    # Process every chunk in order — dropping desyncs GRU/Viterbi from real time.
    q: asyncio.Queue = asyncio.Queue()
    _MAX_COALESCE = 12   # up to ~120 ms per inference when catching up
    _BACKLOG_WARN = 50   # log if we fall >500 ms behind
    _backlog_warned = False

    def _ingest(raw: bytes) -> np.ndarray | None:
        """Resample and roll the audio buffer.  Called from the recv task."""
        chunk_raw = np.frombuffer(raw, dtype=np.float32).copy()
        if len(chunk_raw) < 32:
            return None
        chunk = (
            librosa.resample(chunk_raw, orig_sr=state.client_sr, target_sr=SR)
            if abs(state.client_sr - SR) > 10 and len(chunk_raw) != HOP_LENGTH
            else chunk_raw
        )
        n = len(chunk)
        if n >= ROLLING_SAMPS:
            state.audio_roll[:] = chunk[-ROLLING_SAMPS:]
        else:
            state.audio_roll[:-n] = state.audio_roll[n:]
            state.audio_roll[-n:] = chunk
        state.samples_rx += n
        state.elapsed    += n / SR
        return chunk

    async def _recv_task() -> None:
        """Receive every audio chunk, always keep the buffer fresh."""
        try:
            while True:
                msg = await ws.receive()
                if msg["type"] == "websocket.disconnect":
                    break
                if msg.get("text"):
                    try:
                        data = json.loads(msg["text"])
                        state.client_sr = int(data.get("sr", 44100))
                        print(f"  Client SR: {state.client_sr} Hz")
                    except Exception:
                        pass
                    continue
                raw = msg.get("bytes")
                if not raw:
                    continue
                chunk = _ingest(raw)
                if chunk is None:
                    continue
                await q.put(chunk)
                nonlocal _backlog_warned
                if not _backlog_warned and q.qsize() >= _BACKLOG_WARN:
                    print(f"  [WARNING] Processing backlog: {q.qsize()} chunks (~{q.qsize() * 10} ms)")
                    _backlog_warned = True
        finally:
            with contextlib.suppress(asyncio.QueueFull):
                q.put_nowait(None)           # signal EOF to proc task

    async def _proc_task() -> None:
        """Process every audio chunk in order; coalesce backlog to catch up."""
        while True:
            first = await q.get()
            if first is None:
                break
            chunks = [first]
            eof = False
            while len(chunks) < _MAX_COALESCE:
                try:
                    nxt = q.get_nowait()
                except asyncio.QueueEmpty:
                    break
                if nxt is None:
                    eof = True
                    break
                chunks.append(nxt)
            chunk = np.concatenate(chunks) if len(chunks) > 1 else chunks[0]
            audio_snap = state.audio_roll.copy()
            try:
                result = await asyncio.to_thread(_extract, state, chunk, audio_snap)
                await ws.send_text(json.dumps(result))
            except WebSocketDisconnect:
                break
            except RuntimeError as e:
                if "websocket.send" in str(e):
                    break
                print(f"  Error: {e}")
            except Exception as e:
                print(f"  Error: {e}")
            if eof:
                break

    recv = asyncio.create_task(_recv_task())
    proc = asyncio.create_task(_proc_task())
    try:
        await asyncio.gather(recv, proc)
    except Exception:
        pass
    finally:
        recv.cancel()
        proc.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await recv
        with contextlib.suppress(asyncio.CancelledError):
            await proc
        print("Browser disconnected")


# ── Post-session analysis ───────────────────────────────────────────────

def _segment_phrases(
    f0: np.ndarray,
    t: np.ndarray,
    min_silence_s: float = 0.4,
    min_phrase_s: float = 0.5,
) -> list[tuple[int, int]]:
    """Split f0 array into phrases separated by silence gaps."""
    fps = 100.0
    if len(t) > 2:
        diffs = np.diff(t[:min(50, len(t))])
        valid = diffs[(diffs > 0) & (diffs < 0.5)]
        if len(valid):
            fps = float(np.clip(1.0 / float(np.median(valid)), 10.0, 200.0))

    min_sil = max(1, int(min_silence_s * fps))
    min_len = max(5, int(min_phrase_s  * fps))
    voiced  = f0 > 0

    phrases: list[tuple[int, int]] = []
    in_seg = False
    seg_start = 0
    silence_run = 0

    for i, v in enumerate(voiced):
        if v:
            if not in_seg:
                in_seg = True
                seg_start = i
            silence_run = 0
        else:
            if in_seg:
                silence_run += 1
                if silence_run >= min_sil:
                    end = i - silence_run + 1
                    if end - seg_start >= min_len:
                        phrases.append((seg_start, end))
                    in_seg = False
                    silence_run = 0

    if in_seg:
        end = len(f0)
        if end - seg_start >= min_len:
            phrases.append((seg_start, end))

    return phrases or [(0, len(f0))]


MIN_PHRASE_S = MIN_PHRASE_FRAMES / 100.0


def _phrase_ranges_from_events(
    phrase_events: list[dict],
    t: np.ndarray,
    session_end_t: float,
) -> list[tuple[int, int, float, float, int, int, int, float]]:
    """Map live start/end events to analysis phrase ranges (one card per line)."""
    if not phrase_events or len(t) == 0:
        return []

    events_sorted = sorted(
        [
            e for e in phrase_events
            if isinstance(e, dict) and str(e.get("event")) in ("start", "end")
        ],
        key=lambda e: (float(e.get("t", 0.0)), str(e.get("event"))),
    )
    starts: dict[int, dict] = {}
    ends: dict[int, dict] = {}

    for e in events_sorted:
        pid = int(e.get("phrase_id", 0) or 0)
        if pid <= 0:
            continue
        ev = str(e.get("event"))
        if ev == "start":
            starts[pid] = e
        elif ev == "end":
            ends[pid] = e

    ranges: list[tuple[int, int, float, float, int, int, int, float]] = []
    seen: set[int] = set()

    def _append(pid: int, start_ev: dict | None, end_ev: dict) -> None:
        if pid in seen:
            return
        end_t = float(end_ev.get("t", 0.0) or 0.0)
        boundary_ts = float(
            (start_ev or {}).get("t")
            or end_ev.get("phrase_start_t")
            or end_ev.get("boundary_start_t")
            or 0.0,
        )
        if boundary_ts <= 0:
            boundary_ts = max(0.0, end_t - 1.0)
        if end_t <= boundary_ts:
            return
        if (end_t - boundary_ts) < MIN_PHRASE_S * 0.75:
            return

        vocal_ts = float(
            (start_ev or {}).get("vocal_start_t")
            or end_ev.get("vocal_start_t")
            or boundary_ts,
        )
        v_end_t = float(end_ev.get("vocal_end_t") or end_t)
        start_s = int(
            (start_ev or {}).get("vocal_start_samples")
            or end_ev.get("vocal_start_samples")
            or round(vocal_ts * SR),
        )
        end_s = int(end_ev.get("vocal_end_samples") or round(v_end_t * SR))

        i0 = int(np.searchsorted(t, min(vocal_ts, boundary_ts), side="left"))
        i1 = int(np.searchsorted(t, v_end_t, side="right"))
        if i1 <= i0:
            i0 = int(np.searchsorted(t, boundary_ts, side="left"))
            i1 = int(np.searchsorted(t, end_t, side="right"))
        if i1 <= i0:
            i1 = min(len(t), i0 + 5)
        if i1 - i0 < 3:
            return

        ranges.append((
            max(0, i0), min(len(t), i1),
            vocal_ts, v_end_t,
            start_s, end_s,
            pid, boundary_ts,
        ))
        seen.add(pid)

    for pid, end_ev in sorted(ends.items()):
        _append(pid, starts.get(pid), end_ev)

    # Lines still in progress when the session ends (start marker, no end yet).
    for pid, start_ev in sorted(starts.items()):
        if pid in seen:
            continue
        start_t = float(start_ev.get("t", 0.0) or 0.0)
        if session_end_t - start_t < MIN_PHRASE_S * 0.75:
            continue
        _append(pid, start_ev, {
            "t": session_end_t,
            "vocal_end_t": session_end_t,
            "vocal_end_samples": int(round(session_end_t * SR)),
            "phrase_start_t": start_t,
            "vocal_start_t": start_ev.get("vocal_start_t"),
            "vocal_start_samples": start_ev.get("vocal_start_samples"),
        })

    ranges.sort(key=lambda r: (r[7], r[6]))
    return ranges


def _phrase_bullets(
    pitch_score: float,
    drift_results: list,
    vp,
    cpp_mean: float,
    cpp_std: float,
    dbfs_std: float,
    is_breathy_outlier: bool = False,
) -> list[str]:
    bullets: list[str] = []

    # Pitch accuracy
    if pitch_score >= 82:
        bullets.append("Pitch stable")
    elif pitch_score >= 65:
        bullets.append("Pitch mostly stable")
    else:
        if drift_results:
            trend = float(np.mean([r.trend_cents_s for r in drift_results]))
            if trend < -5.0:
                bullets.append("Pitch drifted flat")
            elif trend > 5.0:
                bullets.append("Pitch drifted sharp")
            else:
                bullets.append("Pitch accuracy needs work")
        else:
            bullets.append("Pitch accuracy needs work")

    # Vibrato (only if vibrato was detected)
    if vp is not None and vp.has_vibrato:
        rate_ok  = 5.0 <= vp.rate_hz <= 7.0
        depth_ok = 25.0 <= vp.depth_cents / 2.0 <= 75.0
        if rate_ok and depth_ok and vp.consistency > 0.2:
            bullets.append("Nice vibrato")
        elif vp.consistency < 0.15:
            bullets.append("Vibrato irregular")
        elif not rate_ok:
            direction = "slow" if vp.rate_hz < 5.0 else "fast"
            bullets.append(f"Vibrato rate {direction} ({vp.rate_hz:.1f} Hz)")
        else:
            bullets.append("Vibrato depth off-range")

    # Breathiness / breath support (cross-phrase outlier takes priority)
    if is_breathy_outlier:
        bullets.append("Much breathier than your other phrases")
    elif cpp_mean > 9.0:
        if cpp_std > 3.5:
            bullets.append("Clear voice, but breath support varies")
        else:
            bullets.append("Great breath support")
    elif cpp_mean > 5.0:
        if cpp_std > 3.5:
            bullets.append("Breath support varies — try to stay consistent")
        else:
            bullets.append("Good breath support")
    elif cpp_mean > 2.0:
        bullets.append("Breathiness rising")
    else:
        bullets.append("Weak breath support")

    return bullets[:3]


def _session_level_variation_db(
    sum_dbfs: np.ndarray,
    phrase_mean_dbfs: list[float],
) -> tuple[float, float]:
    """Return (session std, peak-to-peak) over voiced summaries + cross-phrase spread."""
    voiced = sum_dbfs > -55.0
    session_std = session_range = 0.0
    if voiced.sum() > 5:
        v = sum_dbfs[voiced]
        session_std = float(np.std(v))
        session_range = float(np.ptp(v))
    between = float(np.std(phrase_mean_dbfs)) if len(phrase_mean_dbfs) > 1 else 0.0
    variation = max(session_std, between, session_range * 0.45)
    return variation, session_range


def _run_analysis(frames: list[dict], summaries: list[dict], phrase_events: list[dict] | None = None) -> dict:
    """Run post-session analysis and return structured results."""
    if len(frames) < 20:
        return {"error": "not_enough_data", "session_score": 0,
                "scores": {}, "phrases": [], "trend": {}, "total_duration": 0}

    # ── Build numpy arrays ────────────────────────────────────────────
    f0      = np.array([f.get("f0") or 0.0 for f in frames], dtype=np.float32)
    central = np.array(
        [f.get("central") if f.get("central") is not None else float("nan") for f in frames],
        dtype=np.float32,
    )
    vib = np.array(
        [f.get("vib") if f.get("vib") is not None else float("nan") for f in frames],
        dtype=np.float32,
    )
    # Signed ET deviation from nearest semitone (recompute from f0 so score matches chart).
    et_dev = np.array(
        [
            _hz_to_et(float(f0[i]))[2] if f0[i] > 0 else 0.0
            for i in range(len(f0))
        ],
        dtype=np.float32,
    )
    # scored = steady only; gliss/transition use slide-control in combined_pitch_score
    scored  = np.array([bool(f.get("scored", True)) for f in frames], dtype=bool)
    gestures = np.array(
        [gesture_index(str(f.get("gesture", "steady"))) for f in frames],
        dtype=np.int8,
    )
    t = np.array([f.get("t", i * 0.01) for i, f in enumerate(frames)], dtype=np.float32)

    # Estimate frame rate from timestamps
    fps = 100.0
    if len(t) > 2:
        diffs = np.diff(t[:min(50, len(t))])
        valid = diffs[(diffs > 0) & (diffs < 0.5)]
        if len(valid):
            fps = float(np.clip(1.0 / float(np.median(valid)), 10.0, 200.0))

    # Summary arrays
    sum_t    = np.array([float(s.get("t", i * 0.1))    for i, s in enumerate(summaries)], dtype=np.float32)
    sum_cpp  = np.array([float(s.get("cpp",   0.0))    for s in summaries], dtype=np.float32)
    sum_dbfs = np.array([float(s.get("dbfs", -80.0))   for s in summaries], dtype=np.float32)

    # Live phrase feedback at end boundaries (same scores the coaching card used).
    phrase_live_feedback: dict[int, dict] = {}
    if phrase_events:
        for e in phrase_events:
            if (
                isinstance(e, dict)
                and str(e.get("event")) == "end"
                and e.get("feedback")
                and int(e.get("phrase_id", 0) or 0) > 0
            ):
                phrase_live_feedback[int(e["phrase_id"])] = e["feedback"]

    sum_has_vib = np.array(
        [bool(s.get("has_vibrato", False)) for s in summaries], dtype=bool,
    )
    sum_vib_rate = np.array(
        [float(s.get("vib_rate_hz", float("nan")) or float("nan")) for s in summaries],
        dtype=np.float32,
    )
    sum_vib_depth = np.array(
        [float(s.get("vib_depth_cents", float("nan")) or float("nan")) for s in summaries],
        dtype=np.float32,
    )
    sum_vib_cons = np.array(
        [float(s.get("vib_consistency", 0.0) or 0.0) for s in summaries],
        dtype=np.float32,
    )

    # (frame_start, frame_end, vocal_start_t, vocal_end_t, start_sample, end_sample,
    #  phrase_id, boundary_start_t)
    session_end_t = float(t[-1]) if len(t) else 0.0
    phrase_ranges = _phrase_ranges_from_events(phrase_events or [], t, session_end_t)

    if not phrase_ranges:
        for seg_idx, (i0, i1) in enumerate(
            _segment_phrases(f0, t, min_silence_s=0.55, min_phrase_s=0.7),
        ):
            ts = float(t[i0])
            te = float(t[min(i1 - 1, len(t) - 1)])
            phrase_ranges.append((
                i0, i1, ts, te,
                int(round(ts * SR)), int(round(te * SR)),
                seg_idx + 1,
                ts,
            ))

    phrase_ranges.sort(key=lambda r: (r[6], r[7]))

    # ── Phase 1: compute per-phrase metrics ───────────────────────────
    phrase_metrics: list[dict] = []
    phrase_mean_dbfs_list: list[float] = []
    for _idx, (
        start, end, t_start, t_end, start_smp, end_smp, phrase_id, boundary_start_t,
    ) in enumerate(phrase_ranges):
        f0_p      = f0[start:end]
        central_p = central[start:end]
        vib_p     = vib[start:end]
        et_dev_p  = et_dev[start:end]
        scored_p  = scored[start:end]
        n_voiced  = int((f0_p > 0).sum())

        # Pitch accuracy = ET deviation on steady voiced frames (primary)
        # + drift stability from central pitch (secondary, detects wandering)
        drift_results = analyze_drift(
            f0_p,
            central_p,
            frame_hop_s=1.0 / fps,
            scored_mask=scored_p & (f0_p > 0),
        )
        drift_score   = aggregate_drift_score(drift_results)
        gestures_p = gestures[start:end]
        pitch_score = combined_pitch_score(
            et_dev_p, f0_p, scored_p, drift_score, gestures_p,
        )

        # Vibrato: prefer live phrase-end feedback, then summary flags, then re-analysis.
        vp        = None
        vib_score: float | None = None
        has_vibrato = False
        live_fb = phrase_live_feedback.get(int(phrase_id), {})
        if live_fb.get("vib_score") is not None:
            vib_score = float(live_fb["vib_score"])
            has_vibrato = True
        phrase_gestures = [str(f.get("gesture", "steady")) for f in frames[start:end]]
        vib_label_frac = (
            sum(1 for g in phrase_gestures if g == "vibrato") / max(len(phrase_gestures), 1)
        )
        if len(sum_t):
            sm = (sum_t >= t_start) & (sum_t <= t_end)
            if int(sm.sum()) > 0 and bool(np.any(sum_has_vib[sm])):
                has_vibrato = True
                rates = sum_vib_rate[sm]
                depths = sum_vib_depth[sm]
                cons = sum_vib_cons[sm]
                vr = rates[~np.isnan(rates)]
                vd = depths[~np.isnan(depths)]
                vc = cons[~np.isnan(cons)]
                if len(vr) and vib_score is None:
                    vib_score = _vib_score_from_params(
                        float(np.mean(vr)),
                        float(np.mean(vd)) if len(vd) else float("nan"),
                        float(np.mean(vc)) if len(vc) else 0.0,
                    )
        if vib_score is None and n_voiced >= max(5, int(0.25 * fps)):
            try:
                vib_sig = vib_p
                if int(np.sum(~np.isnan(vib_sig))) < max(8, int(0.15 * len(vib_sig))):
                    vib_sig = bandpass_vibrato(f0_p, central_p, frame_rate_hz=fps)
                vp = extract_vibrato_params(vib_sig, frame_rate_hz=fps, live=True)
                has_vibrato = bool(
                    has_vibrato
                    or (vp is not None and vp.has_vibrato)
                    or vib_label_frac >= 0.08,
                )
                if vp is not None and vp.has_vibrato:
                    vib_score = float(
                        vp.rate_score * 0.4
                        + vp.depth_score * 0.4
                        + min(100.0, vp.consistency * 400.0) * 0.2,
                    )
                elif has_vibrato:
                    vib_score = 58.0
            except Exception:
                if vib_label_frac >= 0.08:
                    has_vibrato = True
                    vib_score = 58.0

        # CPP: level + gentle within-phrase consistency penalty
        if len(sum_t):
            mask    = (sum_t >= t_start) & (sum_t <= t_end)
            cpp_arr = sum_cpp[mask]
        else:
            cpp_arr = np.array([], dtype=np.float32)

        if len(cpp_arr) == 0:
            # No CPP data in this phrase time range — treat breath as unknown.
            cpp_mean     = float("nan")
            cpp_std      = 0.0
            breath_score = 70.0
        else:
            cpp_mean = float(np.mean(cpp_arr))
            cpp_std  = float(np.std(cpp_arr)) if len(cpp_arr) > 2 else 0.0
            # If CPP is near-zero it usually means "not measured" (or very unvoiced),
            # not truly terrible breath support — keep scoring supportive.
            if cpp_mean < 1.5:
                breath_score = 70.0
            else:
                breath_score = float(np.clip(
                    np.interp(cpp_mean,
                              [1.5, 4.0, 7.0, 10.0, 16.0],
                              [58.0, 68.0, 78.0, 88.0, 96.0]),
                    48.0, 96.0,
                ))
                if cpp_std > 2.0:
                    breath_score = max(48.0, breath_score - min(12.0, (cpp_std - 2.0) * 2.0))

        # Loudness: mean level + within-phrase spread (for bullets / per-phrase chart)
        if len(sum_t):
            mask     = (sum_t >= t_start) & (sum_t <= t_end)
            dbfs_arr = sum_dbfs[mask]
        else:
            dbfs_arr = np.array([], dtype=np.float32)
        voiced_dbfs = dbfs_arr[dbfs_arr > -55.0] if len(dbfs_arr) else np.array([], dtype=np.float32)
        if len(voiced_dbfs) > 0:
            phrase_mean_dbfs = float(np.mean(voiced_dbfs))
            dbfs_std = float(np.std(voiced_dbfs)) if len(voiced_dbfs) > 2 else 0.0
            phrase_range_db = float(np.ptp(voiced_dbfs)) if len(voiced_dbfs) > 1 else 0.0
        else:
            phrase_mean_dbfs = -80.0
            dbfs_std = 0.0
            phrase_range_db = 0.0
        phrase_mean_dbfs_list.append(phrase_mean_dbfs)

        phrase_metrics.append({
            "idx":               phrase_id,
            "boundary_start_t":  boundary_start_t,
            "t_start":           t_start,
            "t_end":             t_end,
            "start_sample":      start_smp,
            "end_sample":        end_smp,
            "pitch_score":  pitch_score,
            "drift_results": drift_results,
            "vp":           vp,
            "vib_score":    vib_score,
            "has_vibrato":  has_vibrato,
            "cpp_mean":     cpp_mean,
            "cpp_std":      cpp_std,
            "breath_score": breath_score,
            "dbfs_std":         dbfs_std,
            "phrase_mean_dbfs": phrase_mean_dbfs,
            "phrase_range_db":  phrase_range_db,
        })

    # ── Phase 2: cross-phrase breathiness outlier detection ───────────
    # Flag a phrase as a "breathy outlier" when its mean CPP is notably
    # lower than the session median (i.e. unexpectedly breathy relative
    # to the rest of the session) and falls in the breathy zone (<7 dB).
    cpp_means_arr = np.array([m["cpp_mean"] for m in phrase_metrics])
    if len(cpp_means_arr) > 1:
        session_cpp_med = float(np.median(cpp_means_arr))
        session_cpp_std = float(np.std(cpp_means_arr))
        outlier_gap     = max(3.0, session_cpp_std)
        for m in phrase_metrics:
            m["is_breathy_outlier"] = (
                m["cpp_mean"] < session_cpp_med - outlier_gap
                and m["cpp_mean"] < 7.0
            )
    else:
        for m in phrase_metrics:
            m["is_breathy_outlier"] = False

    level_variation_db, session_loudness_range_db = _session_level_variation_db(
        sum_dbfs, phrase_mean_dbfs_list,
    )
    voiced_phrase_means = [x for x in phrase_mean_dbfs_list if x > -55.0]
    session_level_med = (
        float(np.median(voiced_phrase_means)) if voiced_phrase_means else -80.0
    )

    # ── Phase 3: build final phrase results with bullets ─────────────
    phrase_results = []
    for m in phrase_metrics:
        # Phrase score: same weights as session; omit vibrato when none detected.
        parts: list[tuple[float, float]] = [
            (m["pitch_score"], 0.45),
            (m["breath_score"], 0.30),
        ]
        if m.get("vib_score") is not None:
            parts.insert(1, (m["vib_score"], 0.25))
        wsum = sum(w for _, w in parts)
        phrase_score = sum(s * w for s, w in parts) / wsum
        rating = (
            "Excellent"  if phrase_score >= 82 else
            "Good"       if phrase_score >= 62 else
            "Needs Work"
        )
        bullets = _phrase_bullets(
            m["pitch_score"], m["drift_results"], m["vp"],
            m["cpp_mean"], m["cpp_std"], m["dbfs_std"],
            is_breathy_outlier=m["is_breathy_outlier"],
        )
        if len(voiced_phrase_means) > 1 and len(bullets) < 3:
            delta = m["phrase_mean_dbfs"] - session_level_med
            if delta >= 6.0:
                bullets.append("Louder than your other phrases")
            elif delta <= -6.0:
                bullets.append("Quieter than your other phrases")
        phrase_results.append({
            "idx":            m["idx"],
            "start_t":        round(m["boundary_start_t"], 2),
            "end_t":          round(m["t_end"], 2),
            "start_sample":   int(m["start_sample"]),
            "end_sample":     int(m["end_sample"]),
            "rating":         rating,
            "score":          round(phrase_score, 1),
            "pitch_score":    round(m["pitch_score"], 1),
            "vib_score":      round(m["vib_score"], 1) if m.get("vib_score") is not None else None,
            "has_vibrato":    bool(m.get("has_vibrato")),
            "breath_score":   round(m["breath_score"], 1),
            "phrase_mean_dbfs": round(m["phrase_mean_dbfs"], 1),
            "bullets":        bullets,
        })

    # ── Session-level weighted aggregates ────────────────────────────
    vib_stab: float | None = None
    if phrase_results:
        durations = np.array([p["end_t"] - p["start_t"] for p in phrase_results], dtype=np.float64)
        total_dur = durations.sum()
        w = durations / total_dur if total_dur > 0 else np.ones(len(phrase_results)) / len(phrase_results)
        pitch_acc  = float(np.dot([p["pitch_score"]    for p in phrase_results], w))
        breath_sup = float(np.dot([p["breath_score"]   for p in phrase_results], w))
        vib_phrases = [p for p in phrase_results if p.get("vib_score") is not None]
        if vib_phrases:
            v_w = np.array([p["end_t"] - p["start_t"] for p in vib_phrases], dtype=np.float64)
            v_w = v_w / v_w.sum() if v_w.sum() > 0 else np.ones(len(vib_phrases)) / len(vib_phrases)
            vib_stab = float(np.dot([p["vib_score"] for p in vib_phrases], v_w))
    else:
        pitch_acc = breath_sup = 50.0
        level_variation_db = session_loudness_range_db = 0.0

    # Session score: renormalize when vibrato was not used in the session.
    if vib_stab is not None:
        overall = pitch_acc * 0.45 + vib_stab * 0.25 + breath_sup * 0.30
    else:
        overall = (pitch_acc * 0.45 + breath_sup * 0.30) / 0.75

    scores_out: dict = {
        "pitch_accuracy":    round(pitch_acc, 1),
        "breath_support":    round(breath_sup, 1),
        "level_variation_db": round(level_variation_db, 1),
        "session_loudness_range_db": round(session_loudness_range_db, 1),
    }
    if vib_stab is not None:
        scores_out["vibrato_stability"] = round(vib_stab, 1)

    return {
        "session_score": round(overall, 1),
        "scores": scores_out,
        "phrases": phrase_results,
        "trend": {
            "phrase_labels":   [f"P{p['idx']}" for p in phrase_results],
            "pitch_scores":    [p["pitch_score"]    for p in phrase_results],
            "vib_scores":      [p["vib_score"] if p.get("vib_score") is not None else None for p in phrase_results],
            "breath_scores":   [p["breath_score"]   for p in phrase_results],
            "phrase_mean_dbfs": [p["phrase_mean_dbfs"] for p in phrase_results],
        },
        "total_duration": round(float(t[-1]) if len(t) else 0.0, 2),
    }


@app.post("/analyze")
async def analyze_session(request: Request) -> JSONResponse:
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "invalid_json"}, status_code=400)

    frames    = body.get("frames",    [])
    summaries = body.get("summaries", [])
    phrase_events = body.get("phrase_events", None)

    if not frames:
        return JSONResponse({"error": "no_data"}, status_code=400)

    result = await asyncio.to_thread(_run_analysis, frames, summaries, phrase_events)
    return JSONResponse(result)


@app.get("/")
async def index() -> FileResponse:
    resp = FileResponse(Path(__file__).parent / "index.html")
    resp.headers["Cache-Control"] = "no-store"
    return resp


if __name__ == "__main__":
    print("IRIS ready  →  http://localhost:8000")
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=False, log_level="warning")
