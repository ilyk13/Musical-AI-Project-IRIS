"""IRIS — AI Vocal Coach  ·  Real-time web server

Audio is captured in the BROWSER via the Web Audio API and streamed
as raw Float32 PCM over WebSocket.

Pitch: trained NanoPitch GRU model + gesture-aware scoring.
       Steady frames are scored for ET accuracy; vibrato/glissando/transition
       frames use widened Viterbi transitions and are excluded from accuracy.

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
    GESTURE_STEADY,
    classify_gestures_live,
    gesture_name,
    is_scored_gesture,
    merge_gesture_predictions,
    overlay_vibrato_from_deviation,
    provisional_f0_from_posteriors,
    smooth_gesture_label,
)
from features.pitch.nanopitch import (
    NanoPitchExtractor, _mel_frame_from_segment,
    HOP_LENGTH as NP_HOP, WIN_LENGTH as NP_WIN, NC_CONV_CONTEXT, MEL_MIN_TAIL,
)
from features.pitch.central_pitch import compute_central_pitch
from features.dynamics.dynamic_class import classify_dynamic
from features.breath.cpp import compute_cpp
from features.breath.spectral_tilt import compute_spectral_tilt_slope
from features.vibrato.bandpass import bandpass_vibrato
from features.vibrato.parameters import extract_vibrato_params
from features.pitch.pitch_drift import analyze_drift, aggregate_drift_score
from model.nanopitch import viterbi_stream_gesture

# ── Constants ──────────────────────────────────────────────────────────
SR             = 16_000
HOP_LENGTH     = NP_HOP          # 160 — 10 ms per NanoPitch frame at 16 kHz
WIN_LENGTH     = NP_WIN          # 400 — 25 ms analysis window
ROLLING_SECS   = 3
ROLLING_SAMPS  = SR * ROLLING_SECS    # 48 000
ROLLING_FRAMES = ROLLING_SECS * 100   # 300

MIN_AUDIO_SAMPS = NP_WIN            # 400 samples = first mel frame
MIN_VIB_FRAMES  = 120
# How often to run the slower DSP operations (in chunks)
CPP_EVERY   = 15   # every ~15 chunks ≈ 350 ms
VIB_EVERY   = 15   # every ~15 chunks ≈ 350 ms

app = FastAPI()

# ── Model — loaded once at startup ────────────────────────────────────
# Checkpoint search order: env var, local IRIS training run, NanoPitchFork sibling
NANOPITCH_CHECKPOINT_CANDIDATES = [
    Path(__file__).parent.parent / "NanoPitchFork" / "training" / "runs" / "exp6" / "checkpoints" / "best.pth",
    Path(__file__).parent / "runs" / "exp1" / "checkpoints" / "best.pth",
    Path(__file__).parent.parent / "NanoPitchFork" / "training" / "runs" / "exp4-cosine-logits" / "checkpoints" / "best.pth",
]
NANOPITCH_PLUS_CANDIDATES = [
    Path(__file__).parent / "runs" / "vocalset_plus" / "checkpoints" / "best.pth",
]
_extractor: NanoPitchExtractor | None = None
_checkpoint_used: Path | None = None
_gesture_source: str = "heuristic"  # heuristic | model


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


def _load_and_warmup(pitch_ckpt: str | None, plus_ckpt: str | None) -> NanoPitchExtractor:
    """Load pitch model; optionally replace with NanoPitchPlus for gesture head."""
    global _gesture_source
    if plus_ckpt:
        ext = NanoPitchExtractor.from_checkpoint(plus_ckpt, prefer_plus=True)
        _gesture_source = "model"
    elif pitch_ckpt:
        ext = NanoPitchExtractor.from_checkpoint(pitch_ckpt)
        _gesture_source = "heuristic"
    else:
        ext = NanoPitchExtractor.from_pretrained(local_path=None)
        _gesture_source = "heuristic"

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
    global _extractor, _checkpoint_used
    plus_ckpt = _resolve_plus_checkpoint()
    pitch_ckpt = _resolve_checkpoint()
    if plus_ckpt:
        _checkpoint_used = plus_ckpt
        _extractor = await asyncio.to_thread(
            _load_and_warmup, None, str(plus_ckpt),
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
    last_vib_depth_cents: float = float('nan')
    last_vib_consistency: float = 0.0


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
            gesture_logits: list[np.ndarray] = []
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
                        _, pitch_f, gest_l, _, _, state.streaming_state = \
                            _extractor.model.forward_single_frame(
                                frame_t, state.streaming_state)
                        gesture_logits.append(gest_l[0, 0].cpu().numpy())
                    else:
                        _, pitch_f, state.streaming_state = \
                            _extractor.model.forward_single_frame(
                                frame_t, state.streaming_state)
                state.model_frame_n += 1
                posteriors.append(pitch_f[0, 0].cpu().numpy())
                frame_ids.append(state.model_frame_n)

            if posteriors:
                post = np.stack(posteriors)

                # Gesture from raw argmax f0 (pre-Viterbi) — keeps vibrato modulation.
                prov = provisional_f0_from_posteriors(post)
                for pf in prov:
                    state.raw_f0_history.append(float(pf))
                raw_track = np.array(list(state.raw_f0_history), dtype=np.float32)
                heuristic_gest = classify_gestures_live(raw_track, posteriorgram=post)[-len(prov):]

                if gesture_logits:
                    gest_arr = merge_gesture_predictions(
                        heuristic_gest,
                        np.stack(gesture_logits),
                    )
                else:
                    gest_arr = heuristic_gest

                f0_raw, state.viterbi_state = viterbi_stream_gesture(
                    post, gest_arr, state.viterbi_state,
                )
                for i, frame_n in enumerate(frame_ids):
                    if frame_n <= NC_CONV_CONTEXT:
                        f0_raw[i] = 0.0
                f0_arr = f0_raw.astype(np.float32)

                # Keep gesture labels from raw-track heuristics (+ model), not Viterbi f0.
                gesture_arr = gest_arr.astype(np.int8)
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

    # ── Update rolling f0 + gesture history ───────────────────────
    for f, g in zip(f0_arr, gesture_arr):
        state.f0_history.append(float(f))
        state.gesture_history.append(int(g))

    # ── Central pitch (per-frame) ──────────────────────────────────
    f0_roll     = np.array(list(state.f0_history), dtype=np.float32)
    central_arr = compute_central_pitch(f0_roll, method='median')
    if len(central_arr) >= n_new:
        central_latest = central_arr[-n_new:]
    else:
        central_latest = np.full(n_new, np.nan)

    # ── CPP + tilt (every CPP_EVERY chunks; skip until buffer has real audio)
    if state.chunk_n % CPP_EVERY == 0 and state.samples_rx >= 4096:
        cpp_len = max(len(chunk) * CPP_EVERY, 1600)
        cpp_buf = buf[-cpp_len:].copy()
        state.last_cpp  = float(compute_cpp(cpp_buf, sr=SR))
        state.last_tilt = float(compute_spectral_tilt_slope(cpp_buf, sr=SR))

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

        # Extract vibrato parameters from the refreshed cached signal
        try:
            vp = extract_vibrato_params(state.cached_vib, frame_rate_hz=100.0)
            if vp is not None:
                state.last_vib_consistency = vp.consistency
                if vp.has_vibrato:
                    state.last_vib_rate_hz     = vp.rate_hz
                    state.last_vib_depth_cents = vp.depth_cents
        except Exception:
            pass

    vib_latest = (
        state.cached_vib[-n_new:].astype(np.float32)
        if len(state.cached_vib) >= n_new
        else np.zeros(n_new, dtype=np.float32)
    )

    # Align gesture labels with vibrato deviation chart (same band-pass signal).
    if len(vib_latest) and np.any(f0_arr > 0):
        gesture_arr = overlay_vibrato_from_deviation(
            gesture_arr, f0_arr, vib_latest,
        ).astype(np.int8)
        if n_new > 0 and len(state.gesture_history) >= n_new:
            hist = list(state.gesture_history)
            for i in range(n_new):
                hist[-n_new + i] = int(gesture_arr[i])
            state.gesture_history = deque(hist, maxlen=ROLLING_FRAMES)

    # ── ET deviation + gesture display ───────────────────────────────
    f0_now              = float(f0_arr[-1]) if len(f0_arr) and f0_arr[-1] > 0 else 0.0
    gesture_now         = int(gesture_arr[-1]) if len(gesture_arr) else GESTURE_STEADY
    vib_display_win     = min(45, len(state.cached_vib))
    gesture_display     = smooth_gesture_label(
        np.array(list(state.gesture_history)[-vib_display_win:], dtype=np.int8),
        vib_recent=state.cached_vib[-vib_display_win:],
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
        })

    return {
        "frames": frames,
        "summary": {
            "dbfs":             round(state.last_dbfs, 1),
            "dynamic":          state.last_dynamic,
            "cpp":              round(state.last_cpp, 2),
            "tilt":             round(state.last_tilt, 1),
            "et_note":          et_note,
            "et_dev_cents":     round(et_dev, 1),
            "gesture":          gesture_name(gesture_display),
            "gesture_raw":      gesture_name(gesture_now),
            "gesture_source":   _gesture_source,
            "vib_rate_hz":      None if np.isnan(state.last_vib_rate_hz)     else round(state.last_vib_rate_hz, 2),
            "vib_depth_cents":  None if np.isnan(state.last_vib_depth_cents) else round(state.last_vib_depth_cents, 1),
            "vib_consistency":  round(state.last_vib_consistency, 3),
        },
    }


# ── WebSocket handler ──────────────────────────────────────────────────
@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket) -> None:
    await ws.accept()
    print("Browser connected")
    state   = ClientState()
    # Queue between recv and proc tasks.  maxsize=2 means if processing falls
    # behind, the oldest unprocessed chunk is dropped so we always work on
    # the most recent audio rather than building an ever-growing backlog.
    q: asyncio.Queue = asyncio.Queue(maxsize=2)

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
                # Drop the oldest pending chunk if processing is behind
                if q.full():
                    with contextlib.suppress(asyncio.QueueEmpty):
                        q.get_nowait()
                await q.put(chunk)
        finally:
            with contextlib.suppress(asyncio.QueueFull):
                q.put_nowait(None)           # signal EOF to proc task

    async def _proc_task() -> None:
        """Process the latest chunk and send results.  Runs serially."""
        while True:
            chunk = await q.get()
            if chunk is None:
                break
            # Snapshot the rolling buffer in the event-loop thread
            # before handing off to the worker thread (no race with _ingest).
            audio_snap = state.audio_roll.copy()
            try:
                result = await asyncio.to_thread(_extract, state, chunk, audio_snap)
                await ws.send_text(json.dumps(result))
            except WebSocketDisconnect:
                break
            except Exception as e:
                print(f"  Error: {e}")

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
    if pitch_score >= 85:
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

    # Dynamics (only if there is room)
    if len(bullets) < 3:
        if dbfs_std > 10.0:
            bullets.append("Dynamics inconsistent")
        else:
            bullets.append("Good dynamics")

    return bullets[:3]


def _run_analysis(frames: list[dict], summaries: list[dict]) -> dict:
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
    # ET accuracy per frame: deviation from nearest equal-temperament note
    et_dev  = np.array([float(f.get("et_dev_cents", 0.0)) for f in frames], dtype=np.float32)
    # scored = True only for steady frames (excludes vibrato/glissando/transition)
    scored  = np.array([bool(f.get("scored", True)) for f in frames], dtype=bool)
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

    # ── Phrase segmentation ───────────────────────────────────────────
    phrase_ranges = _segment_phrases(f0, t)

    # ── Phase 1: compute per-phrase metrics ───────────────────────────
    phrase_metrics: list[dict] = []
    for idx, (start, end) in enumerate(phrase_ranges):
        f0_p      = f0[start:end]
        central_p = central[start:end]
        vib_p     = vib[start:end]
        et_dev_p  = et_dev[start:end]
        scored_p  = scored[start:end]
        t_start   = float(t[start])
        t_end     = float(t[min(end - 1, len(t) - 1)])
        n_voiced  = int((f0_p > 0).sum())

        # Pitch accuracy = ET deviation on steady voiced frames (primary)
        # + drift stability from central pitch (secondary, detects wandering)
        drift_results = analyze_drift(f0_p, central_p, frame_hop_s=1.0 / fps)
        drift_score   = aggregate_drift_score(drift_results)

        et_mask = scored_p & (f0_p > 0)
        if et_mask.sum() >= 3:
            mean_et_dev = float(np.mean(np.abs(et_dev_p[et_mask])))
            # 0 cents → 100, 50 cents (max possible) → 0
            et_score = float(max(0.0, 100.0 * (1.0 - mean_et_dev / 50.0)))
        else:
            et_score = drift_score  # not enough steady frames, fall back

        # ET accuracy is the main signal; drift catches gradual wandering
        pitch_score = 0.75 * et_score + 0.25 * drift_score

        # Vibrato parameters
        vp        = None
        vib_score = 50.0
        if n_voiced >= max(5, int(0.25 * fps)):
            try:
                vp = extract_vibrato_params(vib_p, frame_rate_hz=fps)
                if vp is not None and vp.has_vibrato:
                    vib_score = float(
                        vp.rate_score * 0.4
                        + vp.depth_score * 0.4
                        + min(100.0, vp.consistency * 400.0) * 0.2
                    )
            except Exception:
                pass

        # CPP: level + gentle within-phrase consistency penalty
        if len(sum_t):
            mask    = (sum_t >= t_start) & (sum_t <= t_end)
            cpp_arr = sum_cpp[mask]
        else:
            cpp_arr = np.array([], dtype=np.float32)

        if len(cpp_arr) == 0:
            # No CPP data in this phrase time range — neutral score
            cpp_mean     = 0.0
            cpp_std      = 0.0
            breath_score = 50.0
        else:
            cpp_mean = float(np.mean(cpp_arr))
            cpp_std  = float(np.std(cpp_arr)) if len(cpp_arr) > 2 else 0.0
            # Piecewise-linear score calibrated to the live-UI CPP thresholds:
            #   >10 dB (Clear voice)      → 70–100
            #   5–10 dB (Some breathiness)→ 40–70
            #   1.5–5 dB (Breathy)        → 10–40
            #   <1.5 dB                   →  0–10
            breath_score = float(np.clip(
                np.interp(cpp_mean,
                          [0.0,  1.5,  5.0,  10.0, 18.0],
                          [0.0, 10.0, 40.0,  70.0, 100.0]),
                0.0, 100.0,
            ))
            # Gentle within-phrase consistency penalty (caps at −15 pts).
            # Only applied when std exceeds 2 dB (normal variation); the
            # cross-phrase outlier flag handles larger consistency concerns.
            if cpp_std > 2.0:
                breath_score = max(0.0, breath_score - min(15.0, (cpp_std - 2.0) * 2.5))

        # Dynamics control
        if len(sum_t):
            mask     = (sum_t >= t_start) & (sum_t <= t_end)
            dbfs_arr = sum_dbfs[mask]
        else:
            dbfs_arr = np.array([], dtype=np.float32)
        if len(dbfs_arr) > 2:
            dbfs_std  = float(np.std(dbfs_arr))
            if dbfs_std < 2.0:
                dyn_score = 60.0
            elif dbfs_std <= 8.0:
                dyn_score = 100.0
            else:
                dyn_score = float(max(0.0, 100.0 - (dbfs_std - 8.0) * 4.0))
        else:
            dbfs_std  = 5.0
            dyn_score = 75.0

        phrase_metrics.append({
            "idx":          idx,
            "t_start":      t_start,
            "t_end":        t_end,
            "pitch_score":  pitch_score,
            "drift_results": drift_results,
            "vp":           vp,
            "vib_score":    vib_score,
            "cpp_mean":     cpp_mean,
            "cpp_std":      cpp_std,
            "breath_score": breath_score,
            "dbfs_std":     dbfs_std,
            "dyn_score":    dyn_score,
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

    # ── Phase 3: build final phrase results with bullets ─────────────
    phrase_results = []
    for m in phrase_metrics:
        phrase_score = (
            m["pitch_score"] * 0.40
            + m["vib_score"] * 0.20
            + m["breath_score"] * 0.25
            + m["dyn_score"] * 0.15
        )
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
        phrase_results.append({
            "idx":            m["idx"] + 1,
            "start_t":        round(m["t_start"], 2),
            "end_t":          round(m["t_end"], 2),
            "rating":         rating,
            "score":          round(phrase_score, 1),
            "pitch_score":    round(m["pitch_score"], 1),
            "vib_score":      round(m["vib_score"], 1),
            "breath_score":   round(m["breath_score"], 1),
            "dynamics_score": round(m["dyn_score"], 1),
            "bullets":        bullets,
        })

    # ── Session-level weighted aggregates ────────────────────────────
    if phrase_results:
        durations = np.array([p["end_t"] - p["start_t"] for p in phrase_results], dtype=np.float64)
        total_dur = durations.sum()
        w = durations / total_dur if total_dur > 0 else np.ones(len(phrase_results)) / len(phrase_results)
        pitch_acc  = float(np.dot([p["pitch_score"]    for p in phrase_results], w))
        vib_stab   = float(np.dot([p["vib_score"]      for p in phrase_results], w))
        breath_sup = float(np.dot([p["breath_score"]   for p in phrase_results], w))
        dyn_ctrl   = float(np.dot([p["dynamics_score"] for p in phrase_results], w))
    else:
        pitch_acc = vib_stab = breath_sup = dyn_ctrl = 50.0

    overall = pitch_acc * 0.40 + vib_stab * 0.20 + breath_sup * 0.25 + dyn_ctrl * 0.15

    return {
        "session_score": round(overall, 1),
        "scores": {
            "pitch_accuracy":    round(pitch_acc, 1),
            "vibrato_stability": round(vib_stab, 1),
            "breath_support":    round(breath_sup, 1),
            "dynamics_control":  round(dyn_ctrl, 1),
        },
        "phrases": phrase_results,
        "trend": {
            "phrase_labels":   [f"P{p['idx']}" for p in phrase_results],
            "pitch_scores":    [p["pitch_score"]    for p in phrase_results],
            "vib_scores":      [p["vib_score"]      for p in phrase_results],
            "breath_scores":   [p["breath_score"]   for p in phrase_results],
            "dynamics_scores": [p["dynamics_score"] for p in phrase_results],
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

    if not frames:
        return JSONResponse({"error": "no_data"}, status_code=400)

    result = await asyncio.to_thread(_run_analysis, frames, summaries)
    return JSONResponse(result)


@app.get("/")
async def index() -> FileResponse:
    resp = FileResponse(Path(__file__).parent / "index.html")
    resp.headers["Cache-Control"] = "no-store"
    return resp


if __name__ == "__main__":
    print("IRIS ready  →  http://localhost:8000")
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=False, log_level="warning")
