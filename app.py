"""IRIS — AI Vocal Coach  ·  Real-time web server

Audio is captured in the BROWSER via the Web Audio API and streamed
as raw Float32 PCM over WebSocket.

Pitch: trained NanoPitch GRU model (runs/exp1/checkpoints/best.pth).
       Inference runs on a 128 ms rolling window for low latency.
       Falls back to random weights with a warning if no checkpoint is found.

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
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from scipy.signal import medfilt, savgol_filter

sys.path.insert(0, str(Path(__file__).parent))

from features.pitch.nanopitch import NanoPitchExtractor, _compute_mel
from features.pitch.central_pitch import compute_central_pitch
from features.dynamics.dynamic_class import classify_dynamic
from features.breath.cpp import compute_cpp
from features.breath.spectral_tilt import compute_spectral_tilt_slope
from features.vibrato.bandpass import bandpass_vibrato
from model.nanopitch import bin_to_f0, PITCH_BINS

# ── Constants ──────────────────────────────────────────────────────────
SR             = 16_000
HOP_LENGTH     = 160          # 10 ms per NanoPitch frame at 16 kHz
MODEL_WIN      = 1600         # 100 ms context window fed to NanoPitch (10 frames)
ROLLING_SECS   = 3
ROLLING_SAMPS  = SR * ROLLING_SECS    # 48 000
ROLLING_FRAMES = ROLLING_SECS * 100   # 300

MIN_AUDIO_SAMPS = MODEL_WIN   # need this many samples before first pitch estimate
MIN_VIB_FRAMES  = 120
ACCURACY_WIN    = 300
ET_TUNE_CENTS   = 25.0

# How often to run the slower DSP operations (in chunks)
CPP_EVERY   = 15   # every ~15 chunks ≈ 350 ms
VIB_EVERY   = 50   # every ~50 chunks ≈ 1.1 s

app = FastAPI()

# ── Model — loaded once at startup ────────────────────────────────────
CHECKPOINT = Path(__file__).parent / "runs" / "exp1" / "checkpoints" / "best.pth"
_extractor: NanoPitchExtractor | None = None


def _load_and_warmup(local: str | None) -> NanoPitchExtractor:
    """Load model and run a dummy forward pass to trigger any JIT compilation."""
    ext = NanoPitchExtractor.from_pretrained(local_path=local)
    # Warmup: first PyTorch call is slow (internal setup, numba, etc.)
    dummy_mel  = np.zeros((MODEL_WIN,), dtype=np.float32)
    dummy_feat = _compute_mel(dummy_mel, sr=SR)
    dummy_t    = torch.from_numpy(dummy_feat).unsqueeze(0)
    with torch.no_grad():
        ext.model(dummy_t)
    return ext


@app.on_event("startup")
async def _startup() -> None:
    global _extractor
    local = str(CHECKPOINT) if CHECKPOINT.exists() else None
    _extractor = await asyncio.to_thread(_load_and_warmup, local)
    source = f"trained checkpoint ({CHECKPOINT})" if CHECKPOINT.exists() else "random weights"
    print(f"NanoPitch ready — {source}")


_NOTE_NAMES = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']

def _hz_to_et(hz: float) -> tuple[float, str, float]:
    if hz <= 0:
        return 0.0, '—', 0.0
    semitones    = 12.0 * np.log2(hz / 440.0)
    nearest_semi = round(semitones)
    et_hz        = 440.0 * 2.0 ** (nearest_semi / 12.0)
    deviation    = (semitones - nearest_semi) * 100.0
    note_idx     = int((nearest_semi % 12 + 12) % 12)
    octave       = 4 + int(np.floor(nearest_semi / 12))
    return et_hz, f"{_NOTE_NAMES[note_idx]}{octave}", float(deviation)


# ── Stateful Viterbi ───────────────────────────────────────────────────
_N = PITCH_BINS  # 360 voiced bins

def _viterbi_step(
    post: np.ndarray,
    prev: np.ndarray,
    transition_width: int = 12,
    onset_penalty: float = 1.5,
    voicing_threshold: float = 0.2,
) -> tuple[np.ndarray, np.ndarray]:
    """Process one chunk through a stateful greedy Viterbi decoder.

    Args:
        post:  (T, 360) pitch posteriorgram for the new frames
        prev:  (_N+1,) log-probability vector from the previous call
               (all -inf on first call signals cold start)
    Returns:
        f0_hz: (T,) decoded fundamental frequency in Hz (0 = unvoiced)
        new_prev: updated state to pass into the next call
    """
    T = post.shape[0]
    if T == 0:
        return np.zeros(0, dtype=np.float32), prev

    tw = int(transition_width)
    W  = 2 * tw + 1
    f0_hz = np.zeros(T, dtype=np.float32)

    for t in range(T):
        max_p   = float(post[t].max())
        log_obs = np.log(post[t] + 1e-10)
        uv_obs  = np.log(max(1.0 - max_p, 1e-10))
        curr    = np.full(_N + 1, -np.inf, dtype=np.float64)

        if np.all(np.isinf(prev)):
            # Cold start
            if max_p > voicing_threshold:
                curr[:_N] = log_obs
            curr[_N] = uv_obs
        else:
            padded = np.pad(prev[:_N], (tw, tw), constant_values=-np.inf)
            wins   = np.lib.stride_tricks.as_strided(
                padded, shape=(_N, W),
                strides=(padded.strides[0], padded.strides[0]))
            best_v = np.max(wins, axis=1)
            curr[:_N] = np.maximum(best_v, prev[_N] - onset_penalty) + log_obs
            curr[_N]  = max(prev[_N], float(prev[:_N].max()) - onset_penalty) + uv_obs

        best = int(np.argmax(curr))
        if best < _N:
            f0_hz[t] = float(bin_to_f0(float(best)))
        prev = curr

    return f0_hz, prev


# ── Per-connection state ───────────────────────────────────────────────
@dataclass
class ClientState:
    audio_roll:      np.ndarray = field(default_factory=lambda: np.zeros(ROLLING_SAMPS, dtype=np.float32))
    f0_history:      deque      = field(default_factory=lambda: deque(maxlen=ROLLING_FRAMES))
    et_accuracy:     deque      = field(default_factory=lambda: deque(maxlen=ACCURACY_WIN))
    samples_rx:      int        = 0
    elapsed:         float      = 0.0
    client_sr:       int        = 44100
    chunk_n:         int        = 0
    # cached slow-DSP results (updated every N chunks)
    last_cpp:        float      = 0.0
    last_tilt:       float      = 0.0
    last_dbfs:       float      = -80.0
    last_dynamic:    str        = "silent"
    cached_vib:      np.ndarray = field(default_factory=lambda: np.zeros(1, dtype=np.float32))
    # stateful inference — carry GRU hidden states and Viterbi state across chunks
    gru_states:      list | None = field(default=None)
    viterbi_prev:    np.ndarray  = field(
        default_factory=lambda: np.full(_N + 1, -np.inf, dtype=np.float64)
    )


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
    n_new  = max(1, len(chunk) // HOP_LENGTH)
    state.chunk_n += 1

    # ── Energy / VAD ────────────────────────────────────────────────
    rms_lin = float(np.sqrt(np.mean(chunk ** 2)))
    dbfs    = float(20.0 * np.log10(max(rms_lin, 1e-10)))
    voiced  = rms_lin > 0.001  # ~−60 dBFS — low to work without browser AGC

    state.last_dbfs    = dbfs
    state.last_dynamic = classify_dynamic(dbfs) or "silent"

    # ── Pitch — YIN (always) + NanoPitch (accuracy upgrade) ─────────
    # YIN runs unconditionally on every voiced chunk: pure DSP, no training-
    # distribution assumptions, never misses a periodic signal.
    # NanoPitch runs alongside; where its Viterbi is confident (non-zero f0)
    # we prefer its output because it's smoother and more pitch-accurate for
    # singing.  Where NanoPitch is silent, YIN fills in automatically.
    f0_arr = np.zeros(n_new, dtype=np.float32)

    if state.samples_rx >= MIN_AUDIO_SAMPS and voiced:
        model_win = buf[-MODEL_WIN:].copy()

        # ── 1. Fast YIN — always runs, always provides a baseline ────
        f0_yin = np.zeros(n_new, dtype=np.float32)
        try:
            yin_raw = librosa.yin(
                model_win, fmin=65.0, fmax=2093.0, sr=SR,
                hop_length=HOP_LENGTH, frame_length=1024,
            ).astype(np.float32)
            yin_raw = np.where((yin_raw >= 65) & (yin_raw <= 2093), yin_raw, 0.0)
            if len(yin_raw) >= n_new:
                f0_yin = yin_raw[-n_new:]
        except Exception:
            pass

        f0_arr = f0_yin  # YIN is always the fallback

        # ── 2. NanoPitch — upgrades accuracy when the model is confident
        if _extractor is not None:
            try:
                mel   = _compute_mel(model_win, sr=SR)
                mel_t = torch.from_numpy(mel).unsqueeze(0)

                with torch.no_grad():
                    _, pitch_out, new_gru = _extractor.model(
                        mel_t, states=state.gru_states
                    )
                state.gru_states = new_gru

                post  = pitch_out[0].cpu().numpy()
                f0_np, state.viterbi_prev = _viterbi_step(
                    post, state.viterbi_prev
                )
                f0_np = np.where((f0_np >= 65) & (f0_np <= 2093), f0_np, 0.0)

                if len(f0_np) >= n_new:
                    f0_np_latest = f0_np[-n_new:]
                    # Prefer NanoPitch where it's confident; YIN elsewhere
                    f0_arr = np.where(f0_np_latest > 0, f0_np_latest, f0_yin)
            except Exception as exc:
                print(f"  [NanoPitch inference error] {exc}")

        if len(f0_arr) >= 3:
            f0_arr = medfilt(f0_arr, kernel_size=3).astype(np.float32)

    # ── Update rolling f0 history ─────────────────────────────────
    for f in f0_arr:
        state.f0_history.append(float(f))

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
                vib_full = bandpass_vibrato(f0_roll, frame_rate_hz=100.0)
                valid    = ~np.isnan(vib_full)
                if valid.sum() > 20:
                    win = min(21, max(5, (valid.sum() // 4) * 2 + 1))
                    vib_full[valid] = savgol_filter(vib_full[valid], win, 3)
                state.cached_vib = np.clip(
                    np.nan_to_num(vib_full, nan=0.0), -80, 80
                ).astype(np.float32)
            except Exception:
                pass
        else:
            state.cached_vib = np.zeros(max(len(f0_roll), 1), dtype=np.float32)

    vib_latest = (
        state.cached_vib[-n_new:].astype(np.float32)
        if len(state.cached_vib) >= n_new
        else np.zeros(n_new, dtype=np.float32)
    )

    # ── ET deviation + accuracy ────────────────────────────────────
    f0_now              = float(f0_arr[-1]) if len(f0_arr) and f0_arr[-1] > 0 else 0.0
    et_hz, et_note, et_dev = _hz_to_et(f0_now)
    if f0_now > 0:
        state.et_accuracy.append(abs(et_dev) <= ET_TUNE_CENTS)
    accuracy_pct = (
        100.0 * sum(state.et_accuracy) / len(state.et_accuracy)
        if state.et_accuracy else 0.0
    )

    # ── Build per-frame list ───────────────────────────────────────
    frames = []
    for i, f in enumerate(f0_arr):
        c = float(central_latest[i]) if i < len(central_latest) and not np.isnan(central_latest[i]) else None
        frames.append({
            "t":       round(state.elapsed - (n_new - i - 1) * hop_s, 3),
            "f0":      round(float(f), 2)  if float(f) > 0     else None,
            "central": round(c, 2)          if c and c > 0      else None,
            "vib":     round(float(vib_latest[i]) if i < len(vib_latest) else 0.0, 2),
        })

    return {
        "frames": frames,
        "summary": {
            "dbfs":         round(state.last_dbfs, 1),
            "dynamic":      state.last_dynamic,
            "cpp":          round(state.last_cpp, 2),
            "tilt":         round(state.last_tilt, 1),
            "et_note":      et_note,
            "et_dev_cents": round(et_dev, 1),
            "accuracy_pct": round(accuracy_pct, 1),
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
            if abs(state.client_sr - SR) > 10 else chunk_raw
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
                result = await asyncio.to_thread(
                    _extract, state, chunk, audio_snap
                )
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


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(Path(__file__).parent / "index.html")


if __name__ == "__main__":
    print("IRIS ready  →  http://localhost:8000")
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=False, log_level="warning")
