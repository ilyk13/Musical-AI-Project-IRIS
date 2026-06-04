# IRIS — 5-Minute Demo Script

**Audience:** Knows NanoPitch in general (mel → GRU → posteriorgram → Viterbi, ~10 ms frames, browser/WASM). Does **not** need internal training run names.  
**Goal:** Show how IRIS turns that pitch engine into a **real-time vocal coach**, not another pitch demo.  
**Live URL:** `python3 app.py` → http://localhost:8000 → **Start Listening**

---

## Timing overview

| Time | Section |
|------|---------|
| 0:00–0:40 | What IRIS is + one-sentence stack |
| 0:40–1:30 | Pipeline (NanoPitch at the center) |
| 1:30–2:40 | Data & ML (what we trained, what we rejected) |
| 2:40–3:50 | UI & product decisions |
| 3:50–4:40 | Innovations on top of NanoPitch |
| 4:40–5:00 | Failure modes & roadmap |

---

## 0:00 — Opening (40 s)

> **Say:** “NanoPitch gives you a reliable f0 track in the browser. IRIS asks: *given that track*, what should a singer hear back in under 100 ms — pitch, line structure, breath, and phrasing — without turning the chart into noise?”

**One slide / sentence:**

- **IRIS** = FastAPI server + Chart.js dashboard wrapping our **production NanoPitch weights**, plus three coached layers: **GestureTCN**, **BreathCNN**, and classical DSP (CPP, dynamics, vibrato display).

**Optional 10 s live:** Start mic → hum one steady note → point at tuner + pitch trace.

---

## 0:40 — Architecture (50 s)

> **Say:** “We kept the pitch path boring on purpose — same as your NanoPitch mental model.”

```
Mic (16 kHz, 10 ms) ──WebSocket──► FastAPI
                                    │
                    ┌───────────────┼───────────────┐
                    ▼               ▼               ▼
              NanoPitch          GestureTCN        BreathCNN
              Viterbi ±12        on f0 features    480-sample windows
                    │               │               │
                    └───────────────┴───────────────┘
                                    │
                    DSP: CPP, spectral tilt, RMS dynamics
                    PhraseTracker (voicing + breath + silence rules)
                                    │
                                    ▼
                         JSON ~10 Hz ──► Chart.js dashboard
```

**NanoPitch-familiar details (don’t over-explain):**

| Piece | IRIS choice |
|-------|-------------|
| Weights | **Production checkpoint** (exp6) — not the high-RPA / low-recall cosine-logits variant (exp4-cosine-logits) |
| Decode | **Standard ±12-bin streaming Viterbi** — matches WASM; **not** gesture-widened decode |
| Why | Gesture-aware Viterbi was tried; noisy gesture labels **dropped voiced frames** on vibrato-heavy singing |
| Latency | ~10 ms frames; server coalesces backlog; client chart capped ~25 Hz redraw |

**Checkpoint env vars:** `NANOPITCH_CHECKPOINT`, `GESTURE_TCN_CHECKPOINT`, `BREATH_CHECKPOINT`

### Which pitch checkpoint? (speaker note — optional 20 s)

We trained several NanoPitch variants on the same GTSinger-style test harness (realtime Viterbi = what the live app uses). **Say this in plain language, not run names:**

> “One variant only outputs pitch when it’s very sure — great accuracy on the frames it commits to, but it **skips** a lot of real singing. Production uses the variant that **tracks voicing more aggressively** so the chart stays connected; we give up a few points of pitch accuracy on paper for a usable live line.”

| | High-accuracy / conservative voicing (exp4) | **Production** (exp6) |
|---|-------------------------------------------|------------------------|
| **Pitch accuracy (RPA)** on co-voiced frames, clean audio | **~96%** | ~90% |
| **Voicing recall (VDR)** — GT sung frames that get a pitch | ~67% | **~94%** |
| **Median error** on co-voiced frames | **~7¢** | ~13¢ |
| **At −5 dB noise: voicing recall** | ~57% | **~84%** |
| **Why we didn’t ship the conservative one** | Posteriors often too flat after sigmoid for Viterbi’s voicing gate → **blank live chart** | Matches browser/WASM decode; continuous f0 for coaching |

**One-liner for Q&A:** “Higher pitch accuracy vs higher **voicing recall** — we chose recall for a vocal coach.”

---

## 1:30 — Data & ML (70 s)

> **Say:** “All our learned heads sit on **VocalSet + Annotated-VocalSet**, preprocessed to NanoPitch’s grid: 16 kHz, 160-sample hop, HTK mel 0–8 kHz.”

### Training data (`data/vocalset_preprocess.py`)

| Output | Contents |
|--------|----------|
| `train.npz` / `val.npz` | mel, f0, VAD, **gesture** (f0 heuristics), register, dynamics |
| `breath.npz` | 480-sample waveform windows + **breath** labels (unvoiced low-energy gaps between phrases) |

Minimal download: five VocalSet techniques (~1–2 GB audio) + annotations.

### Models (what ships vs what we shelved)

| Model | Params (order of) | Role | Status |
|-------|-------------------|------|--------|
| **NanoPitch** (production / exp6) | ~333K | Pitch + VAD | **Shipped** |
| **GestureTCN** (3-class) | ~63K | steady / vibrato / transition* on 8-D f0 features | **Shipped** |
| **BreathCNN** | small 1D CNN | Inhale / gap detection for phrasing | **Shipped** |
| NanoPitch+ multitask (v4) | shared GRU + extra heads | Joint gesture/register/dynamics | **Shelved** (~30% macro F1 gesture) |

\*Transition is trained and merged offline; **live UI remaps transition → steady** for this demo (pitch-accuracy focus).

### Gesture: why not multitask NanoPitch+?

Offline VocalSet val (same f0 features, different heads):

| Model | Macro F1 | Steady recall | Transition recall | Problem |
|-------|----------|---------------|-------------------|---------|
| NanoPitch+ multitask (v4) | ~30% | 45% | 4% | Gesture head starved; pitch RPA fine |
| GestureTCN (4-class) | ~37% | 35% | 51% | Glissando false positives on steady |
| **GestureTCN (3-class)** | **~51%** | **~78%** | **~60%** | **Shipped** — gliss folded into steady |

**Live merge** (`TCN_MERGE_KW` in `app.py`): high `transition_conf_min` / margin so steady notes aren’t called transitions; then overlay/reconcile vibrato from band-pass deviation; demote unconfirmed glissando.

**Eval tooling:** `model/eval_gesture.py`, `model/eval_merge.py` (merge precision on val).

### Breath / phrases

- **BreathCNN** — BCE with **pos_weight ≈ 8** (rare positive frames).
- **PhraseTracker** (`features/breath/phrase.py`) — state machine on top of BreathCNN + voicing:
  - Start: ~420 ms voiced (`MIN_PHRASE_FRAMES`)
  - End: long silence, sustained inhale (`breath_run`), or **short pause + breath peak** (fast lyrics)
- **App bridge** (`_track_phrases`): separates **note hops** (≤100 ms, low breath) from **lyric breaths** (breath prob peak ≥ 0.38) so fast songs don’t chop every note.

### Pitch scoring (learned + rules)

- **Steady frames only** for ET accuracy (`features/pitch/accuracy.py`) — median-heavy deviation, bias term, instability on frame steps.
- **Glissando** — slide-wander penalty, not “distance from nearest key.”
- **Phrase composite** — pitch + optional vibrato + CPP-based breath support at line end.

---

## 2:40 — UI & product decisions (70 s)

> **Say:** “The hard part wasn’t inference — it was **when not to coach**.”

### Layout (`index.html`)

| Element | Decision |
|---------|----------|
| **Pitch chart** | 30 s scroll; ET coloring on **steady** frames; central pitch overlay |
| **Vibrato chart** | 6 s band-pass **display** signal (3–9 Hz) — not the same gate as vibrato *score* |
| **Tuner** | ±50 ¢ from nearest ET note — immediate feedback |
| **Phrase panel** | In phrase / between phrases; last line score after **end** event |
| **Coaching cards** | Pitch / vibrato / breath tips; **phrase review** card with composite score + bullets |
| **End session** | POST `/analyze` — replays phrases with duration-weighted session bars |

### Gating (silence ≠ singing)

| Gate | Threshold / rule | Why |
|------|------------------|-----|
| `vocal_active` | RMS ≥ **−50 dBFS** + recent voiced f0 | No cards/phrases on room noise |
| Chart f0 | Pitch conf ≥ **0.14** zeroes trace | Chart matches “trust the model” |
| Phrase f0 | Separate **`f0_phrase`** + conf **0.10–0.12** | Lines still track when chart blanks brief dips |
| Coaching | Requires `vocal_active` in summary | Avoid spam when user isn’t singing |

### Latency vs stability

| Knob | Value | Effect |
|------|-------|--------|
| `CHART_MS` | 40 ms | Cap chart redraw rate |
| `CHART_HOLD_MS` | 350 ms | Brief gap-fill on chart only (not coaching) |
| WS coalesce | backlog merge on server | Catch up after load spikes without drowning client |
| CPP / vibrato params | every ~300 ms | Keeps realtime CPU bounded |

### Phrase markers on chart

- Boundaries synced from **`summary.phrase_boundaries`** every WS message (not only on end).
- Marker time = **`vocal_start_t` / `vocal_end_t`** (when singing actually started/stopped), not model confirm lag.
- Client: `_ingestPhraseBoundary` + Chart.js plugin for vertical markers.

### Demo-focused UI cuts

| Cut | Reason |
|-----|--------|
| **No live transition label/card** | Transition recall OK but precision poor; steals attention from pitch demo |
| **Glissando demoted** | TCN gliss FPs on steady singing hurt coaching tone |
| **Phrase replay modal** | Per-line pitch / breath / loudness; vibrato tab only if vibrato was scored |

**Live demo beats (pick 2–3):**

1. Steady note → tuner + green pitch stability.  
2. Line with vibrato → vibrato chart + gesture overlay.  
3. Two lyric lines with a **real breath** between → chart markers + phrase review card.  
4. End session → session donut + phrase cards.

---

## 3:50 — Innovations beyond NanoPitch (50 s)

> **Say:** “NanoPitch is the sensor; IRIS is the interpretation layer.”

1. **Decoupled decode and coaching** — Viterbi stays WASM-standard; gesture/breath only affect labels and scores, not f0 decode. Prevents feedback loops when classifiers err.

2. **Dual f0 tracks** — `f0_arr` (strict, for chart) vs `f0_phrase` (lenient, for line tracking). Same posteriorgram, different confidence gates.

3. **GestureTCN on trajectory features** — 8-D features from **pre-Viterbi** f0/probabilities; TCN is causal and tiny vs retraining the GRU.

4. **Heuristic + neural merge** — Vibrato band-pass and jump entropy catch what TCN misses; merge thresholds tuned for **precision** (`eval_merge.py`).

5. **Phrase-aware product** — BreathCNN + rule tracker + hop/breath bridge → per-line composite score and chart boundaries (not just a scrolling f0 plot).

6. **Session continuity** — Live `phrase_events` feed `/analyze` so offline scoring uses **same boundaries** as the chart (no re-segmentation surprise).

7. **Classical + learned breath** — CPP/spectral tilt for **support quality**; BreathCNN for **when** a line ended (inhale between lyrics).

---

## 4:40 — Failure modes & further work (20 s)

> **Say:** “These are the honest limits we’d tackle next.”

| Area | Failure mode | Mitigation today | Next step |
|------|--------------|------------------|-----------|
| **Gesture** | Transition/gliss false positives on steady notes | 3-class head, merge thresholds, live transition off | Retrain or calibrate merge on target repertoire; re-enable transition UI when precision ↑ |
| **Phrasing** | Missed breaths at fast tempo / false ends on silence | Breath peak + hop bridge; conservative cooldown | More breath labels; singer-specific threshold calibration |
| **Pitch** | Low conf zeros chart while singing softly | Phrase track uses lower conf | User-adjustable conf floor; alternate checkpoint only if it beats production on *your* mic |
| **Room noise** | dBFS gate helps but not perfect | −50 dBFS + `vocal_active` | Noise gate or learned VAD-only mode |
| **Latency** | CPP/vibrato on slower cadence | Decimated DSP | GPU batching or move BreathCNN to ONNX in browser |
| **Eval gap** | Live has no GT; offline F1 ≠ “feels right” | Phrase scores + user testing | Recorded demo set with human phrase boundaries |
| **Multitask** | NanoPitch+ gesture head underperforms TCN | Kept for research | Don’t ship joint head until it beats TCN without hurting RPA |

**Closing line:**

> “IRIS doesn’t replace NanoPitch — it **consumes** it: one streaming pitch track, three small specialists, and a UI that only speaks when you’re actually singing.”

---

## Appendix — Quick reference

### Start server

```bash
pip3 install -r requirements.txt
python3 app.py
# Hard-refresh browser after code changes
```

### Train path (minimal)

```bash
python3 data/vocalset_download.py
python3 data/vocalset_download.py --extract-minimal
python3 data/vocalset_preprocess.py
python3 model/train_breath.py --data-dir data/vocalset/processed
python3 model/train_gesture.py --n-classes 3 --output-dir runs/gesture_tcn_3class
```

### Key files

| File | Responsibility |
|------|----------------|
| `app.py` | WebSocket, `_extract`, `_track_phrases`, gating, merge |
| `index.html` | Dashboard, chart plugins, coaching, analyze export |
| `features/breath/phrase.py` | PhraseTracker + live line scores |
| `features/pitch/gesture.py` | Heuristics, merge, live classifiers |
| `features/pitch/accuracy.py` | Pitch/bias/instability/slide scoring |
| `model/gesture_tcn.py` | Production gesture net |
| `model/breath_cnn.py` | Phrase breath detector |

### Checkpoint paths (internal)

| Role | Default path |
|------|----------------|
| Pitch (production) | `../NanoPitchFork/training/runs/exp6/checkpoints/best.pth` |
| Pitch (high-RPA, not live) | `.../exp4-cosine-logits/checkpoints/best.pth` |
| Gesture | `runs/gesture_tcn_3class/checkpoints/best.pth` |
| Breath | `runs/breath_cnn/checkpoints/best.pth` |

Eval JSON: `NanoPitchFork/training/results_exp6.json`, `results_exp4.json`.

### Numbers to cite (offline)

- Production NanoPitch RPA (exp6): ~**83–86%** at −5 dB, ~**90%** clean (NanoPitchFork synthetic test set, realtime Viterbi)
- Conservative variant RPA (exp4): ~**91–96%** clean but VDR ~**57–67%** (same harness)
- GestureTCN macro F1 (3-class): ~**51%** (VocalSet val)
- Steady / transition recall: ~**78% / 60%** (same eval)

---

*Ilysia Krzywonos & Chris Zhao — IRIS AI Vocal Coach*
