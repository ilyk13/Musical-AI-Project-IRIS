# IRIS — AI Vocal Coach
**Ilysia Krzywonos & Chris Zhao**

Real-time singing feedback from the browser mic — pitch, gesture (vibrato/glissando/transition), breathiness, and dynamics with sub-100 ms latency.

---

## Current progress

### Working
- **NanoPitch** — streaming GRU pitch tracker (10 ms frames, Viterbi decode, live tuner + pitch chart)
- **Live dashboard** — pitch, vibrato wave, breathiness (CPP), dynamics (pp→ff), phrase panel, coaching bar
- **NanoPitch+ (VocalSet fine-tune)** — optional gesture head merged with f0 heuristics in the live pipeline
- **Gesture detection** — steady / vibrato / glissando / transition; vibrato aligned with chart; glissando from sustained f0 ramps on Viterbi track
- **Live phrasing** — BreathCNN + voicing cues; chart markers on phrase start/end; per-phrase score + coaching card
- **Post-session analysis** — `/analyze` segments recording into phrases with pitch / vibrato / breath bullets

### Trained, still fine-tuning
- Multi-task **NanoPitch+** heads (v4 best: ~30% macro gesture F1, ~82% RPA) — register/dynamics heads not in UI yet
- **Hybrid gesture** — heuristics + deviation overlay win over weak model predictions; model helps on steady frames
- Glissando recall improved in v4 labels/training; live glissando uses strict extended-slide heuristics (not model-only)

### Next
- Register / dynamics heads in the live UI
- Richer per-phrase replay in the session review modal
- Further gesture model iterations (transition recall, glissando precision)

---

## Quick Start

```bash
pip3 install -r requirements.txt
python3 app.py
```

Open **http://localhost:8000**, click **Start Listening**, and sing.

**Optional:** use a fine-tuned NanoPitch+ checkpoint (gesture head + VocalSet fine-tune):

```bash
export NANOPITCH_USE_PLUS=1
export NANOPITCH_PLUS_CHECKPOINT=runs/vocalset_plus_v4/checkpoints/best.pth
python3 app.py
```

The app auto-discovers `runs/vocalset_plus_v3/checkpoints/best.pth` or `runs/vocalset_plus/checkpoints/best.pth` if present. For **v4**, set `NANOPITCH_PLUS_CHECKPOINT` explicitly. Override the base pitch checkpoint with `NANOPITCH_CHECKPOINT`. BreathCNN auto-loads from `runs/breath_cnn/checkpoints/best.pth` when present (`BREATH_CHECKPOINT` to override).

---

## Project Overview

IRIS listens to a singer in real time and provides immediate, data-driven feedback across five dimensions:

| Dimension | What is measured | How |
|---|---|---|
| **Pitch** | f0, cent deviation from equal temperament | NanoPitch / NanoPitch+ GRU + streaming Viterbi |
| **Gesture** | steady / vibrato / glissando / transition | f0 heuristics + vibrato overlay + optional NanoPitch+ merge |
| **Phrasing** | Phrase start/end, composite line score | BreathCNN + voicing tracker (`features/breath/phrase.py`) |
| **Vibrato** | Oscillation in cents (display) | 3–9 Hz band-pass on f0 track |
| **Breathiness** | Cepstral Peak Prominence (CPP) | Analytical DSP |
| **Dynamics** | RMS loudness, dynamic level (pp → ff) | Analytical DSP |
| **Voice Activity** | Voiced / unvoiced frame detection | Model VAD head |

---

## Architecture

```
Browser microphone  (Web Audio API → 16 kHz, 10 ms frames)
        │
        │  Float32 PCM over WebSocket
        ▼
┌─────────────────────────────────────────────────────────────┐
│  Python server  (FastAPI + uvicorn)                         │
│                                                             │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  NanoPitch / NanoPitch+  (streaming GRU)             │   │
│  │    → 360-bin pitch posteriorgram + VAD               │   │
│  │    → gesture logits (NanoPitch+ only)                │   │
│  │    → gesture from raw f0 heuristics (+ model merge)  │   │
│  │    → vibrato deviation overlay → display label       │   │
│  │    → standard streaming Viterbi → f0 Hz             │   │
│  │    → central pitch (median, 3 s)                     │   │
│  │    → ET deviation (tuner needle)                       │   │
│  ├──────────────────────────────────────────────────────┤   │
│  │  BreathCNN  ·  breath prob → phrase boundaries       │   │
│  │  PhraseTracker  ·  per-phrase pitch/vibrato/breath   │   │
│  ├──────────────────────────────────────────────────────┤   │
│  │  Dynamics  ·  RMS → dBFS → pp/p/mp/mf/f/ff          │   │
│  ├──────────────────────────────────────────────────────┤   │
│  │  Breathiness  ·  CPP + spectral tilt (every 15 ch.)  │   │
│  ├──────────────────────────────────────────────────────┤   │
│  │  Vibrato display  ·  central-relative cents + 3–9 Hz BP │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                             │
│  JSON frames over WebSocket  (~10 updates/second)           │
└─────────────────────────────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────┐
│  Browser dashboard  (Chart.js)        │
│  • Scrolling pitch chart (30 s)       │
│  • Vibrato deviation wave (6 s)       │
│  • Tuner needle + note + gesture readout                │
│  • Phrase panel + chart boundary markers                │
│  • CPP / dynamics / coaching cards                    │
└───────────────────────────────────────┘
```

---

## Repository Layout

```
IRIS/
├── app.py                        # FastAPI web server — entry point
├── index.html                    # Browser dashboard (Chart.js, Web Audio API)
├── requirements.txt
│
├── model/
│   ├── nanopitch.py              # NanoPitch + NanoPitchPlus, Viterbi decoders
│   ├── train.py                  # Base NanoPitch trainer (NanoPitch-PreExtract)
│   ├── train_multitask.py        # NanoPitch+ multi-task trainer (VocalSet)
│   ├── eval_multitask.py         # Offline checkpoint eval (RPA + gesture F1)
│   ├── train_breath.py           # BreathCNN trainer
│   └── breath_cnn.py             # Waveform breath-event detector
│
├── features/
│   ├── pitch/
│   │   ├── nanopitch.py          # NanoPitchExtractor — checkpoint loading
│   │   ├── central_pitch.py      # Median-smoothed f0 reference line
│   │   ├── gesture.py            # Live gesture heuristics, merge, overlay, glissando
│   │   └── pitch_drift.py        # Per-note drift scoring
│   ├── vibrato/
│   │   ├── bandpass.py           # Isolate vibrato oscillation via band-pass filter
│   │   └── parameters.py         # Rate, depth, consistency scoring
│   ├── breath/
│   │   ├── cpp.py                # Cepstral Peak Prominence (breathiness proxy)
│   │   ├── phrase.py             # Live phrase boundaries + per-phrase scoring
│   │   └── spectral_tilt.py      # Spectral tilt slope + H1–H2
│   └── dynamics/
│       ├── rms.py                # RMS → dBFS
│       └── dynamic_class.py      # Map dBFS to pp/p/mp/mf/f/ff labels
│
├── data/
│   ├── download.py               # NanoPitch-PreExtract (noise augmentation)
│   ├── vocalset_download.py      # Annotated-VocalSet + selective VocalSet extract
│   ├── vocalset_preprocess.py    # WAV + CSV → train.npz / val.npz / breath.npz
│   ├── vocalset_labels.py        # Gesture/register/dynamics vocab + heuristics
│   ├── clean.npz                 # Pre-extracted clean singing mel + f0 + VAD
│   ├── noise.npz                 # Pre-extracted noise mel (for augmentation)
│   └── vocalset/                 # Downloaded annotations + audio (gitignored)
│
├── runs/
│   ├── exp1/checkpoints/         # Base NanoPitch checkpoint (optional)
│   └── vocalset_plus_v4/         # Latest NanoPitch+ fine-tune (gesture labels v4.2)
│
└── utils/
    └── audio_io.py               # Audio load/save helpers (librosa + soundfile)
```

---

## The NanoPitch Model

NanoPitch is a lightweight GRU-based neural network for real-time pitch tracking, adapted from [Smule Labs' NanoPitch](https://github.com/smulelabs/nanopitch). Total parameters: **~333K** — small enough to run on a laptop CPU.

### Architecture

```
40 mel bands (input, 10 ms hop, 16 kHz)
    │
    ▼
Conv1d(40→64, k=3, causal) + tanh     ← local pattern extraction
Conv1d(64→96, k=3, causal) + tanh     ← feature combination
    │
    ▼
GRU(96)  →  GRU(96)  →  GRU(96)      ← temporal modeling
    │
    ▼
Concat [conv_out, gru1, gru2, gru3]   ← 384-dim skip connection
    │
    ├──→ Linear(384→1)   + sigmoid    → VAD probability
    └──→ Linear(384→360) + sigmoid    → pitch posteriorgram
```

**NanoPitchPlus** adds three heads on the same 384-d concat:

```
    ├──→ Linear(384→4)   → gesture  (steady / vibrato / glissando / transition)
    ├──→ Linear(384→4)   → register (chest / mixed / head / falsetto)
    └──→ Linear(384→7)   → dynamics (pp … ff)
```

The 360 output bins cover B0–B6 (31.7–2006 Hz) at 20 cents/bin resolution. At inference, a **streaming Viterbi decoder** tracks the pitch posteriorgram frame-by-frame with GRU state carried across the session. **NanoPitchPlus** adds gesture, register, and dynamics heads on the same backbone.

Causal convolutions (left-padding only) ensure the model never looks at future frames, making it suitable for real-time streaming.

### Training

Training uses pre-extracted features from [smulelabs/NanoPitch-PreExtract](https://huggingface.co/datasets/smulelabs/NanoPitch-PreExtract):

```bash
python3 data/download.py                               # download clean.npz, noise.npz, test.npz
python3 model/train.py --data-dir data/ --output-dir runs/exp1
tensorboard --logdir runs/exp1/tb                      # monitor training
```

**Training pipeline:**
1. Load 40-band log-mel spectrograms + RMVPE ground-truth f0 + VAD labels
2. Mix clean vocal mel with random noise mel at a random SNR (−5 to +20 dB) per batch
3. Build soft Gaussian pitch targets (σ = 0.8 bins ≈ 16 cents) around ground-truth f0
4. BCE loss on VAD + voiced-weighted BCE loss on pitch posteriorgram
5. AdamW optimizer + OneCycleLR schedule (per-batch stepping)

A trained checkpoint lives at `runs/exp1/checkpoints/best.pth` (or use a NanoPitchFork sibling checkpoint — see `app.py` search order).

The live app loads checkpoints automatically at startup. Set `NANOPITCH_CHECKPOINT` or `NANOPITCH_PLUS_CHECKPOINT` to override.

---

## Live dashboard

| UI element | Behaviour |
|---|---|
| **Pitch chart** | 30 s scrolling f0 + central line; brief unvoiced gaps hold the last pitch (~350 ms), then drop to the log-scale floor |
| **Vibrato chart** | 6 s band-pass oscillation in cents (qualitative — not ground-truth accuracy) |
| **Tuner** | Needle ±50 ¢ from nearest ET note; “vibrato center” / “sliding” sub-labels when active |
| **Phrase panel** | Listening… → In phrase → Between phrases; shows last line tier + score (e.g. `Great · 84`) |
| **Chart markers** | Vertical “phrase” dividers on start/end events |
| **Coaching cards** | Live pitch / vibrato / glissando / breath tips; phrase review card pinned after each line |
| **Card throttle** | Refreshes when advice category changes or every ~14 s; gesture changes bypass throttle |

---

## Live phrase tracking & scoring

Phrases are detected in real time by **BreathCNN** (480-sample waveform windows) plus **voicing**:

1. **Pending** — voiced audio detected; waits ~350 ms of continuous voicing before confirming.
2. **In phrase** — confirmed; pitch/gesture/breath metrics accumulate on steady frames.
3. **End** — ~280 ms silence or sustained breath probability ends the line; feedback is emitted.

Each completed phrase returns:

| Field | Meaning |
|---|---|
| `phrase_score` | Composite 0–100 (pitch + vibrato + breath, gentle curves) |
| `score_label` | Tier: Excellent / Great / Good / Solid / Keep refining |
| `score_breakdown` | e.g. `Pitch 79 · Vibrato 92 · Breath 81` |
| `headline` / `detail` | Coaching title + tip shown in the review card |

Pitch scoring uses **central-relative drift** on steady frames only (same idea as post-session ET scoring). Vibrato phrases still receive a pitch sub-score, but pitch tips are suppressed when vibrato dominates the line.

Boundary events are sent as `phrase_events` over the WebSocket so chart markers are not dropped when multiple frames land in one audio chunk.

---

## Feature Extraction Details

### Pitch (f0)

The live app runs **streaming NanoPitch or NanoPitch+** inference: each 10 ms mel frame passes through the GRU stack with persistent hidden state. The pitch posteriorgram is decoded with **standard streaming Viterbi** (`viterbi_stream`, ±12 bins). Gesture labels drive UI and scoring only — they do not widen the Viterbi path in the current live app.

**Central pitch** is the median-filtered f0 over a 3-second rolling buffer — a slow reference line on the chart. The tuner needle shows instantaneous deviation from the nearest equal-temperament note (±50 cents).

**Gesture classification** uses f0-track heuristics on the **raw pre-Viterbi** argmax track (preserves vibrato modulation). When NanoPitch+ is loaded, model gesture logits are merged with heuristics:

- Heuristic **glissando / transition / vibrato** are preserved unless the model strongly predicts transition.
- The model may promote **steady → glissando** when confident; it does **not** promote steady → vibrato alone (precision guard).
- **Vibrato overlay** promotes steady/glissando → vibrato when band-pass deviation matches the chart.
- **Glissando** requires a sustained directional slide (~55¢+ over ~350 ms) on the smoother Viterbi f0 track; brief pitch drift does not trigger it.
- **Display label** (`smooth_gesture_label`) favours vibrato when oscillation is clear; glissando when slide frames dominate.

ET pitch accuracy coaching applies on **steady** frames only. Glissando and transition frames are excluded from phrase pitch scoring and flat/sharp cards.

### Vibrato

Deviation is computed in **cents relative to the rolling central pitch** (not A440), then band-pass filtered (4th-order Butterworth, **3–9 Hz**) to isolate oscillation. A Savitzky-Golay smoother cleans the signal before plotting. The chart refreshes every ~350 ms (`VIB_EVERY=15` chunks). Requires a minimum of 120 voiced frames in the 3 s rolling buffer (~1.2 s of singing) before the vibrato chart activates.

The vibrato chart shows modulation strength in cents; it does **not** by itself measure pitch accuracy against ground truth. **Glissando** is detected from sustained monotonic f0 motion and net displacement on the Viterbi track, not from the vibrato band-pass alone.

Ideal professional vibrato: roughly 5–7 Hz rate, 25–75 cents depth.

### Breathiness — CPP

Cepstral Peak Prominence measures how dominant the periodic (harmonic) component of the voice is relative to the cepstral noise floor. A clear, well-supported voice has high CPP (>15 dB). A breathy or airy voice has lower CPP because incomplete glottal closure weakens the harmonics.

Spectral tilt (linear regression slope on the log power spectrum in dB/octave) provides a secondary breathiness measure. A steeper negative slope indicates more energy concentrated in low frequencies, characteristic of breathy phonation.

### Dynamics

Per-frame RMS energy is converted to dBFS. Thresholds (provisional, calibrated to typical recording levels) map dBFS values to standard dynamic markings:

| Level | Threshold |
|---|---|
| pp (pianissimo) | −55 dBFS |
| p (piano) | −45 dBFS |
| mp (mezzopiano) | −38 dBFS |
| mf (mezzoforte) | −30 dBFS |
| f (forte) | −22 dBFS |
| ff (fortissimo) | −14 dBFS |

---

## Evaluation Metrics

### Live dashboard (no ground truth)

These measure singing relative to **equal temperament**, not annotated f0:

| Metric | Definition |
|---|---|
| **Tuner deviation** | Instantaneous cents above/below nearest ET note |
| **Vibrato chart** | Isolated pitch oscillation in cents (qualitative) |
| **Phrase score** | Composite line score with tier label after each phrase |

### Offline evaluation (with ground truth)

Use these to measure model quality against annotated or synthetic reference f0:

| Metric | Definition |
|---|---|
| **RPA** (Raw Pitch Accuracy) | % of co-voiced frames where \|error\| < **50 cents** |
| **VDR** (Voicing Detection Rate) | % of truly voiced GT frames detected as voiced |
| **RCA** (Raw Chroma Accuracy) | Like RPA but octave-invariant |
| **Median cents** | Typical pitch error magnitude (lower is better) |
| **Gesture accuracy** | % of frames with correct gesture label (NanoPitch+ only) |

**Evaluate a NanoPitch+ checkpoint on VocalSet val:**

```bash
python3 model/eval_multitask.py --checkpoint runs/vocalset_plus_v4/checkpoints/best.pth
```

Or via the trainer (32-clip val subset, printed every 5 training epochs):

```bash
python3 model/train_multitask.py \
  --data-dir data/vocalset/processed \
  --resume runs/vocalset_plus_v4/checkpoints/best.pth \
  --epochs 0
```

**Base NanoPitch on synthetic noisy test set** (if `NanoPitchFork/data/test.npz` is available):

```bash
cd ../NanoPitchFork/training
python3 evaluate.py \
  --checkpoint ../../IRIS/runs/vocalset_plus/checkpoints/best.pth \
  --data-dir ../data
```

Reference numbers: base NanoPitch exp6 ~**83–86% RPA** on synthetic test. VocalSet+ **v4** best ~**82% RPA**, ~**30% macro gesture F1** (steady precision high; glissando/transition still challenging).

---

## Dependencies

| Package | Use |
|---|---|
| `torch` | NanoPitch model, training |
| `librosa` | Audio resampling, mel spectrograms |
| `scipy` | Band-pass filter, median filter, Savitzky-Golay smoother |
| `numpy` | All numerical computation |
| `fastapi` + `uvicorn` | Web server and WebSocket |
| `soundfile` | Audio file I/O |
| `huggingface_hub` | Dataset download |
| `tensorboard` | Training monitoring |
| `tqdm` | Training progress bars |

---

## Phase 2 — NanoPitch+ (VocalSet)

Phase 2 extends the base NanoPitch GRU with multi-task heads trained on
**VocalSet** audio and **Annotated-VocalSet** per-frame labels (F0, amplitude,
onset/offset, transition). This replaces the original GTSinger plan in the
project spec while keeping the same architecture goals.

| Head | Labels | Source |
|---|---|---|
| Pitch + VAD | f0, voiced/unvoiced | Annotated-VocalSet F0 contour |
| Gesture | steady / vibrato / glissando / transition | Transition column + f0 shape + technique |
| Register | chest / mixed / head / falsetto | VocalSet technique folders (weak) |
| Dynamics | pp … ff | Annotated-VocalSet amplitude |
| BreathCNN | breath event | Heuristic gaps between phrases (raw waveform) |

**Gesture-aware Viterbi** (`viterbi_decode_gesture`) widens the allowed pitch
transition neighbourhood on vibrato/glissando/transition frames — used in
training/eval; the live app uses standard Viterbi for decode stability.

### Minimal download (recommended)

You need **annotations for every clip** (~411 MB) and **audio only for the clips you train on** (~1–2 GB extracted).

| Step | What | Size |
|---|---|---|
| 1 | Annotated-VocalSet (labels) | ~411 MB |
| 2 | VocalSet1-2.zip (download once, extract subset) | **~5.6 GB** zip → ~1–2 GB extracted |
| 3 | NanoPitch noise (augmentation) | ~50 MB |

**Default minimal techniques** (gesture + register coverage): `straight`, `vibrato`, `belt`, `slow_piano`, `messa`

```bash
cd /path/to/IRIS

# Step 1 — annotations (~411 MB, automatic)
python3 data/vocalset_download.py

# Step 2 — download VocalSet1-2.zip in your browser (one-time, ~5.6 GB):
#   https://zenodo.org/records/1442513/files/VocalSet1-2.zip
# Save to: data/vocalset/VocalSet1-2.zip

# Step 2b — extract only 5 technique folders (~1–2 GB on disk)
python3 data/vocalset_download.py --extract-minimal

# Step 3 — noise for SNR augmentation during training
python3 data/download.py

# Step 4 — build .npz tensors (defaults to minimal techniques)
python3 data/vocalset_preprocess.py

# Smoke test first (optional, ~100 files):
python3 data/vocalset_preprocess.py --max-files 100

# Step 5 — fine-tune NanoPitch+ (v4 recommended: fixed glissando labels)
python3 model/train_multitask.py \
  --data-dir data/vocalset/processed \
  --resume ../NanoPitchFork/training/runs/exp6/checkpoints/best.pth \
  --output-dir runs/vocalset_plus_v4 \
  --epochs 40 \
  --w-gesture 2.0

# Eval-only
python3 model/eval_multitask.py --checkpoint runs/vocalset_plus_v4/checkpoints/best.pth

# Step 5b — BreathCNN for live phrase boundaries (one-time)
python3 model/train_breath.py --data-dir data/vocalset/processed --output-dir runs/breath_cnn

# Step 6 — live app
export NANOPITCH_USE_PLUS=1
export NANOPITCH_PLUS_CHECKPOINT=runs/vocalset_plus_v4/checkpoints/best.pth
python3 app.py
```

### Full setup (all techniques)

```bash
python3 data/vocalset_download.py --download-audio
python3 data/download.py
python3 data/vocalset_preprocess.py --techniques straight,vibrato,belt,slow_piano,messa,forte,breathy,trill
python3 model/train_multitask.py --data-dir data/vocalset/processed --resume runs/exp1/checkpoints/best.pth
python3 model/train_breath.py --data-dir data/vocalset/processed
```

Key files:

- `features/breath/phrase.py` — live phrase tracker + per-phrase scoring
- `data/vocalset_labels.py` — label vocabularies and heuristics (shared with `gesture.py`)
- `data/vocalset_preprocess.py` — WAV + CSV → `train.npz`, `val.npz`, `breath.npz`
- `model/nanopitch.py` — `NanoPitchPlus`, `viterbi_decode_gesture`
- `model/breath_cnn.py` — waveform breath detector
- `model/train_multitask.py` — multi-task GRU trainer
- `model/train_breath.py` — BreathCNN trainer

These heads allow the system to score pitch accuracy only on steady frames, skip
scoring during transitions, and segment the performance into musical phrases for
per-phrase feedback.

### Gesture-aware scoring (live app)

The live app combines **f0 heuristics**, **Viterbi-track glissando detection**, and an optional **trained gesture head**:

| Component | Role |
|---|---|
| **Heuristic gesture** | Vibrato band-pass, glissando slope + net ramp, transition jumps on raw f0 |
| **Vibrato overlay** | Promotes steady/glissando → vibrato when band-pass deviation matches the chart |
| **Glissando gate** | Extended slide only (~55¢+ over ~350 ms); collapses short runs |
| **Model merge** | NanoPitch+ logits merged with heuristics; motion classes from heuristics win |
| **Display smoothing** | ~450 ms window; vibrato vs glissando priority from deviation + frame counts |
| **Phrase scoring** | Steady frames only for pitch; composite tier score at phrase end |
| **Coaching cards** | Pitch / vibrato / glissando / breath tips; phrase review card with score badge |

Set `NANOPITCH_USE_PLUS=1` and `NANOPITCH_PLUS_CHECKPOINT=runs/vocalset_plus_v4/checkpoints/best.pth` to load the learned gesture head. Heuristics and the vibrato deviation overlay remain active when the model under-predicts non-steady gestures.

**Offline gesture-aware Viterbi** (`viterbi_decode_gesture` in `model/nanopitch.py`) is used in training/eval tooling; the live app uses standard Viterbi for pitch decode stability.
