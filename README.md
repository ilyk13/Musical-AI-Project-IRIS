# IRIS — AI Vocal Coach
**Ilysia Krzywonos & Chris Zhao**

Real-time singing feedback from the browser mic — pitch, gesture (vibrato/glissando/transition), breathiness, and dynamics with sub-100 ms latency.

---

## Current progress

### Working
- **NanoPitch** — streaming GRU pitch tracker (10 ms frames, Viterbi decode, live tuner + pitch chart)
- **Live dashboard** — pitch, vibrato wave, breathiness (CPP), dynamics (pp→ff), coaching bar
- **NanoPitch+ (VocalSet fine-tune)** — gesture-aware pitch decoding in the live pipeline
- **Gesture pill** — steady / vibrato / glissando / transition (model + heuristics; vibrato aligned with chart)

### Trained, still fine-tuning
- Multi-task **NanoPitch+** heads: gesture (~28% val acc), register, dynamics — not all in UI yet
- **Gesture-aware Viterbi** — wider pitch paths on non-steady frames; ET scoring on steady notes only
- Hybrid design: learned gesture head + signal-processing fallbacks where the model is weak

### Next: phrasing awareness
- **BreathCNN** + phrase boundaries → segment performances, not just frames
- Per-phrase feedback: drift, vibrato quality, dynamics arc
- Richer coaching: evaluation after this phrase… instead of only live tuner/CPP hints

---

## Quick Start

```bash
pip3 install -r requirements.txt
python3 app.py
```

Open **http://localhost:8000**, click **Start Listening**, and sing.

**Optional:** use a fine-tuned NanoPitch+ checkpoint (gesture head + VocalSet fine-tune):

```bash
export NANOPITCH_PLUS_CHECKPOINT=runs/vocalset_plus/checkpoints/best.pth
python3 app.py
```

The app auto-discovers `runs/vocalset_plus/checkpoints/best.pth` if present. Override the base pitch checkpoint with `NANOPITCH_CHECKPOINT`.

---

## Project Overview

IRIS listens to a singer in real time and provides immediate, data-driven feedback across five dimensions:

| Dimension | What is measured | How |
|---|---|---|
| **Pitch** | f0, cent deviation from equal temperament | NanoPitch / NanoPitch+ GRU + streaming Viterbi |
| **Gesture** | steady / vibrato / glissando / transition | f0 heuristics + vibrato deviation overlay + optional NanoPitch+ head |
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
│  │    → vibrato deviation overlay → gesture pill label  │   │
│  │    → gesture-aware streaming Viterbi → f0 Hz         │   │
│  │    → central pitch (median, 3 s)                     │   │
│  │    → ET deviation (tuner needle)                       │   │
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
│  • Tuner needle + note + gesture pill │
│  • CPP / dynamics / VAD stats         │
│  • Throttled coaching feedback bar    │
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
│   ├── train_breath.py           # BreathCNN trainer
│   └── breath_cnn.py             # Waveform breath-event detector
│
├── features/
│   ├── pitch/
│   │   ├── nanopitch.py          # NanoPitchExtractor — checkpoint loading
│   │   ├── central_pitch.py      # Median-smoothed f0 reference line
│   │   ├── gesture.py            # Live gesture heuristics, model merge, vibrato overlay
│   │   └── pitch_drift.py        # Per-note drift scoring
│   ├── vibrato/
│   │   ├── bandpass.py           # Isolate vibrato oscillation via band-pass filter
│   │   └── parameters.py         # Rate, depth, consistency scoring
│   ├── breath/
│   │   ├── cpp.py                # Cepstral Peak Prominence (breathiness proxy)
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
│   └── vocalset_plus/checkpoints/  # NanoPitch+ fine-tune output
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
| **Tuner** | Needle ±50 ¢ from nearest ET note; updates in real time |
| **Gesture pill** | steady / vibrato / glissando / transition; vibrato uses the same deviation signal as the chart (~450 ms window) |
| **Feedback bar** | Coaching text; updates when advice **category** changes, at most every ~22 s; ignores brief silence gaps |

---

## Feature Extraction Details

### Pitch (f0)

The live app runs **streaming NanoPitch or NanoPitch+** inference: each 10 ms mel frame passes through the GRU stack with persistent hidden state. The pitch posteriorgram is decoded with **gesture-aware streaming Viterbi** (`viterbi_stream_gesture`), which widens allowed pitch transitions on vibrato/glissando/transition frames.

**Central pitch** is the median-filtered f0 over a 3-second rolling buffer — a slow reference line on the chart. The tuner needle shows instantaneous deviation from the nearest equal-temperament note (±50 cents).

**Gesture classification** uses f0-track heuristics on the **raw pre-Viterbi** pitch track (preserves vibrato modulation). When NanoPitch+ is loaded, model gesture logits are merged with heuristics (non-steady heuristic detections win over a steady model prediction). **Vibrato** is then reinforced from the same central-relative deviation used for the vibrato chart (`overlay_vibrato_from_deviation`), so the gesture pill tracks oscillation you see on the chart without needing long sustained vibrato.

### Vibrato

Deviation is computed in **cents relative to the rolling central pitch** (not A440), then band-pass filtered (4th-order Butterworth, **3–9 Hz**) to isolate oscillation. A Savitzky-Golay smoother cleans the signal before plotting. The chart refreshes every ~350 ms (`VIB_EVERY=15` chunks). Requires a minimum of 120 voiced frames in the 3 s rolling buffer (~1.2 s of singing) before the vibrato chart activates.

The vibrato chart shows modulation strength in cents; it does **not** by itself measure pitch accuracy against ground truth. Glissando and transition gestures still come from f0 slope/jump heuristics on the raw pre-Viterbi track.

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

### Offline evaluation (with ground truth)

Use these to measure model quality against annotated or synthetic reference f0:

| Metric | Definition |
|---|---|
| **RPA** (Raw Pitch Accuracy) | % of co-voiced frames where \|error\| < **50 cents** |
| **VDR** (Voicing Detection Rate) | % of truly voiced GT frames detected as voiced |
| **RCA** (Raw Chroma Accuracy) | Like RPA but octave-invariant |
| **Median cents** | Typical pitch error magnitude (lower is better) |
| **Gesture accuracy** | % of frames with correct gesture label (NanoPitch+ only) |

**Evaluate a NanoPitch+ checkpoint on VocalSet val** (32-clip subset, printed every 5 training epochs):

```bash
python3 model/train_multitask.py \
  --data-dir data/vocalset/processed \
  --resume runs/vocalset_plus/checkpoints/best.pth \
  --epochs 0
```

**Base NanoPitch on synthetic noisy test set** (if `NanoPitchFork/data/test.npz` is available):

```bash
cd ../NanoPitchFork/training
python3 evaluate.py \
  --checkpoint ../../IRIS/runs/vocalset_plus/checkpoints/best.pth \
  --data-dir ../data
```

Reference numbers (NanoPitch exp6, synthetic test): ~**83–86% realtime RPA** in clean conditions. VocalSet val with NanoPitch+ fine-tune is typically ~**78–86% RPA** depending on val split and training length.

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
transition neighbourhood on vibrato/glissando/transition frames before scoring
accuracy — steady frames keep the standard ±12-bin constraint.

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

# Step 5 — fine-tune NanoPitch+ (recommended starting point)
python3 model/train_multitask.py \
  --data-dir data/vocalset/processed \
  --resume ../NanoPitchFork/training/runs/exp6/checkpoints/best.pth \
  --output-dir runs/vocalset_plus \
  --epochs 40 \
  --w-gesture 2.0

# Eval-only (no training) — prints gesture acc + RPA on val subset
python3 model/train_multitask.py \
  --data-dir data/vocalset/processed \
  --resume runs/vocalset_plus/checkpoints/best.pth \
  --epochs 0

# Step 6 — live app (auto-finds runs/vocalset_plus/checkpoints/best.pth)
export NANOPITCH_PLUS_CHECKPOINT=runs/vocalset_plus/checkpoints/best.pth
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

- `data/vocalset_labels.py` — label vocabularies and heuristics
- `data/vocalset_preprocess.py` — WAV + CSV → `train.npz`, `val.npz`, `breath.npz`
- `model/nanopitch.py` — `NanoPitchPlus`, `viterbi_decode_gesture`
- `model/breath_cnn.py` — waveform breath detector
- `model/train_multitask.py` — multi-task GRU trainer
- `model/train_breath.py` — BreathCNN trainer

These heads allow the system to score pitch accuracy only on steady frames, skip
scoring during transitions, and segment the performance into musical phrases for
per-phrase feedback.

### Gesture-aware scoring (live app)

The live app combines **f0 heuristics** and an optional **trained gesture head**:

| Component | Role |
|---|---|
| **Heuristic gesture** | 3–9 Hz vibrato band-pass, glissando slope, transition jumps on raw pre-Viterbi f0 |
| **Vibrato overlay** | Promotes steady → vibrato when band-pass deviation matches the chart (≥6 ¢ peak or ≥4 ¢ std in a short window) |
| **Gesture display** | `smooth_gesture_label` uses recent deviation frames so the pill responds in ~450 ms |
| **Model gesture head** | NanoPitch+ logits merged with heuristics (heuristic non-steady wins over model steady) |
| **Gesture Viterbi** | Wider pitch transitions on vibrato (30 bins) / glissando (24) / transition (36) |
| **Feedback bar** | Category-based coaching; throttled ~22 s |

Set `NANOPITCH_PLUS_CHECKPOINT=runs/vocalset_plus/checkpoints/best.pth` to load the learned gesture head. Heuristics and the vibrato deviation overlay remain active when the model under-predicts non-steady gestures.
