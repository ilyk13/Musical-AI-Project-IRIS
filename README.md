# IRIS — AI Vocal Coach
**Ilysia Krzywonos & Chris Zhao**

Real-time singing feedback from the browser mic — pitch, gesture (steady / vibrato / transition), breathiness, and dynamics with sub-100 ms latency.

---

## Current progress

### Production stack
- **Pitch** — NanoPitch **exp6** (`../NanoPitchFork/training/runs/exp6/`) + standard streaming Viterbi (±12 bins)
- **Gesture** — **GestureTCN 3-class** (`runs/gesture_tcn_3class/checkpoints/best.pth`) merged with f0 heuristics; glissando demoted for coaching
- **Phrasing** — BreathCNN + `PhraseTracker`; chart markers on phrase start/end; per-phrase coaching card at line end
- **Post-session analysis** — `/analyze` from live phrase events; duration-weighted session scores

### Working
- Live dashboard — pitch chart, vibrato wave, CPP, dynamics, phrase panel, coaching cards
- Phrase replay modal — pitch / breath / loudness; vibrato chart only when vibrato was scored
- GestureTCN training — `model/train_gesture.py --n-classes 3`

### Gesture model comparison (VocalSet val, offline)

We tried a **multi-task NanoPitch+** gesture head on the same GRU backbone; production uses **GestureTCN** instead.

| Model | Macro F1 | Steady R | Transition R | Notes |
|---|---|---|---|---|
| NanoPitch+ v4 *(experiment)* | 30% | 45% | 4% | Joint multitask; ~82% RPA on VocalSet |
| GestureTCN 4-class | 37% | 35% | 51% | High gliss false positives on steady |
| **GestureTCN 3-class** *(shipped)* | **51%** | **78%** | **60%** | Default live gesture model |

Legacy multitask code: `model/train_multitask.py`, `model/eval_multitask.py`, checkpoints under `runs/vocalset_plus_*`.

### Next
- Tighten transition precision in live merge (recall is strong; raw TCN precision is low)

---

## Quick Start

```bash
pip3 install -r requirements.txt
python3 app.py
```

Open **http://localhost:8000**, click **Start Listening**, and sing.

**Auto-discovered checkpoints** (first match wins):

| Role | Default path | Override env |
|---|---|---|
| Pitch | `../NanoPitchFork/training/runs/exp6/checkpoints/best.pth` | `NANOPITCH_CHECKPOINT` |
| Gesture | `runs/gesture_tcn_3class/checkpoints/best.pth` | `GESTURE_TCN_CHECKPOINT` |
| Breath | `runs/breath_cnn/checkpoints/best.pth` | `BREATH_CHECKPOINT` |

**Train / eval GestureTCN:**

```bash
python3 model/train_gesture.py --n-classes 3 --output-dir runs/gesture_tcn_3class
python3 model/eval_gesture.py --checkpoint runs/gesture_tcn_3class/checkpoints/best.pth
```

---

## Project Overview

| Dimension | What is measured | How |
|---|---|---|
| **Pitch** | f0, cent deviation from equal temperament | NanoPitch GRU + streaming Viterbi |
| **Gesture** | steady / vibrato / transition | GestureTCN 3-class + f0 heuristics merge |
| **Phrasing** | Phrase start/end, composite line score | BreathCNN + voicing (`features/breath/phrase.py`) |
| **Vibrato** | Oscillation in cents (display) | 3–9 Hz band-pass on f0 track |
| **Breathiness** | Cepstral Peak Prominence (CPP) | Analytical DSP |
| **Dynamics** | RMS loudness, dynamic level (pp → ff) | Analytical DSP |
| **Voice Activity** | Voiced / unvoiced | NanoPitch VAD head |

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
│  │  NanoPitch exp6  (streaming GRU)                      │   │
│  │    → pitch posteriorgram + VAD                       │   │
│  │    → standard streaming Viterbi → f0 Hz               │   │
│  │    → central pitch + ET deviation (tuner)              │   │
│  ├──────────────────────────────────────────────────────┤   │
│  │  GestureTCN 3-class  (f0 features, causal TCN)         │   │
│  │    + f0 heuristics (pre-Viterbi) → merge → labels    │   │
│  ├──────────────────────────────────────────────────────┤   │
│  │  BreathCNN  ·  phrase boundaries + line scores         │   │
│  ├──────────────────────────────────────────────────────┤   │
│  │  DSP: dynamics, CPP, vibrato display band-pass         │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                             │
│  JSON frames over WebSocket  (~10 updates/second)           │
└─────────────────────────────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────┐
│  Browser dashboard  (Chart.js)        │
│  • Pitch chart, vibrato wave, tuner   │
│  • Phrase panel + coaching cards      │
│  • End Session → analysis + export    │
└───────────────────────────────────────┘
```

---

## Repository Layout

```
IRIS/
├── app.py                        # FastAPI server — entry point
├── index.html                    # Browser dashboard
├── DEMO.md                       # Presentation demo script
│
├── model/
│   ├── nanopitch.py              # NanoPitch, Viterbi (Plus types kept for legacy eval)
│   ├── gesture_tcn.py            # Production gesture classifier
│   ├── gesture_features.py       # 8-D f0 features
│   ├── gesture_classes.py        # 3-class remap + logit expand
│   ├── train_gesture.py          # GestureTCN trainer
│   ├── eval_gesture.py           # GestureTCN eval
│   ├── train.py                  # Base NanoPitch trainer
│   ├── train_multitask.py        # Legacy NanoPitch+ trainer (experiment)
│   ├── eval_multitask.py         # Legacy multitask eval
│   ├── train_breath.py           # BreathCNN trainer
│   └── breath_cnn.py
│
├── features/
│   ├── pitch/                    # nanopitch, gesture, accuracy, drift
│   ├── vibrato/
│   ├── breath/
│   └── dynamics/
│
├── data/                         # VocalSet download + preprocess → npz
│
├── runs/
│   ├── gesture_tcn_3class/       # Production gesture
│   ├── breath_cnn/
│   └── vocalset_plus_*/          # Legacy multitask checkpoints (comparison only)
│
└── utils/
```

---

## NanoPitch (pitch tracking)

NanoPitch is a lightweight GRU-based pitch tracker (~333K params), adapted from [Smule Labs' NanoPitch](https://github.com/smulelabs/nanopitch).

```
40 mel bands → causal conv → 3× GRU → 384-d concat
    ├── VAD (sigmoid)
    └── pitch posteriorgram (360 bins, 20¢ resolution)
         → streaming Viterbi (±12 bins) → f0 Hz
```

Production weights: **exp6** (NanoPitchFork). Train base NanoPitch on [NanoPitch-PreExtract](https://huggingface.co/datasets/smulelabs/NanoPitch-PreExtract):

```bash
python3 data/download.py
python3 model/train.py --data-dir data/ --output-dir runs/exp1
```

Set `NANOPITCH_CHECKPOINT` to override the default exp6 path.

---

## GestureTCN (production gesture)

Standalone **causal TCN** (~63K params) on per-frame f0 trajectory features (8-D). Trained on VocalSet `train.npz` / `val.npz` with **3 classes** (glissando labels folded into steady).

Live pipeline: heuristics on raw pre-Viterbi f0 → merge with TCN logits (`TCN_MERGE_KW` in `app.py` for transition precision).

---

## Live dashboard

| UI element | Behaviour |
|---|---|
| **Pitch chart** | 30 s scrolling f0 + central line; ET color coding on steady frames |
| **Vibrato chart** | 6 s band-pass oscillation in cents (display signal, not the vibrato score gate) |
| **Tuner** | ±50 ¢ from nearest ET note |
| **Phrase panel** | In phrase / between phrases; last line score after phrase end |
| **Coaching cards** | Pitch / vibrato / transition tips; phrase review card |

---

## Live phrase tracking & scoring

**BreathCNN** (480-sample windows) + **PhraseTracker** voicing:

1. **Pending** — ~350 ms voiced audio before phrase confirms  
2. **In phrase** — metrics accumulate on steady frames for pitch  
3. **End** — silence or breath ends line; coaching card with composite score  

`phrase_events` over WebSocket keep chart markers aligned with analysis phrase cards.

---

## Post-session analysis (`/analyze`)

Browser POSTs `frames`, `summaries`, `phrase_events`. Server segments phrases from live boundaries, scores each line (pitch on steady frames, vibrato if detected, breath from CPP), then duration-weights session bars and overall donut score.

---

## Feature extraction (summary)

**Pitch:** streaming NanoPitch + standard Viterbi; gesture does not widen the decode path.

**Gesture:** GestureTCN 3-class + heuristics; transition merge requires high `trans_p` and margin over steady; glissando demoted for coaching.

**Vibrato display:** central-relative cents, 3–9 Hz band-pass. **Vibrato scoring** uses rate/depth/consistency when `has_vibrato` is true.

**Breathiness:** CPP every ~15 chunks. **Dynamics:** RMS → pp…ff.

---

## Evaluation

### Live (no ground truth)

Tuner cents, phrase composite scores, coaching tiers.

### Offline

| Metric | Definition |
|---|---|
| **RPA** | % co-voiced frames within 50¢ of GT |
| **Gesture macro F1** | Per-class F1 on VocalSet val |

```bash
# Gesture (production)
python3 model/eval_gesture.py --checkpoint runs/gesture_tcn_3class/checkpoints/best.pth

# Pitch (synthetic test, NanoPitchFork)
cd ../NanoPitchFork/training && python3 evaluate.py --checkpoint runs/exp6/checkpoints/best.pth
```

Reference: exp6 ~**83–86% RPA** (synthetic); GestureTCN 3-class ~**51% macro F1**, **78% / 60%** steady/transition recall (see comparison table above).

---

## Dependencies

`torch`, `librosa`, `scipy`, `numpy`, `fastapi`, `uvicorn`, `soundfile`, `huggingface_hub`, `tensorboard`, `tqdm`

---

## VocalSet training setup

Annotations (~411 MB) + minimal VocalSet audio (~1–2 GB for five techniques).

```bash
python3 data/vocalset_download.py
# Download VocalSet1-2.zip → data/vocalset/VocalSet1-2.zip, then:
python3 data/vocalset_download.py --extract-minimal
python3 data/download.py
python3 data/vocalset_preprocess.py

# BreathCNN (phrase boundaries)
python3 model/train_breath.py --data-dir data/vocalset/processed --output-dir runs/breath_cnn

# GestureTCN 3-class (production gesture)
python3 model/train_gesture.py --n-classes 3 --output-dir runs/gesture_tcn_3class --epochs 40
python3 model/eval_gesture.py --checkpoint runs/gesture_tcn_3class/checkpoints/best.pth

python3 app.py
```

`data/vocalset_preprocess.py` builds gesture labels from Annotated-VocalSet f0 heuristics (shared with `features/pitch/gesture.py`).

---

## Gesture pipeline (live)

| Component | Role |
|---|---|
| **Heuristic gesture** | Vibrato band-pass, transition jumps + posterior entropy |
| **GestureTCN 3-class** | f0 TCN; logits merged with heuristics (gliss suppressed) |
| **Vibrato overlay** | steady → vibrato when deviation matches chart |
| **Gliss demote** | Unconfirmed gliss → steady |
| **Phrase scoring** | Steady frames for pitch; composite score at phrase end |
