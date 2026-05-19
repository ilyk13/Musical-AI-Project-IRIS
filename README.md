# IRIS — AI Vocal Coach
**Ilysia Krzywonos & Chris Zhao**

Real-time feedback on singing technique, analyzing pitch accuracy, vibrato quality, breathiness, and dynamics directly from a browser microphone with sub-100ms latency.

---

## Quick Start

```bash
pip3 install -r requirements.txt
python3 app.py
```

Open **http://localhost:8000**, click **Start Listening**, and sing.

---

## Project Overview

IRIS listens to a singer in real time and provides immediate, data-driven feedback across five dimensions:

| Dimension | What is measured | How |
|---|---|---|
| **Pitch** | Fundamental frequency (f0), cent deviation from equal temperament | NanoPitch GRU model (trained checkpoint) |
| **Vibrato** | Rate (Hz), depth (cents), consistency | Band-pass filter on f0 track |
| **Breathiness** | Cepstral Peak Prominence (CPP) | Analytical DSP |
| **Dynamics** | RMS loudness, dynamic level (pp → ff) | Analytical DSP |
| **Voice Activity** | Voiced / unvoiced frame detection | Model VAD head |

---

## Architecture

```
Browser microphone  (Web Audio API, 2048-sample chunks ≈ 46 ms)
        │
        │  Float32 PCM over WebSocket
        ▼
┌─────────────────────────────────────────────────────────────┐
│  Python server  (FastAPI + uvicorn)                         │
│                                                             │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Pitch  ·  NanoPitch GRU on 128 ms window           │   │
│  │          →  360-bin posteriorgram + VAD              │   │
│  │          →  Viterbi (realtime) → f0 in Hz            │   │
│  │          →  3-frame median smooth                    │   │
│  │          →  central pitch  (median filter on 3 s)   │   │
│  │          →  ET deviation & pitch accuracy %         │   │
│  ├──────────────────────────────────────────────────────┤   │
│  │  Dynamics  ·  RMS → dBFS → pp/p/mp/mf/f/ff          │   │
│  ├──────────────────────────────────────────────────────┤   │
│  │  Breathiness  ·  CPP + spectral tilt (every 3 chunks)│   │
│  ├──────────────────────────────────────────────────────┤   │
│  │  Vibrato  ·  4.5–8 Hz band-pass on f0 rolling buffer │   │
│  │             Savitzky-Golay smooth (every 8 chunks)   │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                             │
│  JSON frames over WebSocket  (~10 updates/second)           │
└─────────────────────────────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────┐
│  Browser dashboard  (Chart.js)        │
│  • Scrolling pitch chart (10 s)       │
│  • Vibrato deviation wave (6 s)       │
│  • Tuner needle + note name           │
│  • Pitch accuracy %                   │
│  • CPP breathiness meter              │
│  • Dynamic level indicator            │
│  • Live text feedback                 │
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
│   ├── nanopitch.py              # NanoPitch GRU architecture + Viterbi decoder
│   └── train.py                  # Training script (uses NanoPitch-PreExtract data)
│
├── features/
│   ├── pitch/
│   │   ├── nanopitch.py          # NanoPitchExtractor — loads checkpoint, runs inference
│   │   ├── central_pitch.py      # Median-smoothed f0 reference line
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
│   ├── download.py               # Download NanoPitch-PreExtract from Hugging Face
│   ├── clean.npz                 # Pre-extracted clean singing mel + f0 + VAD
│   ├── noise.npz                 # Pre-extracted noise mel (for augmentation)
│   └── test.npz                  # Held-out noisy clips for evaluation
│
├── runs/
│   └── exp1/
│       ├── checkpoints/
│       │   └── best.pth          # Best trained NanoPitch checkpoint (50 epochs)
│       └── tb/                   # TensorBoard event logs
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

The 360 output bins cover B0–B6 (31.7–2006 Hz) at 20 cents/bin resolution. A **Viterbi decoder** converts the posteriorgram into a smooth f0 track.

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

A trained checkpoint lives at `runs/exp1/checkpoints/best.pth` (50 epochs).

To use the trained model in the live app, update `app.py`'s `_extract` function to use `NanoPitchExtractor.from_pretrained(local_path="runs/exp1/checkpoints/best.pth")` in place of `librosa.yin`.

---

## Feature Extraction Details

### Pitch (f0)

The live app uses the **trained NanoPitch GRU model** (`runs/exp1/checkpoints/best.pth`) for pitch extraction. On each 128 ms rolling window, a log-mel spectrogram is computed and fed to the model. The model outputs a per-frame pitch posteriorgram and VAD probability. `viterbi_decode_realtime` converts the posteriorgram into a smooth f0 track, and the model's VAD head gates voiced/unvoiced decisions directly.

**Central pitch** is the median-filtered f0 over a 3-second rolling buffer — this removes short-term vibrato and micro-variations to reveal the intended note. The deviation from the nearest equal-temperament note is displayed as a tuner needle (±50 cents).

### Vibrato

The raw f0 track is band-pass filtered (4th-order Butterworth, 4.5–8 Hz) to isolate the vibrato oscillation. A Savitzky-Golay smoother (window=21, degree=3) cleans up the signal before display. Requires a minimum of 120 consecutive voiced frames (~1.2 seconds) before activating.

Vibrato rate (Hz) and depth (cents peak-to-peak) are extracted via FFT on the filtered signal. Ideal professional range: 5–7 Hz rate, 25–75 cents depth.

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

The live dashboard shows:

- **Pitch accuracy %** — rolling percentage of voiced frames within ±25 cents of the nearest equal-temperament note (window: last 300 voiced frames ≈ 30 seconds)
- **Tuner needle** — instantaneous deviation from the nearest ET note, ±50 cents
- **CPP** — per-frame breathiness score in dB
- **VAD/RPA** — the training evaluation computes Voicing Detection Rate and Raw Pitch Accuracy (fraction of co-voiced frames within 50 cents of ground truth)

---

## Dependencies

| Package | Use |
|---|---|
| `torch` | NanoPitch model, training |
| `librosa` | pYIN / YIN pitch detection, audio resampling, mel spectrograms |
| `scipy` | Band-pass filter, median filter, Savitzky-Golay smoother |
| `numpy` | All numerical computation |
| `fastapi` + `uvicorn` | Web server and WebSocket |
| `soundfile` | Audio file I/O |
| `huggingface_hub` | Dataset download |
| `tensorboard` | Training monitoring |
| `tqdm` | Training progress bars |

---

## Phase 2 — Future Work

The current demo uses the base NanoPitch architecture (f0 + VAD only). The planned NanoPitch+ extension adds multi-task output heads trained on GTSinger annotations:

- **Gesture classification head** — steady / vibrato / glissando / transition (frame-level)
- **Register classification head** — chest / mixed / head / falsetto
- **BreathCNN** — small CNN trained to detect breath events from mel spectrograms for phrase segmentation
- **Gesture-aware Viterbi** — modified decoder that applies analytic vibrato/glissando shape constraints before scoring

These heads allow the system to score pitch accuracy only on steady frames, skip scoring during transitions, and segment the performance into musical phrases for per-phrase feedback.
