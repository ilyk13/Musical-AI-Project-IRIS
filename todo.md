| Topic | Status | Blocking dependency |
|---|---|---|
| Gesture model accuracy | **v4 prep** — fixed glissando labels + live merge | Re-preprocess + retrain v4 |

### Retrain gesture (v4.2 — current)

**Glissando fix:** v4.1 had ~1% glissando labels because jump regions were marked
transition before glissando could apply. v4.2 allows scoops on pitch ramps;
target ~5–10% glissando. Training also oversamples glissando/transition clips
and uses 6× class weight for glissando.

1. Stop current run → `python3 data/vocalset_preprocess.py`
2. Verify val glissando **≥3%** (not ~1%)
3. Retrain v4 from scratch (delete or use new output dir)
- Training labels: **vibrato before glissando** in priority
- Stricter glissando (sustained slide + min displacement, excludes vibrato band)
- Separate train vs live thresholds
- Target val mix ≈ steady 85%, vibrato 3–8%, glissando 1–3%, transition 4–8%

1. **Stop** the broken v4 run if still going
2. Rebuild labels: `python3 data/vocalset_preprocess.py`
3. Verify distribution:
   ```bash
   python3 -c "
   import numpy as np
   from collections import Counter
   for s in ('train','val'):
       g = np.load(f'data/vocalset/processed/{s}.npz')['gesture']
       n = len(g); c = Counter(g.tolist())
       print(s, {['steady','vibrato','glissando','transition'][k]: f'{100*v/n:.1f}%' for k,v in sorted(c.items())})
   "
   ```
4. Train:
   ```bash
   python3 model/train_multitask.py \
     --resume runs/vocalset_plus_v3/checkpoints/best.pth \
     --output-dir runs/vocalset_plus_v4 \
     --epochs 40
   ```

### v3 reference (superseded)

```bash
python3 model/train_multitask.py \
  --resume runs/vocalset_plus/checkpoints/best.pth \
  --output-dir runs/vocalset_plus_v3 \
  --epochs 40
```
v3 best: macro F1 34%, vibrato precision **27%**, glissando **0** val frames.
| Hybrid design (heuristic + model) | Working, but model barely contributes | Gesture retrain |
| BreathCNN trained checkpoint | **Done** — `runs/breath_cnn/checkpoints/best.pth` | — |
| BreathCNN wired into `app.py` | **Done** — live phrase boundaries on WebSocket | — |
| Live phrase boundary detection | **Done** — Phrase block + end coaching card | Tune thresholds in `phrase.py` |
| Post-session per-phrase analysis | **Done** | — |
| Richer coaching (phrase-triggered) | Not started | Live phrase boundaries |

The natural order is: **retrain the gesture model → train BreathCNN → wire BreathCNN into `app.py` for live phrase boundaries → add per-phrase live WebSocket messages → extend the coaching bar to consume them**.
