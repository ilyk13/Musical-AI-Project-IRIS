| Topic | Status | Blocking dependency |
|---|---|---|
| Gesture model accuracy | ~28% val acc — needs retrain with new focal loss/weight | Run `train_multitask.py` with new settings |
| Hybrid design (heuristic + model) | Working, but model barely contributes | Gesture retrain |
| BreathCNN trained checkpoint | Not trained — no `.pth` exists | Run `train_breath.py` |
| BreathCNN wired into `app.py` | Not started | Trained BreathCNN checkpoint |
| Live phrase boundary detection | Not started | BreathCNN in `app.py` |
| Post-session per-phrase analysis | **Done** | — |
| Richer coaching (phrase-triggered) | Not started | Live phrase boundaries |

The natural order is: **retrain the gesture model → train BreathCNN → wire BreathCNN into `app.py` for live phrase boundaries → add per-phrase live WebSocket messages → extend the coaching bar to consume them**.
