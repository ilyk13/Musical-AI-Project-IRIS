#!/usr/bin/env python3
"""Evaluate GestureTCN on VocalSet val."""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from model.gesture_classes import gesture_vocab, metrics_from_cm, remap_gesture_labels_4_to_n
from model.gesture_features import build_gesture_features_np
from model.gesture_tcn import GestureTCN
from model.train_multitask import _macro_f1_from_cm


def evaluate_checkpoint(model: GestureTCN, val_path: str, device: torch.device) -> dict:
    n_classes = model.n_classes
    data = np.load(val_path, allow_pickle=True)
    lengths = data["lengths"]
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    offset = 0
    names = gesture_vocab(n_classes)
    model.eval()
    with torch.no_grad():
        for clip_i in range(len(lengths)):
            length = int(lengths[clip_i])
            f0 = data["f0"][offset:offset + length]
            vad = data["vad"][offset:offset + length]
            gest_gt = remap_gesture_labels_4_to_n(
                data["gesture"][offset:offset + length], n_classes,
            )
            feat = build_gesture_features_np(f0, vad)
            x = torch.from_numpy(feat).unsqueeze(0).to(device)
            pred = model(x)[0].argmax(-1).cpu().numpy()
            for t, p in zip(gest_gt, pred):
                if 0 <= t < n_classes and 0 <= p < n_classes:
                    cm[t, p] += 1
            offset += length

    extra = metrics_from_cm(cm, n_classes)
    per_class = {}
    for c, name in enumerate(names):
        tp = cm[c, c]
        pred_n = cm[:, c].sum()
        true_n = cm[c, :].sum()
        prec = float(tp / pred_n) if pred_n else 0.0
        rec = float(tp / true_n) if true_n else 0.0
        f1 = float(2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
        per_class[name] = {
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "support": int(true_n),
        }

    total = int(cm.sum())
    return {
        "n_classes": n_classes,
        "n_clips": len(lengths),
        "n_frames": total,
        "gesture": {
            "accuracy": round(float(np.trace(cm) / total), 4) if total else 0.0,
            "macro_f1": round(_macro_f1_from_cm(cm), 4),
            "balanced_acc": round(extra["balanced_acc"], 4),
            "per_class": per_class,
            "confusion_matrix": cm.tolist(),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="GestureTCN evaluation")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-dir", default="data/vocalset/processed")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    model = GestureTCN.load_checkpoint(args.checkpoint, device)
    val_path = os.path.join(args.data_dir, "val.npz")
    results = evaluate_checkpoint(model, val_path, device)

    out_path = os.path.join(os.path.dirname(args.checkpoint), "eval_results.json")
    with open(out_path, "w") as f:
        json.dump({"checkpoint": args.checkpoint, "results": results}, f, indent=2)

    g = results["gesture"]
    print(f"GestureTCN ({results['n_classes']}-class) val — "
          f"acc {g['accuracy']:.1%}  macro F1 {g['macro_f1']:.1%}  "
          f"balanced {g['balanced_acc']:.1%}")
    for name, m in g["per_class"].items():
        print(f"  {name:12s} P={m['precision']:.1%}  R={m['recall']:.1%}  F1={m['f1']:.1%}")
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
