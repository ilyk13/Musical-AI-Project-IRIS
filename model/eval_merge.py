#!/usr/bin/env python3
"""Evaluate live-style gesture merge (heuristics + GestureTCN) on VocalSet val.

Compares heuristic-only vs merged labels. Focus on transition precision
(fewer steady frames called transition).
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from features.pitch.gesture import (
    GESTURE_STEADY,
    GESTURE_TRANSITION,
    classify_gestures_live,
    merge_gesture_predictions,
)
from model.gesture_classes import expand_logits_to_4class
from model.gesture_features import build_gesture_features_np
from model.gesture_tcn import GestureTCN
from model.train_multitask import _macro_f1_from_cm


def _per_class_pr(cm: np.ndarray, names: list[str]) -> dict:
    out = {}
    for c, name in enumerate(names):
        tp = int(cm[c, c])
        pred_n = int(cm[:, c].sum())
        true_n = int(cm[c, :].sum())
        out[name] = {
            "precision": round(tp / pred_n, 4) if pred_n else 0.0,
            "recall": round(tp / true_n, 4) if true_n else 0.0,
            "support": true_n,
        }
    return out


def eval_merge_on_val(
    model: GestureTCN,
    val_path: str,
    merge_kw: dict,
    *,
    n_eval_classes: int = 4,
) -> dict:
    """GT always 4-class remapped; preds are 4-class after merge."""
    names_4 = ["steady", "vibrato", "glissando", "transition"]
    cm_heur = np.zeros((4, 4), dtype=np.int64)
    cm_merge = np.zeros((4, 4), dtype=np.int64)
    steady_as_trans_heur = 0
    steady_as_trans_merge = 0
    steady_total = 0
    offset = 0
    data = np.load(val_path, allow_pickle=True)
    lengths = data["lengths"]
    model.eval()

    with torch.no_grad():
        for length in lengths:
            length = int(length)
            f0 = data["f0"][offset:offset + length].astype(np.float64)
            vad = data["vad"][offset:offset + length]
            gt = data["gesture"][offset:offset + length].astype(np.int64)
            feat = build_gesture_features_np(f0, vad)
            x = torch.from_numpy(feat).unsqueeze(0)
            logits = model(x)[0].cpu().numpy()
            if model.n_classes == 3:
                logits = expand_logits_to_4class(logits)

            heur = classify_gestures_live(f0.astype(np.float32), posteriorgram=None)
            merged = merge_gesture_predictions(heur, logits, **merge_kw)

            for g, h, m in zip(gt, heur, merged):
                if 0 <= g < 4 and 0 <= h < 4 and 0 <= m < 4:
                    cm_heur[g, h] += 1
                    cm_merge[g, m] += 1
                if g == GESTURE_STEADY:
                    steady_total += 1
                    if h == GESTURE_TRANSITION:
                        steady_as_trans_heur += 1
                    if m == GESTURE_TRANSITION:
                        steady_as_trans_merge += 1
            offset += length

    return {
        "heuristic_only": {
            "macro_f1": round(_macro_f1_from_cm(cm_heur), 4),
            "per_class": _per_class_pr(cm_heur, names_4),
            "steady_to_transition_fp": steady_as_trans_heur,
            "steady_support": steady_total,
            "steady_to_transition_rate": round(
                steady_as_trans_heur / max(steady_total, 1), 4,
            ),
        },
        "heuristic_plus_tcn_merge": {
            "macro_f1": round(_macro_f1_from_cm(cm_merge), 4),
            "per_class": _per_class_pr(cm_merge, names_4),
            "steady_to_transition_fp": steady_as_trans_merge,
            "steady_support": steady_total,
            "steady_to_transition_rate": round(
                steady_as_trans_merge / max(steady_total, 1), 4,
            ),
        },
        "merge_kwargs": merge_kw,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate heuristic + TCN merge on val")
    parser.add_argument("--checkpoint", default="runs/gesture_tcn_3class/checkpoints/best.pth")
    parser.add_argument("--data-dir", default="data/vocalset/processed")
    parser.add_argument("--transition-conf-min", type=float, default=0.84)
    parser.add_argument("--transition-prob-min", type=float, default=0.52)
    parser.add_argument("--transition-margin", type=float, default=0.18)
    parser.add_argument("--sweep", action="store_true",
                        help="Print transition P/R for conf_min 0.78..0.92")
    args = parser.parse_args()

    device = torch.device("cpu")
    model = GestureTCN.load_checkpoint(args.checkpoint, device)
    val_path = os.path.join(args.data_dir, "val.npz")

    if args.sweep:
        print(f"{'conf_min':>8} {'margin':>8}  trans_P  trans_R  steady→trans%")
        for conf in (0.78, 0.80, 0.82, 0.84, 0.86, 0.88, 0.90):
            kw = dict(
                transition_conf_min=conf,
                transition_prob_min=args.transition_prob_min,
                transition_margin=args.transition_margin,
                vibrato_prob_min=0.74,
                model_confidence=0.55,
            )
            r = eval_merge_on_val(model, val_path, kw)
            m = r["heuristic_plus_tcn_merge"]["per_class"]["transition"]
            st = r["heuristic_plus_tcn_merge"]["steady_to_transition_rate"]
            print(
                f"{conf:8.2f} {args.transition_margin:8.2f}  "
                f"{m['precision']:6.1%}  {m['recall']:6.1%}  {st:6.2%}"
            )
        return 0

    merge_kw = dict(
        transition_conf_min=args.transition_conf_min,
        transition_prob_min=args.transition_prob_min,
        transition_margin=args.transition_margin,
        vibrato_prob_min=0.74,
        model_confidence=0.55,
    )
    results = eval_merge_on_val(model, val_path, merge_kw)

    out_path = os.path.join(os.path.dirname(args.checkpoint), "eval_merge_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    for label in ("heuristic_only", "heuristic_plus_tcn_merge"):
        block = results[label]
        t = block["per_class"]["transition"]
        print(f"\n{label}")
        print(f"  macro F1: {block['macro_f1']:.1%}")
        print(f"  transition  P={t['precision']:.1%}  R={t['recall']:.1%}  "
              f"(support {t['support']})")
        print(f"  steady→transition FP rate: {block['steady_to_transition_rate']:.2%} "
              f"({block['steady_to_transition_fp']} / {block['steady_support']} steady frames)")
    print(f"\nmerge_kwargs: {merge_kw}")
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
