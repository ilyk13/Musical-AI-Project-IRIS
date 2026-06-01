#!/usr/bin/env python3
"""Evaluate a NanoPitch+ checkpoint on the full VocalSet val split.

Reports gesture (overall + per-class), pitch (RPA/VDR), and optional
register/dynamics accuracy. Writes a JSON summary for comparing runs
(e.g. before vs after focal-loss retrain).

Usage:
    python3 model/eval_multitask.py \\
        --checkpoint runs/vocalset_plus/checkpoints/best.pth

    python3 model/eval_multitask.py \\
        --checkpoint runs/vocalset_plus/checkpoints/epoch_040.pth \\
        --out runs/vocalset_plus/eval_epoch_040.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data.vocalset_labels import (
    DYNAMIC_SILENCE,
    DYNAMIC_VOCAB,
    GESTURE_VOCAB,
    REGISTER_UNKNOWN,
    REGISTER_VOCAB,
)
from model.nanopitch import (
    NanoPitchPlus,
    viterbi_decode,
    viterbi_decode_gesture,
)


def _device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_plus(checkpoint: str, device: torch.device) -> tuple[NanoPitchPlus, dict]:
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    if not isinstance(ckpt, dict) or "state_dict" not in ckpt:
        raise ValueError(f"Expected training checkpoint dict in {checkpoint}")
    kwargs = ckpt.get("model_kwargs", {})
    model = NanoPitchPlus(
        cond_size=kwargs.get("cond_size", 64),
        gru_size=kwargs.get("gru_size", 96),
    ).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    meta = {
        "epoch": ckpt.get("epoch"),
        "train_loss": ckpt.get("loss"),
        "checkpoint": str(Path(checkpoint).resolve()),
    }
    return model, meta


def _confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> np.ndarray:
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        if 0 <= t < n_classes and 0 <= p < n_classes:
            cm[t, p] += 1
    return cm


def _per_class_f1(cm: np.ndarray, labels: list[str]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for c, name in enumerate(labels):
        tp = cm[c, c]
        pred = cm[:, c].sum()
        true = cm[c, :].sum()
        prec = float(tp / pred) if pred else 0.0
        rec = float(tp / true) if true else 0.0
        f1 = float(2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
        out[name] = {
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "support": int(true),
        }
    return out


def _macro_f1(per_class: dict[str, dict]) -> float:
    f1s = [v["f1"] for v in per_class.values() if v["support"] > 0]
    return float(np.mean(f1s)) if f1s else 0.0


def _rpa(f0_gt: np.ndarray, f0_dec: np.ndarray) -> float | None:
    both = (f0_gt > 0) & (f0_dec > 0)
    if both.sum() == 0:
        return None
    err = np.abs(1200.0 * np.log2(f0_dec[both] / (f0_gt[both] + 1e-10) + 1e-10))
    return float(np.mean(err < 50.0))


def _vdr(f0_gt: np.ndarray, f0_dec: np.ndarray) -> float | None:
    gt_v = f0_gt > 0
    if gt_v.sum() == 0:
        return None
    return float(np.mean((f0_dec > 0)[gt_v]))


@torch.no_grad()
def evaluate_checkpoint(
    model: NanoPitchPlus,
    val_path: str,
    device: torch.device,
    max_clips: int | None = None,
) -> dict:
    data = np.load(val_path, allow_pickle=True)
    lengths = data["lengths"]
    n_clips = len(lengths) if max_clips is None else min(len(lengths), max_clips)

    gest_true_parts: list[np.ndarray] = []
    gest_pred_parts: list[np.ndarray] = []
    reg_true_parts: list[np.ndarray] = []
    reg_pred_parts: list[np.ndarray] = []
    dyn_true_parts: list[np.ndarray] = []
    dyn_pred_parts: list[np.ndarray] = []

    rpa_std: list[float] = []
    rpa_gest: list[float] = []
    vdr_std: list[float] = []
    vdr_gest: list[float] = []

    offset = 0
    for clip_i in range(n_clips):
        length = int(lengths[clip_i])
        mel = torch.from_numpy(
            data["mel"][offset:offset + length].astype(np.float32)
        ).unsqueeze(0).to(device)
        f0_gt = data["f0"][offset:offset + length]
        gest_gt = data["gesture"][offset:offset + length]
        reg_gt = data["register"][offset:offset + length]
        dyn_gt = data["dynamics"][offset:offset + length]

        _, pitch, gest_logits, reg_logits, dyn_logits, _ = model(mel)
        gest_pred = gest_logits[0].argmax(-1).cpu().numpy()
        reg_pred = reg_logits[0].argmax(-1).cpu().numpy()
        dyn_pred = dyn_logits[0].argmax(-1).cpu().numpy()

        gest_true_parts.append(gest_gt)
        gest_pred_parts.append(gest_pred)

        reg_mask = reg_gt >= 0
        if reg_mask.any():
            reg_true_parts.append(reg_gt[reg_mask])
            reg_pred_parts.append(reg_pred[reg_mask])

        dyn_mask = dyn_gt >= 0
        if dyn_mask.any():
            dyn_true_parts.append(dyn_gt[dyn_mask])
            dyn_pred_parts.append(dyn_pred[dyn_mask])

        post = pitch[0].cpu().numpy()
        f0_std = viterbi_decode(post)
        f0_gest = viterbi_decode_gesture(post, gest_pred)
        for f0_dec, rpa_list, vdr_list in (
            (f0_std, rpa_std, vdr_std),
            (f0_gest, rpa_gest, vdr_gest),
        ):
            r = _rpa(f0_gt, f0_dec)
            v = _vdr(f0_gt, f0_dec)
            if r is not None:
                rpa_list.append(r)
            if v is not None:
                vdr_list.append(v)

        offset += length

    gest_true = np.concatenate(gest_true_parts)
    gest_pred = np.concatenate(gest_pred_parts)
    n_gest = len(GESTURE_VOCAB)
    gest_cm = _confusion_matrix(gest_true, gest_pred, n_gest)
    gest_acc = float(np.mean(gest_true == gest_pred))
    gest_counts = np.bincount(gest_true, minlength=n_gest)
    majority = int(gest_counts.argmax())
    majority_acc = float(gest_counts[majority] / max(len(gest_true), 1))
    gest_per_class = _per_class_f1(gest_cm, GESTURE_VOCAB)

    voiced = gest_true > 0  # all gesture classes are voiced labels in practice
    gest_acc_voiced = float(np.mean(gest_true[voiced] == gest_pred[voiced])) if voiced.any() else gest_acc

    results: dict = {
        "n_clips": n_clips,
        "n_frames": int(len(gest_true)),
        "gesture": {
            "accuracy": round(gest_acc, 4),
            "accuracy_voiced_frames": round(gest_acc_voiced, 4),
            "majority_class": GESTURE_VOCAB[majority],
            "majority_baseline_accuracy": round(majority_acc, 4),
            "macro_f1": round(_macro_f1(gest_per_class), 4),
            "per_class": gest_per_class,
            "confusion_matrix": gest_cm.tolist(),
            "class_counts_true": {
                GESTURE_VOCAB[i]: int(gest_counts[i]) for i in range(n_gest)
            },
        },
        "pitch": {
            "rpa_standard_viterbi": round(float(np.mean(rpa_std)), 4) if rpa_std else None,
            "rpa_gesture_viterbi": round(float(np.mean(rpa_gest)), 4) if rpa_gest else None,
            "vdr_standard_viterbi": round(float(np.mean(vdr_std)), 4) if vdr_std else None,
            "vdr_gesture_viterbi": round(float(np.mean(vdr_gest)), 4) if vdr_gest else None,
        },
    }

    if reg_true_parts:
        reg_true = np.concatenate(reg_true_parts)
        reg_pred = np.concatenate(reg_pred_parts)
        n_reg = len(REGISTER_VOCAB)
        reg_cm = _confusion_matrix(reg_true, reg_pred, n_reg)
        reg_per = _per_class_f1(reg_cm, REGISTER_VOCAB)
        results["register"] = {
            "accuracy": round(float(np.mean(reg_true == reg_pred)), 4),
            "macro_f1": round(_macro_f1(reg_per), 4),
            "per_class": reg_per,
        }

    if dyn_true_parts:
        dyn_true = np.concatenate(dyn_true_parts)
        dyn_pred = np.concatenate(dyn_pred_parts)
        n_dyn = len(DYNAMIC_VOCAB)
        dyn_cm = _confusion_matrix(dyn_true, dyn_pred, n_dyn)
        dyn_per = _per_class_f1(dyn_cm, DYNAMIC_VOCAB)
        results["dynamics"] = {
            "accuracy": round(float(np.mean(dyn_true == dyn_pred)), 4),
            "macro_f1": round(_macro_f1(dyn_per), 4),
            "per_class": dyn_per,
        }

    return results


def _print_report(meta: dict, results: dict) -> None:
    print(f"\nCheckpoint: {meta['checkpoint']}")
    if meta.get("epoch") is not None:
        print(f"  epoch={meta['epoch']}  train_loss={meta.get('train_loss')}")
    print(f"Val clips: {results['n_clips']}  frames: {results['n_frames']:,}")

    g = results["gesture"]
    print("\n── Gesture ──")
    print(f"  accuracy:           {g['accuracy']:.1%}")
    print(f"  majority baseline:  {g['majority_baseline_accuracy']:.1%} ({g['majority_class']})")
    print(f"  macro F1:           {g['macro_f1']:.1%}")
    print("  per-class:")
    for name in GESTURE_VOCAB:
        pc = g["per_class"][name]
        print(f"    {name:12s}  P={pc['precision']:.1%}  R={pc['recall']:.1%}  "
              f"F1={pc['f1']:.1%}  n={pc['support']}")

    p = results["pitch"]
    print("\n── Pitch (clip-mean) ──")
    print(f"  RPA standard Viterbi: {p['rpa_standard_viterbi']}")
    print(f"  RPA gesture Viterbi:  {p['rpa_gesture_viterbi']}")
    print(f"  VDR standard:         {p['vdr_standard_viterbi']}")
    print(f"  VDR gesture:          {p['vdr_gesture_viterbi']}")

    if "register" in results:
        r = results["register"]
        print(f"\n── Register ──  acc={r['accuracy']:.1%}  macro_f1={r['macro_f1']:.1%}")
    if "dynamics" in results:
        d = results["dynamics"]
        print(f"── Dynamics ──  acc={d['accuracy']:.1%}  macro_f1={d['macro_f1']:.1%}")
    print()


def main() -> int:
    parser = argparse.ArgumentParser(description="NanoPitch+ val evaluation")
    parser.add_argument(
        "--checkpoint",
        default="runs/vocalset_plus/checkpoints/best.pth",
        help="Path to .pth training checkpoint",
    )
    parser.add_argument(
        "--data-dir",
        default="data/vocalset/processed",
        help="Directory containing val.npz",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--max-clips",
        type=int,
        default=None,
        help="Limit val clips (default: all)",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Write JSON metrics here (default: <checkpoint_dir>/eval_results.json)",
    )
    args = parser.parse_args()

    val_path = os.path.join(args.data_dir, "val.npz")
    if not os.path.exists(val_path):
        print(f"Missing {val_path}. Run: python3 data/vocalset_preprocess.py")
        return 1
    if not os.path.exists(args.checkpoint):
        print(f"Missing checkpoint: {args.checkpoint}")
        return 1

    device = _device(args.device)
    print(f"Device: {device}")
    model, meta = load_plus(args.checkpoint, device)
    print("NanoPitchPlus loaded.")

    results = evaluate_checkpoint(model, val_path, device, max_clips=args.max_clips)
    payload = {"meta": meta, "results": results}

    out_path = args.out
    if out_path is None:
        out_path = str(Path(args.checkpoint).parent / "eval_results.json")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Wrote {out_path}")

    _print_report(meta, results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
