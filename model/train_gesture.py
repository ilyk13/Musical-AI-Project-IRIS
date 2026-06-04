#!/usr/bin/env python3
"""Train standalone GestureTCN on VocalSet f0 + gesture labels."""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from model.gesture_classes import (
    compute_tcn_class_weights,
    gesture_vocab,
    metrics_from_cm,
    remap_gesture_labels_4_to_n,
)
from model.gesture_features import build_gesture_features_np, N_GESTURE_FEATURES
from model.gesture_tcn import GestureTCN
from model.train_multitask import focal_cross_entropy, _gesture_selection_score, _macro_f1_from_cm


parser = argparse.ArgumentParser(description="GestureTCN trainer")
parser.add_argument("--data-dir", default="data/vocalset/processed")
parser.add_argument("--output-dir", default="runs/gesture_tcn_3class")
parser.add_argument("--n-classes", type=int, default=3, choices=[3, 4],
                    help="3 = steady/vibrato/transition (gliss→steady); 4 = full vocab")
parser.add_argument("--device", default="auto")
parser.add_argument("--epochs", type=int, default=40)
parser.add_argument("--batch-size", type=int, default=32)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--seq-len", type=int, default=200)
parser.add_argument("--channels", type=int, default=64)
parser.add_argument("--dropout", type=float, default=0.15)
parser.add_argument("--focal-gamma", type=float, default=1.5)
parser.add_argument("--w-gesture", type=float, default=1.0)
parser.add_argument("--rare-gesture-oversample", type=int, default=2,
                    help="Extra copies of segments containing transition (not gliss)")
parser.add_argument("--transition-weight-cap", type=float, default=5.0)
parser.add_argument("--gesture-weight-max-ratio", type=float, default=2.5)
parser.add_argument("--eval-every", type=int, default=5)
parser.add_argument("--min-steady-recall", type=float, default=0.40)
parser.add_argument("--min-transition-recall", type=float, default=0.05)


class GestureFeatureDataset(Dataset):
    def __init__(
        self,
        npz_path: str,
        seq_len: int = 200,
        rare_oversample: int = 2,
        n_classes: int = 3,
    ):
        self.seq_len = seq_len
        self.n_classes = n_classes
        data = np.load(npz_path, allow_pickle=True)
        self.f0 = data["f0"].astype(np.float32)
        self.vad = data["vad"].astype(np.float32)
        self.gesture = remap_gesture_labels_4_to_n(data["gesture"].astype(np.int64), n_classes)
        self.lengths = data["lengths"]
        self.segs = self._build_segments(self.lengths, seq_len)
        self.rng = np.random.default_rng()

        trans_idx = 2 if n_classes == 3 else 3
        extra: list[tuple[int, int]] = []
        for start, end in self.segs:
            g = self.gesture[start:end]
            if np.any(g == trans_idx):
                extra.append((start, end))
        n_extra = max(0, int(rare_oversample))
        if extra and n_extra > 0:
            self.segs = self.segs + extra * n_extra
        print(f"  {npz_path}: {len(self.segs)} segments  n_classes={n_classes}")

    def _build_segments(self, lengths, min_len):
        segs, offset = [], 0
        for length in lengths:
            if length >= min_len:
                segs.append((offset, offset + length))
            offset += length
        return segs

    def __len__(self):
        return max(len(self.segs), 1)

    def __getitem__(self, idx):
        start, end = self.segs[self.rng.integers(len(self.segs))]
        s = start + self.rng.integers(0, end - start - self.seq_len + 1)
        e = s + self.seq_len
        f0 = self.f0[s:e]
        vad = self.vad[s:e]
        gest = self.gesture[s:e]
        feat = build_gesture_features_np(f0, vad)
        return (
            torch.from_numpy(feat),
            torch.from_numpy(gest),
            torch.from_numpy((vad > 0.5) | (f0 > 0)),
        )


@torch.no_grad()
def evaluate(model, val_path, device, n_classes: int, max_clips: int = 0) -> dict:
    data = np.load(val_path, allow_pickle=True)
    lengths = data["lengths"]
    n_clips = len(lengths) if max_clips <= 0 else min(len(lengths), max_clips)
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    offset = 0
    model.eval()
    for clip_i in range(n_clips):
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
    macro_f1 = _macro_f1_from_cm(cm)
    return {
        "macro_f1": macro_f1,
        "steady_recall": extra["steady_recall"],
        "vibrato_precision": extra["vibrato_precision"],
        "vibrato_recall": extra["vibrato_recall"],
        "transition_recall": extra["transition_recall"],
        "balanced_acc": extra["balanced_acc"],
        "gesture_acc": float(np.trace(cm) / cm.sum()) if cm.sum() else 0.0,
    }


def _qualifies(metrics: dict, args) -> bool:
    return (
        metrics.get("steady_recall", 0) >= args.min_steady_recall
        and metrics.get("transition_recall", 0) >= args.min_transition_recall
    )


def main() -> int:
    args = parser.parse_args()
    n_classes = args.n_classes
    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}  n_classes={n_classes}  vocab={gesture_vocab(n_classes)}")

    output_dir = os.path.abspath(args.output_dir)
    ckpt_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    train_npz = os.path.join(args.data_dir, "train.npz")
    val_npz = os.path.join(args.data_dir, "val.npz")
    if not os.path.exists(train_npz):
        print(f"Missing {train_npz}")
        return 1

    train_ds = GestureFeatureDataset(
        train_npz, args.seq_len,
        rare_oversample=args.rare_gesture_oversample,
        n_classes=n_classes,
    )
    loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True,
        pin_memory=(device.type == "cuda"),
    )

    model = GestureTCN(channels=args.channels, dropout=args.dropout, n_classes=n_classes).to(device)
    raw_gesture = np.load(train_npz, allow_pickle=True)["gesture"]
    gw = compute_tcn_class_weights(
        raw_gesture,
        n_classes,
        max_ratio=args.gesture_weight_max_ratio,
        transition_cap=args.transition_weight_cap,
    )
    class_weight = torch.tensor(gw, device=device)
    print("Class weights:", dict(zip(gesture_vocab(n_classes), gw.round(3))))

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, total_steps=len(loader) * args.epochs, pct_start=0.08,
    )
    writer = SummaryWriter(os.path.join(output_dir, "tb"))
    best_score = -1.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        n_batches = 0
        for feat, gest, voiced in tqdm(loader, desc=f"Epoch {epoch}"):
            feat = feat.to(device)
            gest = gest.to(device)
            voiced = voiced.to(device)
            logits = model(feat)
            v = voiced.reshape(-1)
            loss = focal_cross_entropy(
                logits.reshape(-1, n_classes)[v],
                gest.reshape(-1)[v],
                gamma=args.focal_gamma,
                class_weight=class_weight,
            ) * args.w_gesture
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            scheduler.step()
            running += loss.item()
            n_batches += 1

        avg = running / max(n_batches, 1)
        writer.add_scalar("train/loss", avg, epoch)
        print(f"Epoch {epoch}  loss={avg:.5f}")

        val_metrics = {"macro_f1": 0.0, "steady_recall": 0.0, "transition_recall": 0.0,
                       "balanced_acc": 0.0, "gesture_acc": 0.0,
                       "vibrato_precision": 0.0, "vibrato_recall": 0.0}
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            val_metrics = evaluate(model, val_npz, device, n_classes)
            writer.add_scalar("eval/macro_f1", val_metrics["macro_f1"], epoch)
            writer.add_scalar("eval/steady_recall", val_metrics["steady_recall"], epoch)
            print(f"  val macro F1: {val_metrics['macro_f1']:.1%}  "
                  f"steady rec: {val_metrics['steady_recall']:.1%}  "
                  f"trans rec: {val_metrics['transition_recall']:.1%}  "
                  f"balanced: {val_metrics['balanced_acc']:.1%}")

        ckpt = {
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "model_type": "GestureTCN",
            "model_kwargs": {
                "n_features": N_GESTURE_FEATURES,
                "n_classes": n_classes,
                "channels": args.channels,
                "dropout": args.dropout,
            },
            "loss": avg,
            **{f"val_{k}": v for k, v in val_metrics.items()},
        }
        torch.save(ckpt, os.path.join(ckpt_dir, f"epoch_{epoch:03d}.pth"))

        if _qualifies(val_metrics, args):
            score = _gesture_selection_score({**val_metrics, "rpa_std": 0.0})
            if score > best_score:
                best_score = score
                torch.save(ckpt, os.path.join(ckpt_dir, "best.pth"))
                print(f"  → best.pth  score={score:.3f}  macro F1={val_metrics['macro_f1']:.1%}")

    writer.close()
    print(f"Done. Checkpoints: {ckpt_dir}")
    print(f"Eval: python3 model/eval_gesture.py --checkpoint {ckpt_dir}/best.pth")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
