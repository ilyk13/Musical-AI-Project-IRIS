#!/usr/bin/env python3
"""
Multi-task NanoPitch+ trainer — VocalSet annotations.

Trains f0 + VAD (existing heads) plus gesture, register, and dynamics heads
on preprocessed VocalSet tensors from vocalset_preprocess.py.

Optionally fine-tunes from a NanoPitch pitch checkpoint (--resume).

Usage:
    python3 data/vocalset_preprocess.py
    python3 model/train_multitask.py --data-dir data/vocalset/processed
    python3 model/train_multitask.py --resume runs/exp1/checkpoints/best.pth
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import warnings

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data.vocalset_labels import DYNAMIC_SILENCE, GESTURE_VOCAB, REGISTER_UNKNOWN
from model.nanopitch import (
    NanoPitchPlus,
    PITCH_BINS,
    PITCH_CENTS_PER_BIN,
    PITCH_FMIN,
    viterbi_decode,
    viterbi_decode_gesture,
)
from model.train import augment_mel_batch


parser = argparse.ArgumentParser(description="NanoPitch+ multi-task trainer")
parser.add_argument("--data-dir", default="data/vocalset/processed")
parser.add_argument("--noise-dir", default="data",
                    help="Directory with noise.npz for SNR augmentation")
parser.add_argument("--output-dir", default="runs/vocalset_plus")
parser.add_argument("--resume", default=None,
                    help="NanoPitch or NanoPitch+ checkpoint (.pth)")
parser.add_argument("--device", default="auto")
parser.add_argument("--cond-size", type=int, default=64)
parser.add_argument("--gru-size", type=int, default=96)
parser.add_argument("--epochs", type=int, default=30)
parser.add_argument("--batch-size", type=int, default=16)
parser.add_argument("--lr", type=float, default=5e-4)
parser.add_argument("--seq-len", type=int, default=200)
parser.add_argument("--num-workers", type=int, default=0)
parser.add_argument("--w-vad", type=float, default=0.1)
parser.add_argument("--w-pitch", type=float, default=5.0)
parser.add_argument("--w-gesture", type=float, default=1.5)
parser.add_argument("--w-register", type=float, default=0.0)
parser.add_argument("--w-dynamics", type=float, default=0.0)
parser.add_argument("--pitch-sigma-bins", type=float, default=0.8)
parser.add_argument("--pitch-pos-weight", type=float, default=5.0)
parser.add_argument("--snr-range", type=float, nargs=2, default=[-5.0, 20.0])
parser.add_argument(
    "--gesture-weights",
    choices=("auto", "none"),
    default="auto",
    help="Inverse-frequency class weights for gesture focal loss (capped)",
)
parser.add_argument(
    "--gesture-weight-max-ratio",
    type=float,
    default=2.5,
    help="Cap inverse-frequency class weights (steady/vibrato/transition)",
)
parser.add_argument(
    "--glissando-weight-cap",
    type=float,
    default=6.0,
    help="Higher cap for glissando class weight (rarest gesture)",
)
parser.add_argument("--focal-gamma", type=float, default=1.0)
parser.add_argument(
    "--freeze-backbone-epochs",
    type=int,
    default=20,
    help="Train only dense_gesture; backbone frozen (0=disabled)",
)
parser.add_argument(
    "--unfreeze-lr",
    type=float,
    default=1e-4,
    help="LR after backbone unfreeze",
)
parser.add_argument(
    "--min-steady-recall",
    type=float,
    default=0.45,
    help="Minimum steady-class recall to allow best.pth",
)
parser.add_argument(
    "--min-vibrato-precision",
    type=float,
    default=0.35,
    help="Minimum vibrato precision to allow best.pth",
)
parser.add_argument(
    "--min-vibrato-recall",
    type=float,
    default=0.25,
    help="Alternative best.pth gate: vibrato recall if precision is low early in training",
)
parser.add_argument(
    "--min-rpa-for-best",
    type=float,
    default=0.75,
    help="Minimum val RPA (standard Viterbi) to allow best.pth",
)
parser.add_argument(
    "--eval-every",
    type=int,
    default=5,
    help="Run val gesture metrics every N epochs (for best.pth selection)",
)
parser.add_argument(
    "--eval-max-clips",
    type=int,
    default=0,
    help="Val clips for in-training eval (0 = all clips)",
)


class VocalSetDataset(Dataset):
    def __init__(self, npz_path: str, noise_npz: str | None, seq_len: int = 200):
        self.seq_len = seq_len
        data = np.load(npz_path, allow_pickle=True)
        self.mel = data["mel"].astype(np.float32)
        self.f0 = data["f0"].astype(np.float32)
        self.vad = data["vad"].astype(np.float32)
        self.gesture = data["gesture"].astype(np.int64)
        self.register = data["register"].astype(np.int64)
        self.dynamics = data["dynamics"].astype(np.int64)
        self.lengths = data["lengths"]
        self.segs = self._build_segments(self.lengths, seq_len)
        self.rng = np.random.default_rng()
        # Oversample clips containing rare gestures (glissando / transition).
        extra: list[tuple[int, int]] = []
        for start, end in self.segs:
            g = self.gesture[start:end]
            if np.any(g == 2) or np.any(g == 3):
                extra.append((start, end))
        if extra:
            self.segs = self.segs + extra * 2

        self.noise_mel = None
        self.noise_segs = []
        if noise_npz and os.path.exists(noise_npz):
            noise = np.load(noise_npz)
            self.noise_mel = noise["mel"].astype(np.float32)
            self.noise_segs = self._build_segments(noise["lengths"], seq_len)

        print(f"  {npz_path}: {len(self.segs)} segments")

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

        mel_clean = self.mel[s:e]
        mel_noise = mel_clean.copy()
        if self.noise_mel is not None and self.noise_segs:
            ns, ne = self.noise_segs[self.rng.integers(len(self.noise_segs))]
            n = ns + self.rng.integers(0, ne - ns - self.seq_len + 1)
            mel_noise = self.noise_mel[n:n + self.seq_len]

        return (
            mel_clean,
            mel_noise,
            self.vad[s:e],
            self.f0[s:e],
            self.gesture[s:e],
            self.register[s:e],
            self.dynamics[s:e],
        )


def compute_gesture_class_weights(
    train_npz: str,
    n_classes: int = 4,
    max_ratio: float = 3.0,
    glissando_cap: float = 6.0,
) -> np.ndarray:
    """Inverse-frequency weights, normalized to min=1 with per-class caps."""
    gesture = np.load(train_npz, allow_pickle=True)["gesture"].astype(np.int64)
    counts = np.bincount(gesture.clip(0, n_classes - 1), minlength=n_classes).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    weights = gesture.size / (n_classes * counts)
    weights /= weights.min()
    for i in range(n_classes):
        cap = glissando_cap if i == 2 else max_ratio  # GESTURE_GLISSANDO
        weights[i] = min(weights[i], max(1.0, float(cap)))
    weights /= weights.sum()
    weights *= n_classes
    return weights.astype(np.float32)


def set_gesture_only_phase(model: NanoPitchPlus, gesture_only: bool) -> None:
    """Freeze conv/GRU/VAD/pitch; train dense_gesture only when gesture_only=True."""
    for name, param in model.named_parameters():
        param.requires_grad = name.startswith("dense_gesture") if gesture_only else True


def _gesture_metrics_from_cm(cm: np.ndarray) -> dict:
    n = cm.shape[0]
    recalls, precisions = [], []
    per_recall, per_precision = {}, {}
    for c in range(n):
        true = cm[c, :].sum()
        pred = cm[:, c].sum()
        if true == 0:
            per_recall[GESTURE_VOCAB[c]] = None
        else:
            rec = float(cm[c, c] / true)
            recalls.append(rec)
            per_recall[GESTURE_VOCAB[c]] = rec
        if pred == 0:
            per_precision[GESTURE_VOCAB[c]] = None
        else:
            prec = float(cm[c, c] / pred)
            precisions.append(prec)
            per_precision[GESTURE_VOCAB[c]] = prec
    steady_recall = per_recall.get(GESTURE_VOCAB[0]) or 0.0
    vib_prec = per_precision.get(GESTURE_VOCAB[1])
    vib_rec = per_recall.get(GESTURE_VOCAB[1])
    return {
        "steady_recall": steady_recall,
        "vibrato_precision": vib_prec if vib_prec is not None else 0.0,
        "vibrato_recall": vib_rec if vib_rec is not None else 0.0,
        "balanced_acc": float(np.mean(recalls)) if recalls else 0.0,
        "per_class_recall": per_recall,
        "per_class_precision": per_precision,
    }


def _qualifies_for_best(metrics: dict, args) -> bool:
    vib_prec = metrics.get("vibrato_precision", 0.0)
    vib_rec = metrics.get("vibrato_recall", 0.0)
    return (
        metrics.get("steady_recall", 0.0) >= args.min_steady_recall
        and metrics.get("rpa_std", 0.0) >= args.min_rpa_for_best
        and (
            vib_prec >= args.min_vibrato_precision
            or vib_rec >= args.min_vibrato_recall
        )
    )


def focal_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    gamma: float = 2.0,
    ignore_index: int = -1,
    class_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """Focal loss for gesture classification.

    Down-weights correctly-classified easy frames (mostly "steady") so the
    model's gradient is driven by the harder, rarer vibrato and transition
    examples.  gamma=2.0 is the standard value from the original paper.

    Loss = -(1 - p_t)^gamma * log(p_t),  where p_t = softmax probability of
    the correct class.
    """
    ce = F.cross_entropy(
        logits, targets,
        ignore_index=ignore_index,
        reduction="none",
        weight=class_weight,
    )
    valid = targets != ignore_index
    if not valid.any():
        return torch.zeros((), device=logits.device)
    pt = torch.exp(-ce[valid])
    return ((1.0 - pt) ** gamma * ce[valid]).mean()


def train_one_epoch(model, loader, optimizer, scheduler, writer, epoch, device, args):
    model.train()
    bce_vad = nn.BCEWithLogitsLoss(reduction="none")
    bce_pitch = nn.BCEWithLogitsLoss(
        reduction="none",
        pos_weight=torch.tensor(args.pitch_pos_weight, device=device),
    )
    ce = nn.CrossEntropyLoss(reduction="none", ignore_index=-1)

    running = {k: 0.0 for k in ("loss", "vad", "pitch", "gesture", "register", "dynamics")}
    n_batches = 0
    global_step = (epoch - 1) * len(loader)

    for batch in tqdm(loader, desc=f"Epoch {epoch}"):
        mel_clean, mel_noise, vad_t, f0_t, gest_t, reg_t, dyn_t = [
            x.to(device) for x in batch
        ]
        B, T = mel_clean.shape[:2]

        bin_idx = torch.arange(PITCH_BINS, device=device).view(1, 1, -1)
        f0_safe = f0_t.clamp(min=1e-10)
        bins = (1200.0 * torch.log2(f0_safe / PITCH_FMIN) / PITCH_CENTS_PER_BIN).unsqueeze(-1)
        pitch_target = torch.exp(-0.5 * ((bin_idx - bins) / args.pitch_sigma_bins) ** 2)
        voiced_mask = (f0_t > 0).float().unsqueeze(-1)
        pitch_target = pitch_target * voiced_mask

        mel_mix = augment_mel_batch(mel_clean, mel_noise, args.snr_range, device)

        vad_l, pitch_l, gest_l, reg_l, dyn_l, _ = model(mel_mix, return_logits=True)

        vad_weight = 0.5 * (1.0 - vad_t) + 3.0 * vad_t
        vad_loss = (vad_weight * bce_vad(vad_l.squeeze(-1), vad_t)).mean()
        pitch_loss = (vad_t.unsqueeze(-1) * bce_pitch(pitch_l, pitch_target)).mean()
        gesture_loss = focal_cross_entropy(
            gest_l.reshape(-1, gest_l.shape[-1]),
            gest_t.reshape(-1),
            gamma=args.focal_gamma,
            class_weight=getattr(args, "gesture_class_weight", None),
        )

        reg_mask = reg_t >= 0
        if reg_mask.any():
            register_loss = ce(
                reg_l.reshape(-1, reg_l.shape[-1])[reg_mask.reshape(-1)],
                reg_t.reshape(-1)[reg_mask.reshape(-1)],
            ).mean()
        else:
            register_loss = torch.zeros((), device=device)

        dyn_mask = dyn_t >= 0
        if dyn_mask.any():
            dynamics_loss = ce(
                dyn_l.reshape(-1, dyn_l.shape[-1])[dyn_mask.reshape(-1)],
                dyn_t.reshape(-1)[dyn_mask.reshape(-1)],
            ).mean()
        else:
            dynamics_loss = torch.zeros((), device=device)

        freeze = (
            args.freeze_backbone_epochs > 0
            and epoch <= args.freeze_backbone_epochs
        )
        w_vad = 0.0 if freeze else args.w_vad
        w_pitch = 0.0 if freeze else args.w_pitch
        loss = (
            w_vad * vad_loss
            + w_pitch * pitch_loss
            + args.w_gesture * gesture_loss
            + args.w_register * register_loss
            + args.w_dynamics * dynamics_loss
        )

        optimizer.zero_grad()
        loss.backward()
        trainable = [p for p in model.parameters() if p.requires_grad]
        if trainable:
            torch.nn.utils.clip_grad_norm_(trainable, 5.0)
        optimizer.step()
        scheduler.step()

        running["loss"] += loss.item()
        running["vad"] += vad_loss.item()
        running["pitch"] += pitch_loss.item()
        running["gesture"] += gesture_loss.item()
        running["register"] += register_loss.item()
        running["dynamics"] += dynamics_loss.item()
        n_batches += 1
        global_step += 1
        writer.add_scalar("train/lr", scheduler.get_last_lr()[0], global_step)

    if n_batches == 0:
        return float("nan")
    for k in running:
        writer.add_scalar(f"train/{k}", running[k] / n_batches, epoch)
    return running["loss"] / n_batches


def _macro_f1_from_cm(cm: np.ndarray) -> float:
    f1s = []
    for c in range(cm.shape[0]):
        tp = cm[c, c]
        pred = cm[:, c].sum()
        true = cm[c, :].sum()
        if true == 0:
            continue
        prec = tp / pred if pred else 0.0
        rec = tp / true if true else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        f1s.append(f1)
    return float(np.mean(f1s)) if f1s else 0.0


@torch.no_grad()
def evaluate(model, val_path, device, writer, epoch, max_clips: int = 0) -> dict:
    """Val metrics for checkpoint selection. Returns macro_f1, gesture_acc, rpa."""
    empty = {
        "macro_f1": 0.0, "gesture_acc": 0.0, "rpa_std": 0.0,
        "steady_recall": 0.0, "vibrato_precision": 0.0, "vibrato_recall": 0.0,
        "balanced_acc": 0.0,
    }
    if not os.path.exists(val_path):
        return empty

    data = np.load(val_path, allow_pickle=True)
    lengths = data["lengths"]
    n_clips = len(lengths) if max_clips <= 0 else min(len(lengths), max_clips)
    offset = 0
    n_classes = len(GESTURE_VOCAB)
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    rpa_std, rpa_gest = [], []

    model.eval()
    for clip_i in range(n_clips):
        length = int(lengths[clip_i])
        mel = torch.from_numpy(
            data["mel"][offset:offset + length].astype(np.float32)
        ).unsqueeze(0).to(device)
        f0_gt = data["f0"][offset:offset + length]
        gest_gt = data["gesture"][offset:offset + length]

        _, pitch, gest_logits, _, _, _ = model(mel)
        gest_pred = gest_logits[0].argmax(-1).cpu().numpy()
        for t, p in zip(gest_gt, gest_pred):
            if 0 <= t < n_classes and 0 <= p < n_classes:
                cm[t, p] += 1

        post = pitch[0].cpu().numpy()
        f0_std = viterbi_decode(post)
        f0_gest = viterbi_decode_gesture(post, gest_pred)

        for f0_dec, bucket in ((f0_std, rpa_std), (f0_gest, rpa_gest)):
            both = (f0_gt > 0) & (f0_dec > 0)
            if both.sum() > 0:
                err = np.abs(1200.0 * np.log2(f0_dec[both] / (f0_gt[both] + 1e-10) + 1e-10))
                bucket.append(float(np.mean(err < 50.0)))

        offset += length

    total = cm.sum()
    acc = float(np.trace(cm) / total) if total else 0.0
    macro_f1 = _macro_f1_from_cm(cm)
    counts = cm.sum(axis=1)
    majority = int(counts.argmax()) if counts.sum() else 0
    maj_acc = float(counts[majority] / total) if total else 0.0

    extra = _gesture_metrics_from_cm(cm)
    rpa_mean = float(np.mean(rpa_std)) if rpa_std else 0.0

    writer.add_scalar("eval/gesture_acc", acc, epoch)
    writer.add_scalar("eval/gesture_macro_f1", macro_f1, epoch)
    writer.add_scalar("eval/steady_recall", extra["steady_recall"], epoch)
    writer.add_scalar("eval/balanced_acc", extra["balanced_acc"], epoch)
    writer.add_scalar("eval/rpa_std", rpa_mean, epoch)
    writer.add_scalar("eval/vibrato_precision", extra["vibrato_precision"], epoch)
    print(f"  val clips: {n_clips}  gesture acc: {acc:.1%}  "
          f"(majority {GESTURE_VOCAB[majority]} {maj_acc:.1%})")
    print(f"  gesture macro F1: {macro_f1:.1%}  balanced acc: {extra['balanced_acc']:.1%}")
    print(f"  steady recall: {extra['steady_recall']:.1%}  "
          f"vibrato prec/rec: {extra['vibrato_precision']:.1%} / {extra['vibrato_recall']:.1%}")
    for name, rec in extra["per_class_recall"].items():
        if rec is not None:
            print(f"    recall {name}: {rec:.1%}")
    for name, prec in extra.get("per_class_precision", {}).items():
        if prec is not None:
            print(f"    precision {name}: {prec:.1%}")
    if rpa_std:
        print(f"  RPA (standard Viterbi): {rpa_mean:.1%}")
    if rpa_gest:
        print(f"  RPA (gesture Viterbi):  {np.mean(rpa_gest):.1%}")

    return {
        "macro_f1": macro_f1,
        "gesture_acc": acc,
        "rpa_std": rpa_mean,
        "steady_recall": extra["steady_recall"],
        "vibrato_precision": extra["vibrato_precision"],
        "vibrato_recall": extra["vibrato_recall"],
        "balanced_acc": extra["balanced_acc"],
    }


def _load_model_for_eval_or_train(args, device: torch.device) -> NanoPitchPlus:
    """Load NanoPitchPlus from --resume (Plus ckpt) or base NanoPitch (shared weights only)."""
    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            state = ckpt["state_dict"]
            is_plus = (
                ckpt.get("model_type") == "NanoPitchPlus"
                or "dense_gesture.weight" in state
            )
            if is_plus:
                kwargs = ckpt.get("model_kwargs", {})
                model = NanoPitchPlus(
                    cond_size=kwargs.get("cond_size", args.cond_size),
                    gru_size=kwargs.get("gru_size", args.gru_size),
                )
                model.load_state_dict(state)
                print(f"Loaded NanoPitchPlus from {args.resume}")
                return model.to(device)
        # Base NanoPitch / exp4: shared layers only; gesture/register/dynamics heads stay random
        model = NanoPitchPlus.from_nanopitch_checkpoint(args.resume, device="cpu")
        print(f"Loaded base NanoPitch weights from {args.resume} (Plus heads initialized randomly)")
        return model.to(device)
    return NanoPitchPlus(cond_size=args.cond_size, gru_size=args.gru_size).to(device)


def main():
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
    print(f"Device: {device}")

    output_dir = os.path.abspath(args.output_dir)
    ckpt_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    val_npz = os.path.join(args.data_dir, "val.npz")

    if args.epochs <= 0:
        if not args.resume or not os.path.exists(args.resume):
            print("Eval-only (--epochs 0) requires --resume <checkpoint.pth>")
            return 1
        if not os.path.exists(val_npz):
            print(f"Missing {val_npz}. Run data/vocalset_preprocess.py first.")
            return 1
        print("Eval-only mode (epochs=0). For full val metrics, prefer:")
        print("  python3 model/eval_multitask.py --checkpoint", args.resume)
        model = _load_model_for_eval_or_train(args, device)
        writer = SummaryWriter(log_dir=os.path.join(output_dir, "tb"))
        evaluate(model, val_npz, device, writer, 0, max_clips=args.eval_max_clips)
        writer.close()
        return 0

    train_npz = os.path.join(args.data_dir, "train.npz")
    if not os.path.exists(train_npz):
        print(f"Missing {train_npz}. Run data/vocalset_preprocess.py first.")
        return 1

    noise_npz = os.path.join(args.noise_dir, "noise.npz")
    train_ds = VocalSetDataset(train_npz, noise_npz, args.seq_len)
    loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
        pin_memory=(device.type == "cuda"),
    )

    model = _load_model_for_eval_or_train(args, device)

    if args.gesture_weights == "auto":
        gw = compute_gesture_class_weights(
            train_npz,
            max_ratio=args.gesture_weight_max_ratio,
            glissando_cap=args.glissando_weight_cap,
        )
        args.gesture_class_weight = torch.tensor(gw, device=device)
        print("Gesture class weights (capped):",
              dict(zip(GESTURE_VOCAB, gw.round(3))))
    else:
        args.gesture_class_weight = None

    freeze_epochs = max(0, args.freeze_backbone_epochs)
    if freeze_epochs >= args.epochs:
        print("[WARNING] freeze-backbone-epochs >= total epochs; disabling freeze.")
        freeze_epochs = 0

    writer = SummaryWriter(log_dir=os.path.join(output_dir, "tb"))
    best_macro_f1 = -1.0
    eval_max = args.eval_max_clips
    optimizer = scheduler = None

    for epoch in range(1, args.epochs + 1):
        gesture_only = freeze_epochs > 0 and epoch <= freeze_epochs
        if gesture_only and epoch == 1:
            set_gesture_only_phase(model, True)
            n_steps = len(loader) * freeze_epochs
            optimizer = torch.optim.AdamW(
                [p for p in model.parameters() if p.requires_grad],
                lr=args.lr, betas=(0.8, 0.98), eps=1e-8,
            )
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer, max_lr=args.lr, total_steps=n_steps, pct_start=0.1,
            )
            print(f"Phase 1: gesture head only for epochs 1–{freeze_epochs} "
                  f"(lr={args.lr})")
        elif freeze_epochs > 0 and epoch == freeze_epochs + 1:
            set_gesture_only_phase(model, False)
            n_steps = len(loader) * (args.epochs - freeze_epochs)
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=args.unfreeze_lr, betas=(0.8, 0.98), eps=1e-8,
            )
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer, max_lr=args.unfreeze_lr, total_steps=n_steps, pct_start=0.1,
            )
            print(f"Phase 2: full model epochs {epoch}–{args.epochs} "
                  f"(lr={args.unfreeze_lr})")
        elif freeze_epochs == 0 and epoch == 1:
            set_gesture_only_phase(model, False)
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=args.lr, betas=(0.8, 0.98), eps=1e-8,
            )
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer, max_lr=args.lr,
                total_steps=len(loader) * args.epochs, pct_start=0.1,
            )

        t0 = time.time()
        loss = train_one_epoch(model, loader, optimizer, scheduler, writer, epoch, device, args)
        phase = "gesture-only" if gesture_only else "full"
        print(f"Epoch {epoch} [{phase}]  loss={loss:.5f}  ({time.time()-t0:.1f}s)")

        val_metrics = {
            "macro_f1": 0.0, "gesture_acc": 0.0, "rpa_std": 0.0,
            "steady_recall": 0.0, "vibrato_precision": 0.0, "vibrato_recall": 0.0,
        "balanced_acc": 0.0,
        }
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            val_metrics = evaluate(
                model, val_npz, device, writer, epoch, max_clips=eval_max,
            )

        ckpt = {
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict() if optimizer else {},
            "scheduler": scheduler.state_dict() if scheduler else {},
            "model_kwargs": {"cond_size": args.cond_size, "gru_size": args.gru_size},
            "model_type": "NanoPitchPlus",
            "loss": loss,
            "val_macro_f1": val_metrics["macro_f1"],
            "val_gesture_acc": val_metrics["gesture_acc"],
            "val_steady_recall": val_metrics["steady_recall"],
            "val_vibrato_precision": val_metrics["vibrato_precision"],
            "val_rpa_std": val_metrics["rpa_std"],
        }
        torch.save(ckpt, os.path.join(ckpt_dir, f"epoch_{epoch:03d}.pth"))

        ok = _qualifies_for_best(val_metrics, args)
        if ok and val_metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = val_metrics["macro_f1"]
            torch.save(ckpt, os.path.join(ckpt_dir, "best.pth"))
            print(f"  → new best.pth  macro F1={best_macro_f1:.1%}  "
                  f"steady recall={val_metrics['steady_recall']:.1%}  "
                  f"vibrato prec={val_metrics['vibrato_precision']:.1%}  "
                  f"RPA={val_metrics['rpa_std']:.1%}")
        elif epoch % args.eval_every == 0 or epoch == args.epochs:
            why = []
            if val_metrics["steady_recall"] < args.min_steady_recall:
                why.append(f"steady recall {val_metrics['steady_recall']:.1%} "
                           f"< {args.min_steady_recall:.0%}")
            if (
                val_metrics["vibrato_precision"] < args.min_vibrato_precision
                and val_metrics["vibrato_recall"] < args.min_vibrato_recall
            ):
                why.append(
                    f"vibrato prec/rec {val_metrics['vibrato_precision']:.1%}/"
                    f"{val_metrics['vibrato_recall']:.1%} below "
                    f"{args.min_vibrato_precision:.0%}/{args.min_vibrato_recall:.0%}"
                )
            if val_metrics["rpa_std"] < args.min_rpa_for_best:
                why.append(f"RPA {val_metrics['rpa_std']:.1%} < {args.min_rpa_for_best:.0%}")
            if why:
                print(f"  (skipped best.pth: {'; '.join(why)})")

    writer.close()
    if best_macro_f1 < 0:
        print("Done. No checkpoint met best.pth gates "
              f"(steady recall ≥{args.min_steady_recall:.0%}, "
              f"vibrato precision ≥{args.min_vibrato_precision:.0%}, "
              f"RPA ≥{args.min_rpa_for_best:.0%}).")
        print(f"Pick manually from: {ckpt_dir}")
    else:
        print(f"Done. best.pth macro F1={best_macro_f1:.1%} "
              f"(steady recall ≥{args.min_steady_recall:.0%}, "
              f"vibrato precision ≥{args.min_vibrato_precision:.0%}, "
              f"RPA ≥{args.min_rpa_for_best:.0%}).")
    print(f"Full val: python3 model/eval_multitask.py --checkpoint {ckpt_dir}/best.pth")
    print(f"Checkpoints: {ckpt_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
