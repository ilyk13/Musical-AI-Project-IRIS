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
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data.vocalset_labels import DYNAMIC_SILENCE, REGISTER_UNKNOWN
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
parser.add_argument("--w-gesture", type=float, default=1.0)
parser.add_argument("--w-register", type=float, default=0.5)
parser.add_argument("--w-dynamics", type=float, default=0.5)
parser.add_argument("--pitch-sigma-bins", type=float, default=0.8)
parser.add_argument("--pitch-pos-weight", type=float, default=5.0)
parser.add_argument("--snr-range", type=float, nargs=2, default=[-5.0, 20.0])


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
        gesture_loss = ce(gest_l.reshape(-1, gest_l.shape[-1]), gest_t.reshape(-1)).mean()

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

        loss = (
            args.w_vad * vad_loss
            + args.w_pitch * pitch_loss
            + args.w_gesture * gesture_loss
            + args.w_register * register_loss
            + args.w_dynamics * dynamics_loss
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
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


@torch.no_grad()
def evaluate(model, val_path, device, writer, epoch):
    if not os.path.exists(val_path):
        return
    data = np.load(val_path, allow_pickle=True)
    lengths = data["lengths"]
    offset = 0
    gesture_acc, rpa_std, rpa_gest = [], [], []

    model.eval()
    for length in lengths[: min(32, len(lengths))]:
        mel = torch.from_numpy(
            data["mel"][offset:offset + length].astype(np.float32)
        ).unsqueeze(0).to(device)
        f0_gt = data["f0"][offset:offset + length]
        gest_gt = data["gesture"][offset:offset + length]

        vad, pitch, gest_logits, _, _, _ = model(mel)
        gest_pred = gest_logits[0].argmax(-1).cpu().numpy()
        gesture_acc.append(float(np.mean(gest_pred == gest_gt)))

        post = pitch[0].cpu().numpy()
        f0_std = viterbi_decode(post)
        f0_gest = viterbi_decode_gesture(post, gest_pred)

        for f0_dec, bucket in ((f0_std, rpa_std), (f0_gest, rpa_gest)):
            both = (f0_gt > 0) & (f0_dec > 0)
            if both.sum() > 0:
                err = np.abs(1200.0 * np.log2(f0_dec[both] / (f0_gt[both] + 1e-10) + 1e-10))
                bucket.append(float(np.mean(err < 50.0)))

        offset += length

    if gesture_acc:
        ga = float(np.mean(gesture_acc))
        writer.add_scalar("eval/gesture_acc", ga, epoch)
        print(f"  gesture acc: {ga:.1%}")
    if rpa_std:
        print(f"  RPA (standard Viterbi): {np.mean(rpa_std):.1%}")
    if rpa_gest:
        print(f"  RPA (gesture Viterbi):  {np.mean(rpa_gest):.1%}")


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

    train_npz = os.path.join(args.data_dir, "train.npz")
    if not os.path.exists(train_npz):
        print(f"Missing {train_npz}. Run data/vocalset_preprocess.py first.")
        return 1

    output_dir = os.path.abspath(args.output_dir)
    ckpt_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

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

    if args.resume and os.path.exists(args.resume):
        model = NanoPitchPlus.from_nanopitch_checkpoint(args.resume, device="cpu").to(device)
        print(f"Fine-tuning from {args.resume}")
    else:
        model = NanoPitchPlus(cond_size=args.cond_size, gru_size=args.gru_size).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.8, 0.98), eps=1e-8)
    total_steps = len(loader) * args.epochs
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, total_steps=total_steps, pct_start=0.1,
    )
    writer = SummaryWriter(log_dir=os.path.join(output_dir, "tb"))
    best_loss = float("inf")

    if args.epochs <= 0:
        print("Eval-only mode (epochs=0)")
        evaluate(model, os.path.join(args.data_dir, "val.npz"), device, writer, 0)
        writer.close()
        return 0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        loss = train_one_epoch(model, loader, optimizer, scheduler, writer, epoch, device, args)
        print(f"Epoch {epoch}  loss={loss:.5f}  ({time.time()-t0:.1f}s)")
        if epoch % 5 == 0:
            evaluate(model, os.path.join(args.data_dir, "val.npz"), device, writer, epoch)

        ckpt = {
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "model_kwargs": {"cond_size": args.cond_size, "gru_size": args.gru_size},
            "model_type": "NanoPitchPlus",
            "loss": loss,
        }
        torch.save(ckpt, os.path.join(ckpt_dir, f"epoch_{epoch:03d}.pth"))
        if loss < best_loss:
            best_loss = loss
            torch.save(ckpt, os.path.join(ckpt_dir, "best.pth"))
            print(f"  → new best ({best_loss:.5f})")

    writer.close()
    print(f"Done. Checkpoints: {ckpt_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
