#!/usr/bin/env python3
"""
Train BreathCNN on VocalSet-derived breath labels.

Labels come from vocalset_preprocess.py (unvoiced low-energy gaps between
phrases). At inference, detected breath events segment the performance for
per-phrase feedback.

Usage:
    python3 data/vocalset_preprocess.py
    python3 model/train_breath.py --data-dir data/vocalset/processed
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from model.breath_cnn import BreathCNN, DEFAULT_WINDOW


parser = argparse.ArgumentParser(description="BreathCNN trainer")
parser.add_argument("--data-dir", default="data/vocalset/processed")
parser.add_argument("--output-dir", default="runs/breath_cnn")
parser.add_argument("--device", default="auto")
parser.add_argument("--epochs", type=int, default=20)
parser.add_argument("--batch-size", type=int, default=32)
parser.add_argument("--seq-len", type=int, default=200)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--pos-weight", type=float, default=8.0,
                    help="Up-weight rare breath frames")
parser.add_argument("--num-workers", type=int, default=0)


class BreathDataset(Dataset):
    def __init__(self, npz_path: str, seq_len: int = 200):
        data = np.load(npz_path)
        self.windows = data["windows"].astype(np.float32)
        self.labels = data["labels"].astype(np.float32)
        self.lengths = data["lengths"]
        self.seq_len = seq_len
        self.segs = []
        offset = 0
        for length in self.lengths:
            if length >= seq_len:
                self.segs.append((offset, offset + length))
            offset += length
        self.rng = np.random.default_rng()
        print(f"  {npz_path}: {len(self.segs)} clips")

    def __len__(self):
        return max(len(self.segs) * 2, 1)

    def __getitem__(self, idx):
        start, end = self.segs[self.rng.integers(len(self.segs))]
        s = start + self.rng.integers(0, end - start - self.seq_len + 1)
        e = s + self.seq_len
        return (
            torch.from_numpy(self.windows[s:e]),
            torch.from_numpy(self.labels[s:e]),
        )


def main():
    args = parser.parse_args()
    breath_npz = os.path.join(args.data_dir, "breath.npz")
    if not os.path.exists(breath_npz):
        print(f"Missing {breath_npz}. Run data/vocalset_preprocess.py first.")
        return 1

    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    window = int(np.load(breath_npz)["windows"].shape[1])
    ds = BreathDataset(breath_npz, args.seq_len)
    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, drop_last=True,
    )

    model = BreathCNN(window_size=window).to(device)
    bce = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(args.pos_weight, device=device),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, total_steps=len(loader) * args.epochs, pct_start=0.1,
    )

    out_dir = os.path.abspath(args.output_dir)
    ckpt_dir = os.path.join(out_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join(out_dir, "tb"))
    best = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        n = 0
        t0 = time.time()
        for windows, labels in tqdm(loader, desc=f"Epoch {epoch}"):
            windows = windows.to(device)
            labels = labels.to(device).unsqueeze(-1)
            logits = model(windows, return_logits=True)
            loss = bce(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            running += loss.item()
            n += 1

        avg = running / max(n, 1)
        writer.add_scalar("train/loss", avg, epoch)
        print(f"Epoch {epoch}  loss={avg:.5f}  ({time.time()-t0:.1f}s)")

        ckpt = {
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "window_size": window,
            "loss": avg,
        }
        torch.save(ckpt, os.path.join(ckpt_dir, f"epoch_{epoch:03d}.pth"))
        if avg < best:
            best = avg
            torch.save(ckpt, os.path.join(ckpt_dir, "best.pth"))

    writer.close()
    print(f"Done. Best loss: {best:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
