"""
NanoPitch Training Script
=========================

Trains the NanoPitch model to track pitch and detect voice activity.

Pipeline:
  1. Load pre-extracted mel spectrograms + RMVPE ground-truth f0 from .npz files
  2. Augment: mix clean vocal mel with random noise mel at a random SNR
  3. Predict: model outputs VAD probability + 360-bin pitch posteriorgram
  4. Loss: weighted BCE for VAD + voiced-weighted BCE for pitch posteriorgram
  5. Evaluate every 5 epochs on held-out noisy clips

Usage:
    # Download data first:
    python3 data/download.py

    # Train on CPU:
    python3 model/train.py --data-dir data/ --output-dir runs/exp1

    # Train on Apple Silicon GPU:
    python3 model/train.py --data-dir data/ --output-dir runs/exp1 --device mps

    # Resume:
    python3 model/train.py --data-dir data/ --output-dir runs/exp1 --resume runs/exp1/checkpoints/best.pth

    # Monitor:
    tensorboard --logdir runs/exp1/tb
"""

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

# Add the project root (IRIS/) to the path so we can import model.nanopitch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from model.nanopitch import (
    NanoPitch, viterbi_decode,
    PITCH_BINS, N_MELS, PITCH_FMIN, PITCH_CENTS_PER_BIN,
)


# ── Arguments ─────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="NanoPitch trainer")

parser.add_argument("--data-dir",    default="data",         help="folder with clean.npz / noise.npz / test.npz")
parser.add_argument("--output-dir",  default="runs/default", help="where to save checkpoints and TensorBoard logs")
parser.add_argument("--resume",      default=None,           help="path to checkpoint to resume from")
parser.add_argument("--device",      default="auto",         help="cpu | cuda | mps | auto")

# Model size
parser.add_argument("--cond-size",   type=int,   default=64)
parser.add_argument("--gru-size",    type=int,   default=96)

# Training
parser.add_argument("--epochs",      type=int,   default=50)
parser.add_argument("--batch-size",  type=int,   default=32)
parser.add_argument("--lr",          type=float, default=1e-3)
parser.add_argument("--seq-len",     type=int,   default=200,  help="frames per training clip (200 = 2 s)")
parser.add_argument("--num-workers", type=int,   default=0)

# Loss weights
parser.add_argument("--w-vad",       type=float, default=0.1)
parser.add_argument("--w-pitch",     type=float, default=5.0)

# Pitch target shaping
parser.add_argument("--pitch-sigma-bins",  type=float, default=0.8,
                    help="Gaussian width of the soft pitch target in bins (20 cents/bin)")
parser.add_argument("--pitch-pos-weight",  type=float, default=5.0,
                    help="BCE pos_weight for pitch — up-weights the small number of near-peak bins")

# Data augmentation
parser.add_argument("--snr-range",   type=float, nargs=2, default=[-5.0, 20.0],
                    help="min/max SNR in dB for noise mixing")


# ── Dataset ────────────────────────────────────────────────────────────
class NanoPitchDataset(Dataset):
    """Serves (clean_mel, noise_mel, vad, f0) tuples.

    Noise mixing is done in augment_mel_batch in the training loop so each
    epoch sees different random SNR mixtures.
    """

    def __init__(self, data_dir: str, seq_len: int = 200):
        self.seq_len = seq_len

        print("Loading clean.npz …")
        clean = np.load(os.path.join(data_dir, "clean.npz"))
        self.clean_mel     = clean["mel"]      # (total_frames, 40)
        self.clean_f0      = clean["f0"]       # (total_frames,)  Hz
        self.clean_vad     = clean["vad"]      # (total_frames,)  0/1
        self.clean_lengths = clean["lengths"]  # frames per clip

        print("Loading noise.npz …")
        noise = np.load(os.path.join(data_dir, "noise.npz"))
        self.noise_mel     = noise["mel"]
        self.noise_lengths = noise["lengths"]

        self.clean_segs = self._build_segments(self.clean_lengths, seq_len)
        self.noise_segs = self._build_segments(self.noise_lengths, seq_len)

        print(f"  Clean : {len(self.clean_mel):,} frames, {len(self.clean_segs)} usable segments")
        print(f"  Noise : {len(self.noise_mel):,} frames, {len(self.noise_segs)} usable segments")

        self.rng = np.random.default_rng()

    def _build_segments(self, lengths, min_len):
        segs, offset = [], 0
        for length in lengths:
            if length >= min_len:
                segs.append((offset, offset + length))
            offset += length
        return segs

    def __len__(self):
        return min(len(self.clean_segs) * 3, 10_000)

    def __getitem__(self, idx):
        # Random clean window
        start, end = self.clean_segs[self.rng.integers(len(self.clean_segs))]
        s = start + self.rng.integers(0, end - start - self.seq_len + 1)
        mel_clean = self.clean_mel[s : s + self.seq_len].astype(np.float32)
        f0        = self.clean_f0 [s : s + self.seq_len].astype(np.float32)
        vad       = self.clean_vad[s : s + self.seq_len].astype(np.float32)

        # Independent random noise window
        ns, ne = self.noise_segs[self.rng.integers(len(self.noise_segs))]
        n = ns + self.rng.integers(0, ne - ns - self.seq_len + 1)
        mel_noise = self.noise_mel[n : n + self.seq_len].astype(np.float32)

        return mel_clean, mel_noise, vad, f0


# ── Augmentation ───────────────────────────────────────────────────────
def augment_mel_batch(
    mel_clean: torch.Tensor,
    mel_noise: torch.Tensor,
    snr_range: tuple[float, float],
    device: torch.device,
) -> torch.Tensor:
    """Mix clean and noise log-mel at a random per-example SNR."""
    B, T, F = mel_clean.shape
    snr_db  = torch.empty(B, 1, 1, device=device).uniform_(*snr_range)

    # Calibrate gain so realised SNR matches requested SNR regardless of
    # the absolute level of the two spectrograms.  Factor of 10 (not 20)
    # because mel stores log-power, not log-amplitude.
    p_clean = torch.logsumexp(mel_clean, dim=(1, 2), keepdim=True) - np.log(T * F)
    p_noise = torch.logsumexp(mel_noise, dim=(1, 2), keepdim=True) - np.log(T * F)
    gain    = p_clean - p_noise - snr_db * (np.log(10.0) / 10.0)

    return torch.logaddexp(mel_clean, mel_noise + gain)


# ── Training epoch ─────────────────────────────────────────────────────
def train_one_epoch(model, dataloader, optimizer, scheduler, writer,
                    epoch, device, args):
    model.train()

    # BCEWithLogitsLoss is numerically more stable than sigmoid + BCE.
    # We pass return_logits=True to the model so it skips the final sigmoid.
    bce_vad   = nn.BCEWithLogitsLoss(reduction="none")
    bce_pitch = nn.BCEWithLogitsLoss(
        reduction="none",
        pos_weight=torch.tensor(args.pitch_pos_weight, device=device),
    )

    running = {"loss": 0.0, "vad": 0.0, "pitch": 0.0}
    n_batches = 0
    global_step = (epoch - 1) * len(dataloader)

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}", unit="batch")
    for mel_clean, mel_noise, vad_target, f0_target in pbar:
        mel_clean  = mel_clean.to(device)
        mel_noise  = mel_noise.to(device)
        vad_target = vad_target.to(device)
        f0_target  = f0_target.to(device)

        B, T = mel_clean.shape[:2]

        # ── Build soft pitch target (vectorised, on-device) ──────────
        # Each voiced frame gets a Gaussian bump centred on its true bin.
        # Unvoiced frames (f0 == 0) stay all-zero.
        bin_idx = torch.arange(PITCH_BINS, device=device).view(1, 1, -1)  # (1,1,360)
        f0_safe = f0_target.clamp(min=1e-10)
        bins    = (1200.0 * torch.log2(f0_safe / PITCH_FMIN)
                   / PITCH_CENTS_PER_BIN).unsqueeze(-1)                    # (B,T,1)
        pitch_target = torch.exp(-0.5 * ((bin_idx - bins) / args.pitch_sigma_bins) ** 2)
        voiced_mask  = (f0_target > 0).float().unsqueeze(-1)               # (B,T,1)
        pitch_target = pitch_target * voiced_mask                           # (B,T,360)

        # ── Augment ──────────────────────────────────────────────────
        mel_mix = augment_mel_batch(mel_clean, mel_noise, args.snr_range, device)

        # ── Forward ──────────────────────────────────────────────────
        pred_vad, pred_pitch, _ = model(mel_mix, return_logits=True)
        # pred_vad:   (B, T, 1)
        # pred_pitch: (B, T, 360)

        # ── Loss ─────────────────────────────────────────────────────
        # VAD: weight voiced frames more heavily (3× vs 0.5×) because
        # voiced frames are rarer and more important to get right.
        vad_weight = 0.5 * (1.0 - vad_target) + 3.0 * vad_target
        vad_loss   = (vad_weight * bce_vad(pred_vad.squeeze(-1), vad_target)).mean()

        # Pitch: only penalise on voiced frames.
        voiced_weight = vad_target.unsqueeze(-1)
        pitch_loss    = (voiced_weight * bce_pitch(pred_pitch, pitch_target)).mean()

        loss = args.w_vad * vad_loss + args.w_pitch * pitch_loss

        # ── Backward ─────────────────────────────────────────────────
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        scheduler.step()   # OneCycleLR steps per batch, not per epoch

        # ── Logging ──────────────────────────────────────────────────
        running["loss"]  += loss.item()
        running["vad"]   += vad_loss.item()
        running["pitch"] += pitch_loss.item()
        n_batches += 1
        global_step += 1

        writer.add_scalar("train/lr", scheduler.get_last_lr()[0], global_step)

        pbar.set_postfix(
            loss=f"{running['loss']/n_batches:.4f}",
            vad=f"{running['vad']/n_batches:.4f}",
            pitch=f"{running['pitch']/n_batches:.4f}",
        )

    if n_batches == 0:
        warnings.warn("No batches were processed this epoch.", RuntimeWarning)
        return float("nan")

    for key in running:
        writer.add_scalar(f"train/{key}", running[key] / n_batches, epoch)

    return running["loss"] / n_batches


# ── Evaluation ─────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(model, data_dir, writer, epoch, device):
    model.eval()
    test_path = os.path.join(data_dir, "test.npz")
    if not os.path.exists(test_path):
        print("  [eval] test.npz not found, skipping")
        return

    test  = np.load(test_path)
    clips = test["clips"]   # (N, T, 40)
    f0r   = test["f0"]      # (N, T)
    vadr  = test["vad"]     # (N, T)
    snrs  = test["snr"]     # (N,)

    results_by_snr: dict = {}
    for i in range(len(clips)):
        mel   = torch.from_numpy(clips[i].astype(np.float32)).unsqueeze(0).to(device)
        v, p, _ = model(mel)
        vad_pred = v[0, :, 0].cpu().numpy()
        post     = p[0].cpu().numpy()
        T        = vad_pred.shape[0]

        f0_gt  = f0r [i, :T].astype(np.float32)
        vad_gt = vadr[i, :T].astype(np.float32)
        f0_dec = viterbi_decode(post)

        # VAD accuracy
        vacc = float(np.mean((vad_pred > 0.5) == (vad_gt > 0.5)))

        # Voicing Detection Rate (VDR): % of truly voiced frames detected as voiced
        truly_voiced = f0_gt > 0
        vdr = float(np.mean(f0_dec[truly_voiced] > 0)) if truly_voiced.sum() > 0 else float("nan")

        # Raw Pitch Accuracy (RPA): % of co-voiced frames within 50 cents
        both = (f0_gt > 0) & (f0_dec > 0)
        if both.sum() > 0:
            cent_err = np.abs(1200.0 * np.log2(f0_dec[both] / (f0_gt[both] + 1e-10) + 1e-10))
            rpa = float(np.mean(cent_err < 50.0))
        else:
            rpa = float("nan")

        snr_key = float(snrs[i])
        results_by_snr.setdefault(snr_key, []).append({"vacc": vacc, "vdr": vdr, "rpa": rpa})

    def smean(vals):
        clean_vals = [v for v in vals if not np.isnan(v)]
        return np.mean(clean_vals) if clean_vals else float("nan")

    print(f"\n  {'Condition':<10}  {'VAD Acc':>8}  {'VDR':>8}  {'RPA':>8}")
    print(f"  {'─'*10}  {'─'*8}  {'─'*8}  {'─'*8}")
    for snr in sorted(results_by_snr, key=lambda x: x if np.isfinite(x) else 1e6):
        rows = results_by_snr[snr]
        tag  = "clean" if not np.isfinite(snr) else f"{snr:+.0f} dB"
        va   = smean([r["vacc"] for r in rows])
        vd   = smean([r["vdr"]  for r in rows])
        rp   = smean([r["rpa"]  for r in rows])
        print(f"  {tag:<10}  {va:8.1%}  {vd:8.1%}  {rp:8.1%}")

        if writer:
            stag = tag.replace(" ", "").replace("+", "p").replace("-", "n")
            writer.add_scalar(f"eval/vad_acc_{stag}", va, epoch)
            writer.add_scalar(f"eval/vdr_{stag}",     vd, epoch)
            writer.add_scalar(f"eval/rpa_{stag}",     rp, epoch)
    print()


# ── Main ───────────────────────────────────────────────────────────────
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
    ckpt_dir   = os.path.join(output_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    model       = NanoPitch(cond_size=args.cond_size, gru_size=args.gru_size).to(device)
    start_epoch = 1
    resume_ckpt = None

    if args.resume:
        resume_ckpt = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(resume_ckpt["state_dict"])
        start_epoch = resume_ckpt.get("epoch", 0) + 1
        print(f"Resumed from epoch {start_epoch - 1}")

    dataset    = NanoPitchDataset(args.data_dir, seq_len=args.seq_len)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
        pin_memory=(device.type == "cuda"),
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, betas=(0.8, 0.98), eps=1e-8
    )

    # OneCycleLR must be stepped once per BATCH (inside train_one_epoch).
    # total_steps = batches_per_epoch × epochs, accounting for resumed epochs.
    total_steps = len(dataloader) * args.epochs
    scheduler   = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr,
        total_steps=total_steps,
        pct_start=0.1,
    )

    # Fast-forward scheduler state when resuming
    if resume_ckpt and "scheduler" in resume_ckpt:
        scheduler.load_state_dict(resume_ckpt["scheduler"])
    if resume_ckpt and "optimizer" in resume_ckpt:
        optimizer.load_state_dict(resume_ckpt["optimizer"])

    writer     = SummaryWriter(log_dir=os.path.join(output_dir, "tb"))
    best_loss  = float("inf")

    for epoch in range(start_epoch, start_epoch + args.epochs):
        t0        = time.time()
        train_loss = train_one_epoch(
            model, dataloader, optimizer, scheduler, writer, epoch, device, args
        )
        print(f"  Epoch {epoch}  loss={train_loss:.5f}  ({time.time()-t0:.1f}s)")

        if epoch % 5 == 0 or epoch == start_epoch:
            evaluate(model, args.data_dir, writer, epoch, device)

        ckpt = {
            "epoch":       epoch,
            "state_dict":  model.state_dict(),
            "optimizer":   optimizer.state_dict(),
            "scheduler":   scheduler.state_dict(),
            "model_kwargs": {"cond_size": args.cond_size, "gru_size": args.gru_size},
            "loss":        train_loss,
        }
        torch.save(ckpt, os.path.join(ckpt_dir, f"epoch_{epoch:03d}.pth"))
        if train_loss < best_loss:
            best_loss = train_loss
            torch.save(ckpt, os.path.join(ckpt_dir, "best.pth"))
            print(f"  → new best ({best_loss:.5f}), saved best.pth")

    writer.close()
    print(f"\nDone. Best loss: {best_loss:.5f}")
    print(f"Checkpoints: {ckpt_dir}")
    print(f"\nTo use the trained model in the web app, point NanoPitchExtractor")
    print(f"at the checkpoint:  NanoPitchExtractor.from_pretrained(local_path='{ckpt_dir}/best.pth')")


if __name__ == "__main__":
    main()
