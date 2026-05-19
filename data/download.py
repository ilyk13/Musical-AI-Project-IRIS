#!/usr/bin/env python3
"""
Download NanoPitch pre-extracted data from Hugging Face.

Source:  https://huggingface.co/datasets/smulelabs/NanoPitch-PreExtract

The dataset contains three .npz files:
  clean.npz  — mel spectrograms of clean singing + RMVPE ground-truth f0 + VAD
  noise.npz  — mel spectrograms of environmental noise (for augmentation)
  test.npz   — held-out noisy clips at multiple SNR levels for evaluation

Usage:
    python3 data/download.py
    python3 data/download.py --output-dir data/raw
"""

import argparse
import sys
from pathlib import Path

DEFAULT_REPO_ID = "smulelabs/NanoPitch-PreExtract"
REQUIRED_FILES  = ("clean.npz", "noise.npz", "test.npz")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download NanoPitch pre-extracted dataset from Hugging Face."
    )
    parser.add_argument("--repo-id",    default=DEFAULT_REPO_ID)
    parser.add_argument("--revision",   default="main")
    parser.add_argument("--output-dir", default="data",
                        help="Local directory for downloaded files (default: data/)")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("Missing dependency: huggingface_hub\n"
              "Install it with:  pip install huggingface_hub")
        return 1

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading '{args.repo_id}'  →  {output_dir}")
    snapshot_download(
        repo_id=args.repo_id,
        repo_type="dataset",
        revision=args.revision,
        local_dir=str(output_dir),
        allow_patterns=["*.npz"],
    )

    missing = [f for f in REQUIRED_FILES if not (output_dir / f).exists()]
    if missing:
        print("Download finished but some required files are missing:")
        for f in missing:
            print(f"  - {f}")
        return 2

    print("\nDownload complete:")
    for f in REQUIRED_FILES:
        size_mb = (output_dir / f).stat().st_size / (1024 * 1024)
        print(f"  {f}  ({size_mb:.1f} MB)")

    print("\nNext step — train the model:")
    print("  python3 model/train.py --data-dir data/ --output-dir runs/exp1")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
