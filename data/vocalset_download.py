#!/usr/bin/env python3
"""
Download Annotated-VocalSet annotations and prepare VocalSet directory layout.

Annotated-VocalSet (411 MB): per-frame F0, amplitude, onset/offset, transition.
  https://doi.org/10.5281/zenodo.7061507

VocalSet audio (~5.6 GB zip) from Zenodo:
  https://doi.org/10.5281/zenodo.1442513

Note: Zenodo's UI says "8.1 GB" but VocalSet1-2.zip is actually ~5.99 GB
(5,991,573,193 bytes). Use --verify-zip to confirm your download via MD5.

Python 3.12+ cannot read this zip with zipfile (overlap protection); extraction
uses the system `unzip` command instead.

Usage:
    python3 data/vocalset_download.py --verify-zip
    python3 data/vocalset_download.py --extract-minimal
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.vocalset_labels import MINIMAL_TRAINING_TECHNIQUES, _norm_technique

ANNOTATED_VOCALSET_URL = (
    "https://zenodo.org/records/7061507/files/Annotated%20VocalSet.zip?download=1"
)
VOCALSET_URL = (
    "https://zenodo.org/records/1442513/files/VocalSet1-2.zip?download=1"
)
# Official VocalSet 1.2 zip (Zenodo record 1442513).
VOCALSET_ZIP_EXPECTED_BYTES = 5_991_573_193
VOCALSET_ZIP_MD5 = "c5e5efab412637fc94972b93c343a2f0"


def parse_args():
    p = argparse.ArgumentParser(description="Download VocalSet annotation assets.")
    p.add_argument("--output-dir", default="data/vocalset",
                   help="Root directory for audio + annotations")
    p.add_argument("--annotations-only", action="store_true",
                   help="Only download Annotated-VocalSet (~411 MB)")
    p.add_argument("--download-audio", action="store_true",
                   help="Download VocalSet1-2.zip (~5.6 GB — slow)")
    p.add_argument(
        "--extract-techniques",
        default=None,
        help="Comma-separated technique folders to extract from VocalSet1-2.zip "
             "(e.g. straight,vibrato,belt,soft,mixed)",
    )
    p.add_argument(
        "--extract-minimal",
        action="store_true",
        help=f"Extract default minimal set: {','.join(MINIMAL_TRAINING_TECHNIQUES)}",
    )
    p.add_argument(
        "--vocalset-zip",
        default=None,
        help="Path to VocalSet1-2.zip (default: <output-dir>/VocalSet1-2.zip)",
    )
    p.add_argument(
        "--verify-zip",
        action="store_true",
        help="Verify VocalSet1-2.zip size + MD5, then exit",
    )
    return p.parse_args()


def _download(url: str, dest: Path) -> None:
    import urllib.request

    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading\n  {url}\n  → {dest}")
    urllib.request.urlretrieve(url, dest)  # noqa: S310
    print(f"  saved ({dest.stat().st_size / 1e9:.2f} GB)")


def _extract_zip(zip_path: Path, dest: Path) -> None:
    """Extract small zips (annotations) with Python zipfile."""
    print(f"Extracting {zip_path.name} → {dest}")
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest)
    print("  done")


def _md5_file(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _verify_vocalset_zip(zip_path: Path, *, check_md5: bool = False) -> None:
    """Raise if VocalSet zip is missing or clearly incomplete."""
    if not zip_path.exists():
        raise FileNotFoundError(
            f"VocalSet zip not found: {zip_path}\n"
            "Download VocalSet1-2.zip from https://doi.org/10.5281/zenodo.1442513\n"
            f"Save it as: {zip_path}"
        )

    size = zip_path.stat().st_size
    size_gb = size / 1e9
    # Allow ±3% around the known release size.
    lo = int(VOCALSET_ZIP_EXPECTED_BYTES * 0.97)
    hi = int(VOCALSET_ZIP_EXPECTED_BYTES * 1.03)
    if not (lo <= size <= hi):
        raise ValueError(
            f"VocalSet zip size unexpected: {size_gb:.2f} GB "
            f"(expected ~{VOCALSET_ZIP_EXPECTED_BYTES / 1e9:.2f} GB).\n"
            "If the download finished, run --verify-zip to check MD5.\n"
            "Otherwise re-download from:\n"
            "  https://zenodo.org/records/1442513/files/VocalSet1-2.zip"
        )

    if check_md5:
        print(f"Computing MD5 for {zip_path.name} …")
        digest = _md5_file(zip_path)
        if digest != VOCALSET_ZIP_MD5:
            raise ValueError(
                f"MD5 mismatch: got {digest}, expected {VOCALSET_ZIP_MD5}.\n"
                "Re-download VocalSet1-2.zip from Zenodo."
            )


def _unzip_available() -> bool:
    return shutil.which("unzip") is not None


def extract_techniques_from_zip(
    zip_path: Path,
    dest: Path,
    techniques: tuple[str, ...] | list[str],
    *,
    resume: bool = True,
) -> int:
    """Extract technique folders using system unzip (Python 3.12+ zipfile fails)."""
    _verify_vocalset_zip(zip_path, check_md5=False)

    if not _unzip_available():
        raise RuntimeError(
            "The `unzip` command is required to extract VocalSet1-2.zip.\n"
            "Install unzip or run on macOS/Linux."
        )

    techniques = tuple(_norm_technique(t) for t in techniques)
    dest = dest.resolve()
    audio_root = dest / "audio" / "by_technique"
    audio_root.mkdir(parents=True, exist_ok=True)
    tmp = dest / ".extract_tmp"

    extracted = 0
    skipped = 0

    print(
        f"Extracting techniques via unzip: {', '.join(techniques)} "
        f"(from {zip_path.name})"
    )
    for tech in techniques:
        out_dir = audio_root / tech
        out_dir.mkdir(parents=True, exist_ok=True)
        existing = {p.name for p in out_dir.glob("*.wav")}
        pattern = f"data_by_technique/{tech}/*.wav"

        if resume and existing:
            skipped += len(existing)

        tmp.mkdir(parents=True, exist_ok=True)
        result = subprocess.run(
            ["unzip", "-o", "-q", str(zip_path), pattern, "-d", str(tmp)],
            capture_output=True,
            text=True,
        )
        # unzip exit 11 = no files matched (technique folder missing from zip)
        if result.returncode == 11:
            print(f"  warning: no files for technique '{tech}' in zip")
        elif result.returncode not in (0, 1):
            raise RuntimeError(
                f"unzip failed for {tech} (exit {result.returncode}):\n"
                f"{result.stderr or result.stdout}"
            )

        src_dir = tmp / "data_by_technique" / tech
        if src_dir.exists():
            for wav in src_dir.glob("*.wav"):
                dst = out_dir / wav.name
                if resume and dst.exists() and dst.stat().st_size > 0:
                    wav.unlink()
                    continue
                shutil.move(str(wav), str(dst))
                extracted += 1

        shutil.rmtree(tmp, ignore_errors=True)

    total = sum(1 for _ in audio_root.rglob("*.wav"))
    print(f"  → {audio_root}  ({total} WAV files, {skipped} skipped as existing)")
    return total


def main() -> int:
    args = parse_args()
    root = Path(args.output_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)

    vocalset_zip = Path(args.vocalset_zip) if args.vocalset_zip else root / "VocalSet1-2.zip"

    if args.verify_zip:
        try:
            _verify_vocalset_zip(vocalset_zip, check_md5=True)
            gb = vocalset_zip.stat().st_size / 1e9
            print(f"OK — {vocalset_zip}")
            print(f"  size: {gb:.2f} GB  ({vocalset_zip.stat().st_size:,} bytes)")
            print(f"  md5:  {VOCALSET_ZIP_MD5}")
            return 0
        except (FileNotFoundError, ValueError) as exc:
            print(exc)
            return 1

    ann_zip = root / "Annotated_VocalSet.zip"
    if not (root / "annotations").exists():
        if not ann_zip.exists():
            _download(ANNOTATED_VOCALSET_URL, ann_zip)
        _extract_zip(ann_zip, root / "annotations")
    else:
        print(f"Annotations already present: {root / 'annotations'}")

    if args.extract_minimal or args.extract_techniques:
        techniques = (
            MINIMAL_TRAINING_TECHNIQUES
            if args.extract_minimal
            else tuple(t.strip() for t in args.extract_techniques.split(",") if t.strip())
        )
        try:
            n = extract_techniques_from_zip(vocalset_zip, root, techniques)
        except (FileNotFoundError, ValueError, RuntimeError) as exc:
            print(f"\n{exc}")
            return 1
        if n == 0:
            return 2

    elif args.download_audio and not args.annotations_only:
        print(
            "Full zip download is large (~5.6 GB). Prefer browser download + "
            "--extract-minimal."
        )
        if not vocalset_zip.exists():
            _download(VOCALSET_URL, vocalset_zip)
        if not (root / "audio").exists():
            extract_techniques_from_zip(
                vocalset_zip, root, MINIMAL_TRAINING_TECHNIQUES,
            )
    elif not (root / "audio").exists() or not any((root / "audio").rglob("*.wav")):
        print("\nVocalSet audio not found.")
        print("  1. Download VocalSet1-2.zip (~5.6 GB):")
        print("     https://zenodo.org/records/1442513/files/VocalSet1-2.zip")
        print(f"  2. Save to: {vocalset_zip}")
        print("  3. python3 data/vocalset_download.py --verify-zip")
        print("  4. python3 data/vocalset_download.py --extract-minimal")

    noise_marker = Path("data/noise.npz")
    if not noise_marker.exists():
        print("\nTip: run  python3 data/download.py  for NanoPitch noise augmentation.")

    print("\nNext step — build training tensors:")
    print(f"  python3 data/vocalset_preprocess.py --root {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
