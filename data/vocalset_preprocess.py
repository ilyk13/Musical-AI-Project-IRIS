#!/usr/bin/env python3
"""
Build multi-label training tensors from VocalSet + Annotated-VocalSet.

Outputs (in --output-dir, default data/vocalset/processed):
  train.npz   — mel, f0, vad, gesture, register, dynamics, lengths, file_ids
  val.npz     — held-out singers (from VocalSet train/test split when available)
  breath.npz  — raw waveform windows + breath labels for BreathCNN

Frame alignment: 16 kHz, 10 ms hop (160 samples), HTK mel 0–8 kHz — matches NanoPitch.

Usage:
    python3 data/vocalset_download.py
    # … place VocalSet audio in data/vocalset/audio …
    python3 data/vocalset_preprocess.py
    python3 model/train_multitask.py --data-dir data/vocalset/processed
    python3 model/train_breath.py --data-dir data/vocalset/processed
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import librosa
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.vocalset_labels import (
    DYNAMIC_SILENCE,
    MINIMAL_TRAINING_TECHNIQUES,
    REGISTER_UNKNOWN,
    _norm_technique,
    amplitude_to_dynamic,
    detect_breath_frames,
    label_gestures,
    register_from_technique,
    technique_from_path,
)
from features.pitch.nanopitch import _compute_mel

SR = 16000
HOP = 160
BREATH_WINDOW = 480  # 30 ms context for BreathCNN


def parse_args():
    p = argparse.ArgumentParser(description="Preprocess VocalSet for NanoPitch+ training.")
    p.add_argument("--root", default="data/vocalset",
                   help="Root with audio/ and annotations/ subdirs")
    p.add_argument("--output-dir", default=None,
                   help="Defaults to <root>/processed")
    p.add_argument("--val-ratio", type=float, default=0.15)
    p.add_argument("--max-files", type=int, default=0,
                   help="Limit files (0 = all) for quick tests")
    p.add_argument(
        "--techniques",
        default=None,
        help="Comma-separated technique folders to include "
             f"(default minimal: {','.join(MINIMAL_TRAINING_TECHNIQUES)})",
    )
    p.add_argument("--breath-context", type=int, default=BREATH_WINDOW,
                   help="Waveform samples per breath frame")
    return p.parse_args()


def _find_audio_root(root: Path) -> Path | None:
    candidates = [
        root / "audio" / "by_technique",
        root / "audio" / "VocalSet1-2" / "by_technique",
        root / "audio" / "by_singer",
        root / "audio",
    ]
    for c in candidates:
        if c.exists() and any(c.rglob("*.wav")):
            return c
    return None


def _find_annotation_root(root: Path) -> Path | None:
    ann = root / "annotations"
    if not ann.exists():
        return None
    if any(ann.rglob("*.csv")):
        return ann
    return None


def _stem(path: Path) -> str:
    return path.stem.lower()


def _match_csv(wav_path: Path, csv_files: dict[str, Path]) -> Path | None:
    key = _stem(wav_path)
    if key in csv_files:
        return csv_files[key]
    for stem, path in csv_files.items():
        if stem.startswith(key) or key.startswith(stem):
            return path
    return None


def _csv_priority(path: Path) -> tuple[int, str]:
    """Prefer raw 1 per-frame CSVs over extended note annotations."""
    s = str(path).lower()
    if "extended" in s:
        return (100, s)
    if "raw 1" in s and "/csv/" in s.replace("\\", "/"):
        return (0, s)
    if "raw 2" in s and "/csv/" in s.replace("\\", "/"):
        return (1, s)
    if "raw 3" in s and "/csv/" in s.replace("\\", "/"):
        return (2, s)
    if "raw 4" in s and "/csv/" in s.replace("\\", "/"):
        return (3, s)
    if "raw" in s and "/csv/" in s.replace("\\", "/"):
        return (4, s)
    return (50, s)


def _build_csv_index(ann_root: Path) -> dict[str, Path]:
    """Map wav stem → best available per-frame raw CSV."""
    all_csvs = list(ann_root.rglob("*.csv"))
    ranked = sorted(all_csvs, key=_csv_priority)
    index: dict[str, Path] = {}
    for path in ranked:
        stem = _stem(path)
        if stem not in index:
            index[stem] = path
    return index


def _column_key(rows: list[dict], *names: str) -> str | None:
    """Find a DictReader column key by substring match (handles padded headers)."""
    if not rows:
        return None
    keys = {k.strip().lower(): k for k in rows[0].keys()}
    for name in names:
        name = name.lower()
        if name in keys:
            return keys[name]
        for k, orig in keys.items():
            if name in k:
                return orig
    return None


def _parse_float(val) -> float:
    s = str(val).strip()
    if not s:
        return 0.0
    return float(s)


def _load_raw_csv(csv_path: Path) -> dict:
    with csv_path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        raise ValueError(f"empty CSV: {csv_path}")

    time_key = _column_key(rows, "time")
    f0_key = _column_key(rows, "f0")
    amp_key = _column_key(rows, "amplitude")
    trans_key = _column_key(rows, "transition")
    if not time_key or not f0_key:
        raise ValueError(
            f"not a per-frame raw CSV (missing Time/F0 columns): {csv_path}"
        )

    times = np.array([_parse_float(row.get(time_key, 0)) for row in rows], dtype=np.float64)
    f0 = np.array([_parse_float(row.get(f0_key, 0)) for row in rows], dtype=np.float32)
    amp = np.array(
        [_parse_float(row.get(amp_key, 0)) if amp_key else 0.0 for row in rows],
        dtype=np.float32,
    )
    transition = [
        row.get(trans_key, "") if trans_key else "" for row in rows
    ]
    return {"times": times, "f0": f0, "amp": amp, "transition": transition}


def _align_csv_to_mel(
    mel_frames: int,
    csv_data: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list, np.ndarray]:
    """Resample CSV rows to mel frame grid (10 ms hop)."""
    hop_sec = HOP / SR
    frame_times = np.arange(mel_frames) * hop_sec

    f0 = np.interp(
        frame_times,
        csv_data["times"],
        csv_data["f0"],
        left=0.0,
        right=0.0,
    ).astype(np.float32)
    f0[f0 < 20.0] = 0.0

    amp = np.interp(
        frame_times,
        csv_data["times"],
        csv_data["amp"],
        left=0.0,
        right=0.0,
    ).astype(np.float32)

    # Nearest-neighbour for boolean columns.
    idx = np.searchsorted(csv_data["times"], frame_times, side="left")
    idx = np.clip(idx, 0, len(csv_data["transition"]) - 1)
    transition = [csv_data["transition"][i] for i in idx]

    vad = (f0 > 0).astype(np.float32)
    dynamics = np.array([amplitude_to_dynamic(a) for a in amp], dtype=np.int8)
    return f0, vad, amp, transition, dynamics


def _load_val_singers(root: Path) -> set[str]:
    """Use VocalSet's published test singer list when available."""
    for rel in (
        "audio/test_singers_technique.txt",
        "audio/VocalSet1-2/test_singers_technique.txt",
        "test_singers_technique.txt",
    ):
        path = root / rel
        if path.exists():
            singers = {line.strip().lower() for line in path.read_text().splitlines() if line.strip()}
            print(f"  val singers from {path}: {len(singers)}")
            return singers
    return set()


def _singer_from_path(wav_path: Path) -> str:
    name = wav_path.name.lower()
    # VocalSet pattern: {gender}{id}_{take}_... e.g. f1_01_arpeggio...
    parts = name.split("_")
    return parts[0] if parts else name


def process_file(
    wav_path: Path,
    csv_path: Path,
    breath_context: int,
) -> dict | None:
    try:
        audio, _ = librosa.load(wav_path, sr=SR, mono=True)
    except Exception as exc:
        print(f"  skip {wav_path.name}: load failed ({exc})")
        return None

    try:
        csv_data = _load_raw_csv(csv_path)
    except Exception as exc:
        print(f"  skip {wav_path.name}: CSV failed ({exc})")
        return None

    mel = _compute_mel(audio, SR)
    n = mel.shape[0]
    if n < 10:
        return None

    f0, vad, amp, transition, dynamics = _align_csv_to_mel(n, csv_data)
    technique = technique_from_path(str(wav_path))
    gesture = label_gestures(f0, transition, technique=technique)
    reg = register_from_technique(technique)
    register = np.full(n, reg if reg >= 0 else REGISTER_UNKNOWN, dtype=np.int8)

    breath = detect_breath_frames(audio, f0, HOP)

    # Waveform windows centred on each frame for BreathCNN.
    half = breath_context // 2
    windows = np.zeros((n, breath_context), dtype=np.float32)
    for t in range(n):
        centre = t * HOP + HOP // 2
        s = centre - half
        e = s + breath_context
        seg = np.zeros(breath_context, dtype=np.float32)
        a0 = max(0, s)
        a1 = min(len(audio), e)
        b0 = a0 - s
        b1 = b0 + (a1 - a0)
        if a1 > a0:
            seg[b0:b1] = audio[a0:a1]
        windows[t] = seg

    return {
        "mel": mel,
        "f0": f0,
        "vad": vad,
        "gesture": gesture,
        "register": register,
        "dynamics": dynamics,
        "breath": breath[:n],
        "breath_windows": windows,
        "technique": technique or "",
        "singer": _singer_from_path(wav_path),
        "file_id": wav_path.stem,
    }


def _stack_clips(clips: list[dict]) -> dict:
    keys = ["mel", "f0", "vad", "gesture", "register", "dynamics", "breath", "breath_windows"]
    out = {k: [] for k in keys}
    lengths, file_ids, techniques, singers = [], [], [], []

    for clip in clips:
        for k in keys:
            out[k].append(clip[k])
        lengths.append(len(clip["f0"]))
        file_ids.append(clip["file_id"])
        techniques.append(clip["technique"])
        singers.append(clip["singer"])

    stacked = {k: np.concatenate(v, axis=0) for k, v in out.items()}
    stacked["lengths"] = np.array(lengths, dtype=np.int32)
    stacked["file_ids"] = np.array(file_ids, dtype=object)
    stacked["techniques"] = np.array(techniques, dtype=object)
    stacked["singers"] = np.array(singers, dtype=object)
    return stacked


def _save_npz(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **data)
    mb = path.stat().st_size / 1e6
    print(f"  wrote {path}  ({mb:.1f} MB, {len(data['lengths'])} clips)")


def _filter_by_technique(wav_files: list[Path], techniques: set[str]) -> list[Path]:
    out = []
    for path in wav_files:
        tech = technique_from_path(str(path))
        if tech and _norm_technique(tech) in techniques:
            out.append(path)
    return out


def main() -> int:
    args = parse_args()
    root = Path(args.root).resolve()
    out_dir = Path(args.output_dir or root / "processed").resolve()

    audio_root = _find_audio_root(root)
    ann_root = _find_annotation_root(root)
    if audio_root is None:
        print("No VocalSet audio found. Run vocalset_download.py --download-audio")
        print(f"  or extract WAV files under {root / 'audio'}")
        return 1
    if ann_root is None:
        print("No Annotated-VocalSet CSVs found. Run vocalset_download.py first.")
        return 1

    print(f"Audio:       {audio_root}")
    print(f"Annotations: {ann_root}")
    print(f"Output:      {out_dir}")

    csv_files = _build_csv_index(ann_root)
    print(f"  annotation CSVs indexed: {len(csv_files)}")
    wav_files = sorted(audio_root.rglob("*.wav"))

    if args.techniques:
        techniques = {_norm_technique(t) for t in args.techniques.split(",") if t.strip()}
    else:
        techniques = set(MINIMAL_TRAINING_TECHNIQUES)
    wav_files = _filter_by_technique(wav_files, techniques)
    print(f"  techniques: {', '.join(sorted(techniques))}  →  {len(wav_files)} WAV files")

    if args.max_files:
        wav_files = wav_files[: args.max_files]

    val_singers = _load_val_singers(root)
    train_clips, val_clips = [], []

    matched = 0
    for wav_path in tqdm(wav_files, desc="Processing"):
        csv_path = _match_csv(wav_path, csv_files)
        if csv_path is None:
            continue
        matched += 1
        clip = process_file(wav_path, csv_path, args.breath_context)
        if clip is None:
            continue
        if clip["singer"] in val_singers:
            val_clips.append(clip)
        else:
            train_clips.append(clip)

    if not train_clips:
        print(f"No matched clips (found {matched} CSV pairs). Check paths.")
        return 2

    # Fallback val split if test list missing.
    if not val_clips and args.val_ratio > 0:
        n_val = max(1, int(len(train_clips) * args.val_ratio))
        val_clips = train_clips[-n_val:]
        train_clips = train_clips[:-n_val]

    train_data = _stack_clips(train_clips)
    _save_npz(out_dir / "train.npz", train_data)

    if val_clips:
        val_data = _stack_clips(val_clips)
        _save_npz(out_dir / "val.npz", val_data)

    breath_data = {
        "windows": train_data["breath_windows"],
        "labels": train_data["breath"],
        "lengths": train_data["lengths"],
    }
    _save_npz(out_dir / "breath.npz", breath_data)

    meta = {
        "sr": SR,
        "hop": HOP,
        "breath_context": args.breath_context,
        "techniques": sorted(techniques),
        "n_train_clips": int(len(train_clips)),
        "n_val_clips": int(len(val_clips)),
        "matched_wavs": matched,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
