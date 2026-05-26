"""NanoPitchExtractor — wraps the NanoPitch model for easy f0 extraction.

Handles mel spectrogram computation, model inference, and Viterbi decoding
in one call. Load pretrained weights from Hugging Face on first use.

Usage:
    extractor = NanoPitchExtractor.from_pretrained()
    f0, vad = extractor.extract(audio, sr=16000)
"""

import numpy as np
import torch
import librosa
from pathlib import Path

from model.nanopitch import NanoPitch, NanoPitchPlus, viterbi_decode, N_MELS, PITCH_FMIN

# Mel spectrogram settings that match the NanoPitch training configuration.
# These must stay fixed — changing them would break pretrained weight compatibility.
SR = 16000
HOP_LENGTH = 160    # 10 ms at 16 kHz
WIN_LENGTH = 400    # 25 ms window
N_FFT = 512
FMIN = 0.0          # HTK mel filterbank — matches nanopitch.c / PreExtract
FMAX = 8000.0
LOG_OFFSET = 1e-10
NC_CONV_CONTEXT = 4  # causal conv warmup frames (matches nanopitch.h)
MEL_MIN_TAIL = N_FFT - 32   # librosa uncentered STFT tail (480 at defaults)


class StreamingMel:
    """Librosa-aligned mel frames from a rolling audio buffer."""

    @staticmethod
    def frame_from_buffer(buf: np.ndarray, abs_start: int, roll_start: int) -> np.ndarray:
        """One log-mel frame matching batch _compute_mel(center=False)."""
        pos = abs_start - roll_start
        if pos < 0 or pos >= len(buf):
            raise ValueError("abs_start outside rolling buffer")
        return _mel_frame_from_segment(buf[pos:])


def _mel_frame_from_segment(y_seg: np.ndarray) -> np.ndarray:
    """Match librosa batch frame 0 for segment starting at frame boundary."""
    if len(y_seg) < N_FFT:
        y_seg = np.pad(y_seg, (0, N_FFT - len(y_seg)))
    mel = librosa.feature.melspectrogram(
        y=y_seg,
        sr=SR,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH,
        n_mels=N_MELS,
        fmin=FMIN,
        fmax=FMAX,
        power=2.0,
        center=False,
        htk=True,
    )
    return np.log(mel[:, 0] + LOG_OFFSET).astype(np.float32)


def _mel_frame_from_window(window: np.ndarray) -> np.ndarray:
    return _mel_frame_from_segment(window)


def _compute_mel(audio: np.ndarray, sr: int = SR) -> np.ndarray:
    """Compute log-mel spectrogram matching NanoPitch's training config.

    Args:
        audio: (T,) float32 mono audio at `sr` Hz
        sr:    sample rate (must be 16000)

    Returns:
        mel: (frames, 40) float32 log-mel spectrogram
    """
    if sr != SR:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=SR)

    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=SR,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH,
        n_mels=N_MELS,
        fmin=FMIN,
        fmax=FMAX,
        power=2.0,
        center=False,
        htk=True,
    )
    log_mel = np.log(mel + LOG_OFFSET).T  # (frames, 40)

    return log_mel.astype(np.float32)


class NanoPitchExtractor:
    """High-level interface for extracting f0 and VAD from audio.

    Args:
        model:  a NanoPitch instance (with loaded weights)
        device: 'cpu' or 'cuda'
    """

    def __init__(self, model: NanoPitch, device: str = 'cpu'):
        self.model = model.to(device)
        self.model.eval()
        self.device = device

    @property
    def has_gesture_head(self) -> bool:
        return isinstance(self.model, NanoPitchPlus)

    @classmethod
    def from_checkpoint(
        cls,
        local_path: str,
        device: str = 'cpu',
        prefer_plus: bool = False,
    ) -> "NanoPitchExtractor":
        """Load NanoPitch or NanoPitchPlus from a training checkpoint."""
        ckpt = torch.load(local_path, map_location=device, weights_only=False)
        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            kwargs = ckpt.get("model_kwargs", {})
            is_plus = ckpt.get("model_type") == "NanoPitchPlus" or prefer_plus
            if is_plus:
                model = NanoPitchPlus(
                    cond_size=kwargs.get("cond_size", 64),
                    gru_size=kwargs.get("gru_size", 96),
                )
                model.load_state_dict(ckpt["state_dict"], strict=False)
            else:
                model = NanoPitch(
                    cond_size=kwargs.get("cond_size", 64),
                    gru_size=kwargs.get("gru_size", 96),
                )
                model.load_state_dict(ckpt["state_dict"])
        else:
            model = NanoPitchPlus() if prefer_plus else NanoPitch()
            model.load_state_dict(ckpt)
        kind = "NanoPitchPlus" if isinstance(model, NanoPitchPlus) else "NanoPitch"
        print(f"Loaded {kind} weights from {local_path}")
        return cls(model, device)

    @classmethod
    def from_pretrained(
        cls,
        repo_id: str = "smulelabs/NanoPitch",
        filename: str = "nanopitch.pt",
        device: str = 'cpu',
        local_path: str | None = None,
    ) -> "NanoPitchExtractor":
        """Load a NanoPitch extractor with pretrained weights.

        Tries to download from Hugging Face if `local_path` is not given.
        Falls back to a randomly-initialized model with a warning if the
        download fails (useful for offline demos / CI).

        Args:
            repo_id:    HuggingFace repo containing the checkpoint
            filename:   checkpoint filename within the repo
            device:     torch device to load the model onto
            local_path: if set, load directly from this .pt file instead

        Returns:
            NanoPitchExtractor ready for inference
        """
        if local_path and Path(local_path).exists():
            return cls.from_checkpoint(local_path, device=device)

        model = NanoPitch()
        try:
            from huggingface_hub import hf_hub_download
            ckpt_path = hf_hub_download(repo_id=repo_id, filename=filename)
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            if isinstance(ckpt, dict) and "state_dict" in ckpt:
                model.load_state_dict(ckpt["state_dict"])
            else:
                model.load_state_dict(ckpt)
            print(f"Loaded NanoPitch pretrained weights from {repo_id}/{filename}")
        except Exception as e:
            print(f"[WARNING] Could not load pretrained weights ({e}). "
                  "Running with random weights — pitch output will be meaningless.")

        return cls(model, device)

    def extract(
        self,
        audio: np.ndarray,
        sr: int = SR,
        transition_width: int = 12,
        voicing_threshold: float = 0.3,
        onset_penalty: float = 2.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Extract f0 and voice activity from a mono audio array.

        Args:
            audio:             (T,) float32 mono audio
            sr:                sample rate of `audio`
            transition_width:  Viterbi max pitch jump in bins per frame
            voicing_threshold: min posteriorgram confidence for voiced frames
            onset_penalty:     log-domain voiced↔unvoiced transition cost

        Returns:
            f0_hz: (frames,) float32 — fundamental frequency in Hz (0 = unvoiced)
            vad:   (frames,) float32 — voice activity probability [0, 1]
        """
        mel = _compute_mel(audio, sr)  # (frames, 40)

        mel_tensor = torch.from_numpy(mel).unsqueeze(0).to(self.device)  # (1, T, 40)

        with torch.no_grad():
            vad_out, pitch_out, _ = self.model(mel_tensor)

        vad = vad_out[0, :, 0].cpu().numpy()          # (T,)
        posteriorgram = pitch_out[0].cpu().numpy()     # (T, 360)

        f0_hz = viterbi_decode(
            posteriorgram,
            transition_width=transition_width,
            voicing_threshold=voicing_threshold,
            onset_penalty=onset_penalty,
        )

        return f0_hz, vad

    def hop_seconds(self) -> float:
        """Duration of each output frame in seconds (10 ms)."""
        return HOP_LENGTH / SR

    def frame_times(self, n_frames: int) -> np.ndarray:
        """Return the center time (seconds) of each output frame."""
        return np.arange(n_frames) * self.hop_seconds()
