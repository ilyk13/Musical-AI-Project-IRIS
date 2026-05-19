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

from model.nanopitch import NanoPitch, viterbi_decode, N_MELS, PITCH_FMIN

# Mel spectrogram settings that match the NanoPitch training configuration.
# These must stay fixed — changing them would break pretrained weight compatibility.
SR = 16000
HOP_LENGTH = 160    # 10 ms at 16 kHz
WIN_LENGTH = 400    # 25 ms window
N_FFT = 512
FMIN = 50.0
FMAX = 2000.0


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
        power=2.0,   # log-power mel — matches the NanoPitch-PreExtract training data
    )
    # Epsilon matches NanoPitch-PreExtract: training data floor is -23.031,
    # which equals log(1e-10).  Using 1e-7 would raise the floor to -16.1,
    # making silence look like a weak voiced signal to the model.
    log_mel = np.log(mel + 1e-10).T  # (frames, 40)

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
            ckpt = torch.load(local_path, map_location=device, weights_only=False)
            # Training checkpoints saved by model/train.py have the format:
            #   {"epoch": ..., "state_dict": ..., "model_kwargs": {...}, ...}
            # Plain checkpoints are just the state_dict directly.
            if isinstance(ckpt, dict) and "state_dict" in ckpt:
                kwargs = ckpt.get("model_kwargs", {})
                model = NanoPitch(
                    cond_size=kwargs.get("cond_size", 64),
                    gru_size=kwargs.get("gru_size", 96),
                )
                model.load_state_dict(ckpt["state_dict"])
            else:
                model = NanoPitch()
                model.load_state_dict(ckpt)
            print(f"Loaded NanoPitch weights from {local_path}")
            return cls(model, device)

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
