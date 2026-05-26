"""BreathCNN — lightweight 1D CNN for breath-event detection from raw waveform.

Operates on short per-frame windows (default 480 samples = 30 ms @ 16 kHz)
aligned to the NanoPitch 10 ms hop grid. Used for phrase segmentation in
the AI Vocal Coach pipeline.
"""

from __future__ import annotations

import torch
from torch import nn

DEFAULT_WINDOW = 480


class BreathCNN(nn.Module):
    """Small causal-ish CNN on mono waveform snippets."""

    def __init__(self, window_size: int = DEFAULT_WINDOW):
        super().__init__()
        self.window_size = window_size

        self.net = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=7, stride=2, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv1d(16, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Linear(64, 1)
        n_params = sum(p.numel() for p in self.parameters())
        print(f"BreathCNN: {n_params:,} parameters (window={window_size})")

    def forward(self, windows: torch.Tensor, return_logits: bool = False):
        """Args:
            windows: (B, T, W) or (B, W) mono waveform windows
        Returns:
            prob or logits: (B, T, 1) or (B, 1)
        """
        if windows.dim() == 2:
            x = windows.unsqueeze(1)
            feat = self.net(x).squeeze(-1)
            logits = self.head(feat)
            if return_logits:
                return logits
            return torch.sigmoid(logits)

        B, T, W = windows.shape
        x = windows.reshape(B * T, 1, W)
        feat = self.net(x).squeeze(-1)
        logits = self.head(feat).view(B, T, 1)
        if return_logits:
            return logits
        return torch.sigmoid(logits)


if __name__ == "__main__":
    model = BreathCNN()
    batch = torch.randn(4, 200, DEFAULT_WINDOW)
    out = model(batch)
    print(out.shape)
