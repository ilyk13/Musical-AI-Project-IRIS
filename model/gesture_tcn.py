"""Standalone GestureTCN — causal TCN on f0-trajectory features."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from model.gesture_features import N_GESTURE_FEATURES, GestureFeatureStream

NUM_GESTURES = 4


class _CausalConvBlock(nn.Module):
    def __init__(self, channels: int, kernel: int = 3, dilation: int = 1, dropout: float = 0.1):
        super().__init__()
        self.pad = (kernel - 1) * dilation
        self.conv = nn.Conv1d(channels, channels, kernel, dilation=dilation)
        self.act = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv(F.pad(x, (self.pad, 0)))
        y = self.act(y)
        return self.drop(y)


class GestureTCN(nn.Module):
    """Causal temporal conv net on per-frame f0 features → gesture logits."""

    def __init__(
        self,
        n_features: int = N_GESTURE_FEATURES,
        n_classes: int = NUM_GESTURES,
        channels: int = 64,
        dilations: tuple[int, ...] = (1, 2, 4, 8, 16),
        dropout: float = 0.15,
    ):
        super().__init__()
        self.n_features = n_features
        self.n_classes = n_classes
        self.channels = channels
        self.dilations = dilations

        self.stem = nn.Conv1d(n_features, channels, kernel_size=1)
        self.blocks = nn.ModuleList(
            [_CausalConvBlock(channels, dilation=d, dropout=dropout) for d in dilations]
        )
        self.head = nn.Linear(channels, n_classes)
        n_params = sum(p.numel() for p in self.parameters())
        print(f"GestureTCN: {n_params:,} parameters "
              f"(features={n_features}, channels={channels}, dilations={dilations})")

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Args: features (B, T, F). Returns logits (B, T, n_classes)."""
        x = features.transpose(1, 2)
        x = self.stem(x)
        for block in self.blocks:
            x = x + block(x)
        x = x.transpose(1, 2)
        return self.head(x)

    @property
    def receptive_field(self) -> int:
        return 1 + sum(2 * d for d in self.dilations)

    def init_stream(self, context: int | None = None) -> GestureFeatureStream:
        ctx = context or max(64, self.receptive_field + 8)
        return GestureFeatureStream(context=ctx)

    @classmethod
    def load_checkpoint(cls, path: str | Path, device: str | torch.device = "cpu") -> GestureTCN:
        ckpt = torch.load(path, map_location=device, weights_only=False)
        kwargs = ckpt.get("model_kwargs", {}) if isinstance(ckpt, dict) else {}
        model = cls(
            n_features=kwargs.get("n_features", N_GESTURE_FEATURES),
            n_classes=kwargs.get("n_classes", NUM_GESTURES),
            channels=kwargs.get("channels", 64),
            dilations=tuple(kwargs.get("dilations", (1, 2, 4, 8, 16))),
            dropout=kwargs.get("dropout", 0.15),
        )
        state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
        model.load_state_dict(state)
        model.to(device)
        model.eval()
        return model

    @torch.no_grad()
    def predict_frame(
        self,
        feat_window: torch.Tensor,
        stream: GestureFeatureStream | None = None,
    ) -> torch.Tensor:
        """feat_window: (T, F) numpy-built features; returns logits (n_classes,)."""
        x = torch.from_numpy(feat_window).float().unsqueeze(0)
        dev = next(self.parameters()).device
        x = x.to(dev)
        logits = self.forward(x)[0, -1]
        return logits.cpu()
