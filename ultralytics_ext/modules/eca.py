from __future__ import annotations

import torch
import torch.nn as nn


class ECA(nn.Module):
    """Efficient Channel Attention (ECA).

    Paper: "ECA-Net: Efficient Channel Attention for Deep Convolutional Neural Networks"
    This implementation keeps inference overhead negligible and does NOT change tensor shapes.

    Notes:
      - We use a fixed 1D conv kernel size (default 3) to avoid needing channel count during init.
      - If you want the original adaptive k(channel) behavior, you can extend this module later.
    """

    def __init__(self, k_size: int = 3, alpha: float = 1.0) -> None:
        super().__init__()
        if k_size % 2 == 0 or k_size <= 0:
            raise ValueError(f"k_size must be a positive odd number, got {k_size}")
        if not (0.0 <= float(alpha) <= 1.0):
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
        # alpha controls the dynamic range of the attention scale around 1:
        #   scale = 1 + alpha * (2*sigmoid(...) - 1)
        # Range: (1-alpha, 1+alpha). alpha=1 recovers the original 2*sigmoid behavior.
        self.alpha = float(alpha)
        # Identity-friendly init: conv output starts at 0 -> sigmoid(0)=0.5.
        # In forward we center around 1.0, so initial scale is exactly 1.0.
        nn.init.zeros_(self.conv.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W]
        y = self.avg_pool(x)  # [B, C, 1, 1]
        y = y.squeeze(-1).transpose(-1, -2)  # [B, 1, C]
        y = self.conv(y)  # [B, 1, C]
        y = self.sigmoid(y).transpose(-1, -2).unsqueeze(-1)  # [B, C, 1, 1] in (0,1)
        centered = (y - 0.5) * 2.0  # in (-1, 1)
        # Backward-compat: old checkpoints may not have `alpha` in the pickled module dict.
        alpha = float(getattr(self, "alpha", 1.0))
        scale = 1.0 + alpha * centered  # in (1-alpha, 1+alpha), identity at init
        return x * scale


