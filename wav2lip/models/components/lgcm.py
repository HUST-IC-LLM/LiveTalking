from typing import Optional

import torch
from torch import nn

from ..attns import MulHeadAttn


class LGCM(nn.Module):
    def __init__(
        self,
        channels: int,
        n_heads: int,
        kernel_size: int = 3,
        d_model: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.channels = channels
        self.n_heads = n_heads
        self.kernel_size = kernel_size
        self.d_model = d_model if d_model is not None else channels

        self.local = nn.Sequential(
            nn.Conv2d(
                channels,
                self.d_model,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                bias=False,
            ),
            nn.GELU(),
        )
        self.attn = MulHeadAttn(self.d_model, n_heads)
        self.proj = nn.Linear(self.d_model, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        local = self.local(x)
        out = local.flatten(2).permute(2, 0, 1)
        out = self.attn(out).view(B, H, W, C)
        return x + local + self.proj(out).permute(0, 3, 1, 2)
