from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from ..components import RoPE
from .base import AttnModel


class MulHeadAttn2d(AttnModel):
    def __init__(
        self,
        embed_size: int,
        n_heads: int,
        d_model: Optional[int] = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__(embed_size, d_model, dropout)
        self.n_heads = n_heads
        self.head_dim = self.d_model // n_heads

        self.W_qkv = nn.Conv2d(
            in_channels=self.embed_size,
            out_channels=self.d_model * 3,
            kernel_size=1,
            bias=False,
        )
        self.W_o = nn.Conv2d(
            in_channels=self.d_model,
            out_channels=self.embed_size,
            kernel_size=1,
        )
        self.rope = RoPE(self.head_dim)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        is_causal: bool = False,
    ) -> torch.Tensor:
        B, C, H, W = x.shape
        Q, K, V = self.qkv(x)

        outputs = F.scaled_dot_product_attention(
            Q, K, V, attn_mask=mask, dropout_p=self.dropout, is_causal=is_causal
        )
        outputs = outputs.view(B, self.d_model, H, W)
        outputs = self.W_o(outputs)

        return self.out_dropout(outputs)

    def qkv(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        B, C, H, W = x.shape

        QKV = self.W_qkv(x)
        Q, K, V = torch.chunk(QKV, chunks=3, dim=1)

        Q = Q.view(B, self.n_heads, self.head_dim, H * W).permute(0, 1, 3, 2)
        K = K.view(B, self.n_heads, self.head_dim, H * W).permute(0, 1, 3, 2)
        V = V.view(B, self.n_heads, self.head_dim, H * W).permute(0, 1, 3, 2)

        Q = self.rope(Q)
        K = self.rope(K)

        return Q, K, V
