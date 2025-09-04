from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from .base import AttnModel


class GroupedQueryAttn(AttnModel):
    def __init__(
        self,
        embed_size: int,
        q_heads: int,
        kv_heads: int,
        d_model: Optional[int] = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__(embed_size, d_model, dropout)
        assert self.d_model % q_heads == 0, (
            "[ERROR] embed_size must be divisible by n_heads"
        )

        self.q_heads = q_heads
        self.kv_heads = kv_heads
        self.kv_embed_dim = self.d_model // q_heads * kv_heads
        self.head_dim = self.d_model // q_heads
        self.heads_per_group = q_heads // kv_heads

        self.W_q = nn.Linear(self.embed_size, self.d_model, bias=False)
        self.W_kv = nn.Linear(self.embed_size, self.kv_embed_dim * 2, bias=False)
        self.W_o = nn.Linear(self.d_model, self.embed_size)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        is_causal: bool = False,
    ) -> torch.Tensor:
        B, C, E = x.shape
        Q, K, V = self.qkv(x)

        outputs = F.scaled_dot_product_attention(
            Q,
            K,
            V,
            attn_mask=mask,
            dropout_p=self.dropout,
            is_causal=is_causal,
            enable_gqa=True,
        )
        outputs = outputs.permute(0, 3, 1, 2, 4).reshape(
            B, C, self.heads_per_group * self.kv_heads * self.head_dim
        )
        outputs = self.W_o(outputs)
        return self.out_dropout(outputs)

    def qkv(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        B, C, E = x.shape
        Q = (
            self.W_q(x)
            .view(B, C, self.heads_per_group, self.kv_heads, self.head_dim)
            .permute(0, 2, 3, 1, 4)
        )
        KV = self.W_kv(x).view(B, C, self.kv_heads, 2, self.head_dim).transpose(1, 2)
        K, V = KV.unbind(dim=3)

        return Q, K, V
