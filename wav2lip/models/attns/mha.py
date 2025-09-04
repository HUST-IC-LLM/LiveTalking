import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from ..components import RoPE
from .base import AttnInfraRecord, AttnModel
from .functional import sdp_attn


class MulHeadAttn(AttnModel):
    def __init__(
        self,
        embed_size: int,
        n_heads: int,
        d_model: Optional[int] = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__(embed_size, d_model, dropout)
        assert self.d_model % n_heads == 0, (
            "[ERROR] embed_size must be divisible by n_heads"
        )
        self.n_heads = n_heads
        self.head_dim = self.d_model // n_heads

        self.W_qkv = nn.Linear(self.embed_size, self.d_model * 3, bias=False)
        self.W_o = nn.Linear(self.d_model, self.embed_size)
        self.rope = RoPE(self.head_dim)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        is_causal: bool = False,
    ) -> torch.Tensor:
        B, C, E = x.shape
        Q, K, V = self.qkv(x)

        outputs = F.scaled_dot_product_attention(
            Q, K, V, attn_mask=mask, dropout_p=self.dropout, is_causal=is_causal
        )

        outputs = (outputs.transpose(1, 2)).reshape(B, C, -1)
        outputs = self.W_o(outputs)
        return self.out_dropout(outputs)

    def qkv(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        B, C, E = x.shape

        QKV = self.W_qkv(x)
        Q, K, V = QKV.chunk(3, dim=-1)

        Q = Q.view(B, C, self.n_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, C, self.n_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, C, self.n_heads, self.head_dim).transpose(1, 2)

        Q = self.rope(Q)
        K = self.rope(K)

        return Q, K, V

    def prompt(self, record: AttnInfraRecord) -> AttnInfraRecord:
        B, C, E = record.input_logits.shape
        Q, K, V = self.qkv(record.input_logits)

        outputs, weights = sdp_attn(Q, K, V, mask="^")
        outputs = (outputs.transpose(1, 2)).reshape(B, C, -1)
        outputs = self.W_o(outputs)

        record.k_cache = K
        record.v_cache = V
        record.attn_weights = weights
        record.output_logits = outputs
        return record

    def infer(
        self, record: AttnInfraRecord, use_cache: bool = False
    ) -> AttnInfraRecord:
        B, C, E = record.input_logits.shape

        if (
            use_cache
            and record.k_cache is not None
            and record.v_cache is not None
            and record.attn_weights is not None
        ):
            d_length = C - record.k_cache.shape[2]
            new_inputs = record.input_logits[:, -d_length:, :]

            Q, K, V = self.qkv(new_inputs)

            K = torch.cat([record.k_cache, K], dim=2)
            V = torch.cat([record.v_cache, V], dim=2)

            scores = Q @ K.transpose(-2, -1) / math.sqrt(self.head_dim)
            scores = F.softmax(scores, dim=-1)
            weights = torch.cat(
                [
                    record.attn_weights,
                    torch.zeros(B, self.n_heads, C - d_length, d_length),
                ],
                dim=-1,
            )
            weights = torch.cat([weights, scores], dim=2)

            outputs = weights @ V
            outputs = outputs.transpose(1, 2).reshape(B, C, -1)
            outputs = self.W_o(outputs)

            record.k_cache = K
            record.v_cache = V
            record.attn_weights = weights
            record.output_logits = outputs
            return record

        else:
            return self.prompt(record)
