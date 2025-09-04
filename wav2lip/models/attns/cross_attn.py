import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from ..components import RoPE
from .base import AttnInfraRecord
from .functional import sdp_attn


class MulHeadCrossAttn(nn.Module):
    def __init__(
        self,
        d_q: int,
        d_kv: int,
        n_heads: int,
        d_model: Optional[int] = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.d_q = d_q
        self.d_kv = d_kv
        self.n_heads = n_heads
        self.d_model = d_model if d_model is not None else d_q
        self.head_dim = self.d_model // n_heads
        self.dropout = dropout
        assert self.d_model % self.n_heads == 0, (
            f"[ERROR] d_model {self.d_model} must be divisible by n_heads {self.n_heads}"
        )

        self.W_q = nn.Linear(self.d_q, self.d_model, bias=False)
        self.W_kv = nn.Linear(self.d_kv, 2 * self.d_model, bias=False)
        self.W_o = nn.Linear(self.d_model, self.d_q, bias=False)

        self.rope = RoPE(self.head_dim)

        self.out_dropout = nn.Dropout(dropout)

    def forward(self, x_1: torch.Tensor, x_2: torch.Tensor, mask=None) -> torch.Tensor:
        B, C, E = x_1.shape
        Q, K, V = self.qkv(x_1, x_2)

        outputs = F.scaled_dot_product_attention(
            Q, K, V, attn_mask=mask, dropout_p=self.dropout
        )
        outputs = (outputs.transpose(1, 2)).reshape(B, C, -1)
        outputs = self.W_o(outputs)

        self.attn_dropout = nn.Dropout(self.dropout)
        return self.out_dropout(outputs)

    def qkv(
        self, x_1: torch.Tensor, x_2: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, C_1, E_1 = x_1.shape
        B, C_2, E_2 = x_2.shape
        Q = self.W_q(x_1)
        KV = self.W_kv(x_2)
        K, V = KV.chunk(2, dim=-1)

        Q = Q.view(B, C_1, self.n_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, C_2, self.n_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, C_2, self.n_heads, self.head_dim).transpose(1, 2)

        Q = self.rope(Q)
        V = self.rope(V)

        return Q, K, V

    def prompt(self, record: AttnInfraRecord) -> AttnInfraRecord:
        x_1, x_2 = record.input_logits
        B, C_1, E = x_1.shape
        B, C_2, E = x_2.shape
        Q, K, V = self.qkv(x_1, x_2)
        outputs, weights = sdp_attn(Q, K, V, mask="^", dropout=self.attn_dropout)
        outputs = (outputs.transpose(1, 2)).reshape(B, C_1, -1)
        outputs = self.W_o(outputs)
        record.k_cache = K
        record.v_cache = V
        record.attn_weights = weights
        record.output_logits = outputs
        return record

    def infer(
        self, record: AttnInfraRecord, use_cache: bool = False
    ) -> AttnInfraRecord:
        x_1, x_2 = record.input_logits
        B, C_1, E = x_1.shape
        B, C_2, E = x_2.shape
        if (
            use_cache
            and record.k_cache is not None
            and record.v_cache is not None
            and record.attn_weights is not None
        ):
            d_length = C_1 - record.k_cache.shape[2]
            new_inputs = x_1[:, -d_length:, :]
            Q, K, V = self.qkv(new_inputs, x_2)
            K = torch.cat([record.k_cache, K], dim=2)
            V = torch.cat([record.v_cache, V], dim=2)
            scores = Q @ K.transpose(-2, -1) / (math.sqrt(self.head_dim))
            scores = F.softmax(scores, dim=-1)
            weights = torch.cat(
                [
                    record.attn_weights,
                    torch.zeros(B, self.n_heads, C_1 - d_length, d_length),
                ],
                dim=-1,
            )
            weights = torch.cat([weights, scores], dim=2)
            outputs = weights @ V
            outputs = outputs.transpose(1, 2).reshape(B, C_1, -1)
            outputs = self.W_o(outputs)
            record.k_cache = K
            record.v_cache = V
            record.attn_weights = weights
            record.output_logits = outputs
            return record
        else:
            return self.prompt(record)
