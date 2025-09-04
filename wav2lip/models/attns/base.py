from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import nn


@dataclass
class AttnInfraRecord:
    input_logits: torch.Tensor
    output_logits: Optional[torch.Tensor] = None
    attn_weights: Optional[torch.Tensor] = None
    k_cache: Optional[torch.Tensor] = None
    v_cache: Optional[torch.Tensor] = None


class AttnModel(ABC, nn.Module):
    def __init__(
        self,
        embed_size: int,
        d_model: Optional[int] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed_size = embed_size
        self.d_model: int = d_model if d_model is not None else embed_size
        self.dropout = dropout

        self.out_dropout = nn.Dropout(dropout)

    @abstractmethod
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        is_causal: bool = False,
    ) -> torch.Tensor: ...

    @abstractmethod
    def qkv(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]: ...

    def prompt(self, record: AttnInfraRecord) -> AttnInfraRecord: ...
    def infer(self, record: AttnInfraRecord) -> AttnInfraRecord: ...
