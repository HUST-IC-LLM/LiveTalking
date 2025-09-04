import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F


def masked_softmax(
    x: torch.Tensor,
    mask: Optional[str] | torch.Tensor = None,
    dim: int = -1,
) -> torch.Tensor:
    x = x - x.max(dim=dim, keepdim=True).values

    if mask is None:
        return F.softmax(x, dim=dim)
    elif mask == "^":
        causal_mask = torch.triu(
            torch.ones(x.shape[-2:], dtype=torch.bool, device=x.device), diagonal=1
        )
        x = x.masked_fill(causal_mask, float("-inf"))
    elif isinstance(mask, torch.Tensor):
        x = x.masked_fill(~mask.bool(), float("-inf"))
        return F.softmax(x, dim=dim)
    else:
        raise ValueError("Invalid masked value")

    return F.softmax(x, dim=dim)


def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: Optional[str] | torch.Tensor = None,
    dropout: Optional[torch.nn.Module] = None,
    dim: int = -1,
) -> torch.Tensor:
    d = Q.shape[-1]
    outputs = (Q @ K.transpose(dim, -2)) / math.sqrt(d)
    outputs = masked_softmax(outputs, mask=mask, dim=dim)
    outputs = outputs if dropout is None else dropout(outputs)
    outputs = outputs @ V
    return outputs


def sdp_attn(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: Optional[str] | torch.Tensor = None,
    dropout: Optional[torch.nn.Module] = None,
    dim: int = -1,
) -> Tuple[torch.Tensor, ...]:
    d = Q.shape[-1]
    weights = (Q @ K.transpose(dim, -2)) / math.sqrt(d)
    weights = masked_softmax(weights, mask=mask, dim=dim)
    weights = weights if dropout is None else dropout(weights)
    outputs = weights @ V
    return outputs, weights
