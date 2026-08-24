"""Greedy CTC decode (collapse repeats, drop blanks). Used for the aux
CTC head fallback path and for the phoneme-CTC command model."""
from __future__ import annotations

from typing import List

import torch


@torch.no_grad()
def ctc_greedy_decode(log_probs: torch.Tensor, blank: int = 0) -> List[int]:
    """log_probs: (T, V). Returns collapsed token ids."""
    ids = log_probs.argmax(-1).tolist()
    out: List[int] = []
    prev = None
    for i in ids:
        if i != blank and i != prev:
            out.append(i)
        prev = i
    return out
