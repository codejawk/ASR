"""Transducer joiner.

Projects encoder and predictor into a shared joiner space, sums, and maps
to the vocabulary. The vocab projection is the "tax you pay twice"
(embedding + here), so `joiner_dim` and vocab size are the main knobs.
Keeping joiner_dim=256 and BPE=500 keeps this ~0.4 M params.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class Joiner(nn.Module):
    def __init__(self, enc_dim: int, pred_dim: int, joiner_dim: int, vocab_size: int):
        super().__init__()
        self.enc_proj = nn.Linear(enc_dim, joiner_dim)
        self.pred_proj = nn.Linear(pred_dim, joiner_dim)
        self.out = nn.Linear(joiner_dim, vocab_size)

    def forward(self, enc: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
        """enc: (B, T, De) or (B, De); pred: (B, U, Dp) or (B, Dp).

        Training: enc (B,T,1,De) + pred (B,1,U,Dp) -> (B,T,U,V).
        Step:     enc (B,De) + pred (B,Dp) -> (B,V).
        """
        if enc.dim() == 3 and pred.dim() == 3:
            enc = self.enc_proj(enc).unsqueeze(2)  # (B,T,1,J)
            pred = self.pred_proj(pred).unsqueeze(1)  # (B,1,U,J)
            x = torch.tanh(enc + pred)
            return self.out(x)  # (B,T,U,V)
        else:
            x = torch.tanh(self.enc_proj(enc) + self.pred_proj(pred))
            return self.out(x)  # (B,V)
