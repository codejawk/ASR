"""Stateless transducer predictor (label decoder).

Instead of an LSTM, the predictor is an embedding + a kernel-2 depthwise
conv over the last two label tokens. This is the "stateless decoder" trick
from icefall: it removes almost all predictor parameters (~0.2 M) with no
measurable WER cost on command+dictation domains, and its state is just
"the previous token", which is trivial to carry at inference.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class StatelessDecoder(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int = 256, context: int = 2, blank: int = 0):
        super().__init__()
        self.blank = blank
        self.context = context
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=blank)
        # depthwise conv over the last `context` tokens
        self.conv = nn.Conv1d(embed_dim, embed_dim, kernel_size=context, groups=embed_dim, bias=False)
        self.out_dim = embed_dim

    def forward(self, labels: torch.Tensor) -> torch.Tensor:
        """labels: (B, U) token ids (already left-padded by `context-1` blanks).
        Returns (B, U, embed_dim)."""
        x = self.embed(labels)  # (B, U, E)
        x = x.transpose(1, 2)  # (B, E, U)
        x = nn.functional.pad(x, (self.context - 1, 0), value=0.0)
        x = self.conv(x)
        return x.transpose(1, 2)  # (B, U, E)

    def step(self, prev_tokens: torch.Tensor) -> torch.Tensor:
        """Single decoding step. prev_tokens: (B, context) last `context`
        tokens (left-padded with blank). Returns (B, embed_dim)."""
        x = self.embed(prev_tokens).transpose(1, 2)  # (B, E, context)
        x = self.conv(x)  # (B, E, 1)
        return x.squeeze(-1)
