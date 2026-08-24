"""Model 2, redesigned to actually use the 5 MB budget.

Three capabilities in one always-on-gated model:

  1. **Open-vocabulary KWS via a hypernetwork** (HyperSpotter-style,
     arXiv:2508.04857). A keyword *text* encoder generates a matched-filter
     weight vector for an arbitrary command string at runtime, and detection
     is a dot-product against the pooled speech embedding. => add or change a
     command by typing text; **no retraining**.
  2. **Router head** — predicts the domain/language/intent of the utterance,
     which selects (and pages in) the right Model-1 expert. This is what
     turns two bolted-together models into one routed system.
  3. **Speaker embedding** — an L2-normalized vector for optional owner-
     gating ("only respond to me").

Two-tier power design: a tiny always-on `WakeStub` runs 24/7 on the eNPU;
the full `CommandModel` (speech encoder + hypernet + router + speaker) runs
only *after* wake, so the 5 MB of capacity never costs always-on power.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class CommandModelConfig:
    n_mels: int = 40
    embed_dim: int = 128       # shared speech/keyword embedding space
    enc_channels: int = 64
    char_vocab: int = 40       # keyword text alphabet size (see char ids below)
    kw_hidden: int = 128
    num_domains: int = 4       # router targets -> Model-1 experts
    num_langs: int = 1
    speaker_dim: int = 64


# keyword text alphabet (id 0 = pad)
_KW_ALPHABET = " abcdefghijklmnopqrstuvwxyz'0123456789"


def encode_keyword(text: str, max_len: int = 24) -> List[int]:
    ids = [(_KW_ALPHABET.index(c) + 1) if c in _KW_ALPHABET else 0 for c in text.lower()]
    ids = ids[:max_len] + [0] * max(0, max_len - len(ids))
    return ids


class SpeechEncoder2(nn.Module):
    """Compact log-mel -> pooled embedding. Depthwise-separable convs keep
    it small and quantization-friendly."""

    def __init__(self, cfg: CommandModelConfig):
        super().__init__()
        c = cfg.enc_channels
        self.stem = nn.Sequential(
            nn.Conv2d(1, c, 3, stride=(2, 1), padding=1), nn.BatchNorm2d(c), nn.SiLU(),
        )
        self.body = nn.Sequential(
            self._dsblock(c, c, stride=(2, 1)),
            self._dsblock(c, c * 2, stride=(2, 1)),
            self._dsblock(c * 2, c * 2, stride=(1, 1)),
        )
        self.proj = nn.Linear(c * 2, cfg.embed_dim)

    @staticmethod
    def _dsblock(cin, cout, stride):
        return nn.Sequential(
            nn.Conv2d(cin, cin, 3, stride=stride, padding=1, groups=cin, bias=False),
            nn.Conv2d(cin, cout, 1, bias=False),
            nn.BatchNorm2d(cout), nn.SiLU(),
        )

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        # feats: (B, T, n_mels)
        x = feats.transpose(1, 2).unsqueeze(1)  # (B,1,n_mels,T)
        x = self.stem(x)
        x = self.body(x)
        x = x.mean(dim=(2, 3))  # global pool -> (B, c*2)
        return F.normalize(self.proj(x), dim=-1)  # unit embedding


class KeywordHyperNet(nn.Module):
    """Keyword text -> matched-filter (weight vector + bias) in embedding
    space. This is the 'hyper' part: it *generates* a detector for a keyword
    it may never have seen at train time."""

    def __init__(self, cfg: CommandModelConfig):
        super().__init__()
        self.embed = nn.Embedding(cfg.char_vocab + 1, cfg.kw_hidden, padding_idx=0)
        self.gru = nn.GRU(cfg.kw_hidden, cfg.kw_hidden, batch_first=True, bidirectional=True)
        self.to_filter = nn.Linear(2 * cfg.kw_hidden, cfg.embed_dim)
        self.to_bias = nn.Linear(2 * cfg.kw_hidden, 1)

    def forward(self, kw_ids: torch.Tensor):
        # kw_ids: (B, L) int
        x = self.embed(kw_ids)
        out, _ = self.gru(x)
        pooled = out.mean(dim=1)  # (B, 2H)
        filt = F.normalize(self.to_filter(pooled), dim=-1)  # (B, E)
        bias = self.to_bias(pooled).squeeze(-1)             # (B,)
        return filt, bias


class CommandModel(nn.Module):
    def __init__(self, cfg: CommandModelConfig):
        super().__init__()
        self.cfg = cfg
        self.speech = SpeechEncoder2(cfg)
        self.hypernet = KeywordHyperNet(cfg)
        self.router = nn.Linear(cfg.embed_dim, cfg.num_domains)
        self.lang = nn.Linear(cfg.embed_dim, cfg.num_langs)
        self.speaker = nn.Linear(cfg.embed_dim, cfg.speaker_dim)
        self.logit_scale = nn.Parameter(torch.tensor(10.0))  # temperature for detection

    def embed_audio(self, feats):
        return self.speech(feats)

    def detect_logits(self, feats, kw_ids):
        """Detection score logits for (audio, keyword) pairs, elementwise."""
        emb = self.speech(feats)                 # (B,E)
        filt, bias = self.hypernet(kw_ids)       # (B,E),(B,)
        score = (emb * filt).sum(-1) * self.logit_scale + bias
        return score, emb

    def forward(self, feats, kw_ids):
        score, emb = self.detect_logits(feats, kw_ids)
        return {
            "detect": score,                     # (B,) BCE-with-logits vs is_match
            "router": self.router(emb),          # (B, num_domains)
            "lang": self.lang(emb),              # (B, num_langs)
            "speaker": F.normalize(self.speaker(emb), dim=-1),  # (B, spk_dim)
            "embed": emb,
        }

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


class WakeStub(nn.Module):
    """Tiny always-on wake detector (~<0.1 MB). Runs 24/7 on the eNPU; only
    on a positive does the full CommandModel / router spin up."""

    def __init__(self, n_mels: int = 40, channels: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, channels, 3, stride=2, padding=1), nn.BatchNorm2d(channels), nn.SiLU(),
            nn.Conv2d(channels, channels, 3, stride=2, padding=1, groups=channels, bias=False),
            nn.BatchNorm2d(channels), nn.SiLU(),
        )
        self.head = nn.Linear(channels, 2)  # wake / not-wake

    def forward(self, feats):
        x = feats.transpose(1, 2).unsqueeze(1)
        x = self.net(x).mean(dim=(2, 3))
        return self.head(x)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
