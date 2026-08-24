"""On-device Spoken Language Understanding for Model 2 (the 5 MB, used wisely).

Model 2's job is *commands* — but a good command model does more than pick a
label from a fixed list. This is a compact **joint intent + slot** model
(the classic SLU design): it reads the command transcript and produces a
*structured* action, e.g.

    "set a timer for 5 minutes"  ->  {intent: timer, slots: {number: "5", unit: "minutes"}}
    "call mom"                   ->  {intent: call,  slots: {contact: "mom"}}

Architecture: word embedding -> BiGRU -> (a) intent = pooled -> classifier,
(b) slots = per-token BIO tagging head. Tiny (well under 1 MB), so it fits
easily inside the 5 MB command budget alongside the acoustic KWS/router.

Where the transcript comes from: the always-on acoustic command model
handles the common fast-path commands; for anything with slots, the woken
recognizer (Model 1, or a small command-ASR) provides tokens and this SLU
head parses them. Runs on the wrist — no cloud NLU.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import torch
import torch.nn as nn


@dataclass
class SLUConfig:
    word_vocab: int = 64
    num_intents: int = 6
    num_slot_tags: int = 9      # O + B-/I- per slot type
    embed_dim: int = 96
    hidden: int = 96


class SLUModel(nn.Module):
    def __init__(self, cfg: SLUConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.word_vocab, cfg.embed_dim, padding_idx=0)
        self.gru = nn.GRU(cfg.embed_dim, cfg.hidden, batch_first=True, bidirectional=True)
        self.intent = nn.Linear(2 * cfg.hidden, cfg.num_intents)
        self.slots = nn.Linear(2 * cfg.hidden, cfg.num_slot_tags)

    def forward(self, token_ids: torch.Tensor, lengths: torch.Tensor = None):
        # token_ids: (B, L)
        x = self.embed(token_ids)
        out, _ = self.gru(x)                       # (B, L, 2H)
        # intent from masked mean pool
        if lengths is not None:
            mask = (torch.arange(out.size(1), device=out.device)[None, :] < lengths[:, None]).float()
            pooled = (out * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp_min(1)
        else:
            pooled = out.mean(1)
        return {"intent": self.intent(pooled), "slots": self.slots(out)}

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


@dataclass
class SLUResult:
    intent: str
    slots: Dict[str, str] = field(default_factory=dict)

    def as_dict(self):
        return {"intent": self.intent, "slots": self.slots}


class SLUParser:
    """Turns model outputs into a structured `SLUResult` using the label maps."""

    def __init__(self, intents: List[str], slot_tags: List[str]):
        self.intents = intents
        self.slot_tags = slot_tags  # index -> tag string, e.g. "O", "B-number", "I-unit"

    @torch.no_grad()
    def parse(self, model: SLUModel, token_ids: List[int], words: List[str]) -> SLUResult:
        L = len(words)
        t = torch.tensor([token_ids[:L]], dtype=torch.long)
        out = model(t, torch.tensor([L]))
        intent = self.intents[int(out["intent"].argmax(-1).item())]
        tag_ids = out["slots"].argmax(-1)[0].tolist()
        slots: Dict[str, List[str]] = {}
        for w, ti in zip(words, tag_ids):
            tag = self.slot_tags[ti]
            if tag == "O":
                continue
            _, stype = tag.split("-", 1)
            slots.setdefault(stype, []).append(w)
        return SLUResult(intent=intent, slots={k: " ".join(v) for k, v in slots.items()})
