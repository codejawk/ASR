"""Router-gated Mixture-of-Specialists for Model 1.

Coarse, utterance-level experts (one per domain: comms / media / info /
control, or per-language) — NOT token-level MoE. The router is Model 2's
router head; it picks the expert, the pager loads it from flash, and we
decode with it.

    audio ─► [Model 2 router] ─domain─► [ExpertPager] ─► specialist ─► text
                                          (flash → RAM, LRU)

Two honest framings for what an "expert" is:
  * **Full specialist transducer per domain** (what this class supports):
    each expert is an independent Model-1, trained on its domain, paged so
    only one is resident. Big effective capacity, small resident footprint.
  * Shared-trunk + per-domain heads (cheaper) — a config option teams can
    add later; the routing/paging interface is identical.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import torch

from ..decode import streaming_greedy_search
from .paging import ExpertPager


@dataclass
class MixtureOfSpecialists:
    pager: ExpertPager
    tokenizer: object
    domain_names: List[str]
    router: Optional[object] = None      # a CommandModel (uses .router head)
    frontend: Optional[object] = None
    cmvn: Optional[object] = None

    def route(self, feats_cmd: torch.Tensor) -> int:
        """Pick a domain id from a command-model feature tensor (B=1)."""
        if self.router is None:
            return 0
        with torch.no_grad():
            emb = self.router.embed_audio(feats_cmd.unsqueeze(0))
            return int(self.router.router(emb).argmax(-1).item())

    def transcribe(self, feats_asr: torch.Tensor, domain: int) -> dict:
        """Page in the domain expert and decode `feats_asr` (T, C)."""
        expert, latency = self.pager.get(domain)
        ids = streaming_greedy_search(expert, feats_asr)
        return {
            "domain": domain,
            "domain_name": self.domain_names[domain] if domain < len(self.domain_names) else str(domain),
            "text": self.tokenizer.decode(ids),
            "page_latency_ms": round(1000 * latency, 2),
        }

    def run(self, feats_cmd: torch.Tensor, feats_asr: torch.Tensor) -> dict:
        """Full path: route on the command features, transcribe with the
        selected specialist."""
        domain = self.route(feats_cmd)
        return self.transcribe(feats_asr, domain)

    def report(self) -> dict:
        return self.pager.report()
