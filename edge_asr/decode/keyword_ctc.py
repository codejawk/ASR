"""Runtime-extensible keyword spotting via phoneme/char-CTC + a keyword
list, plus the metric that actually matters on-device.

For the closed-set BC-ResNet path you just argmax the classifier. This
module covers the *open-vocab* path: a small CTC acoustic model emits a
token lattice and keywords are matched against it at runtime — so you add
a command by editing a text file, no retraining.

The key on-device metric is **false-accepts per hour** at a fixed
false-reject rate, not accuracy. `score_false_accepts` computes it over a
stream of negative (no-keyword) audio.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence

import torch

from .ctc_greedy import ctc_greedy_decode


@dataclass
class KeywordSpotter:
    tokenizer: object
    keywords: List[str]
    threshold: float = 0.5
    blank: int = 0

    def _keyword_logprob(self, log_probs: torch.Tensor, keyword: str) -> float:
        """Cheap confidence: mean per-frame max-prob over the frames whose
        greedy token sequence contains the keyword's token subsequence.
        A production system replaces this with FST-constrained forced
        alignment; the interface (a score to threshold) is identical."""
        target = self.tokenizer.encode(keyword)
        hyp = ctc_greedy_decode(log_probs, self.blank)
        # subsequence containment
        if not _contains_subseq(hyp, target):
            return 0.0
        conf = log_probs.max(dim=-1).values.exp().mean().item()
        return conf

    def detect(self, log_probs: torch.Tensor) -> Dict[str, float]:
        """Return {keyword: confidence} for keywords above threshold."""
        hits = {}
        for kw in self.keywords:
            c = self._keyword_logprob(log_probs, kw)
            if c >= self.threshold:
                hits[kw] = c
        return hits

    def score_false_accepts(
        self, negative_streams: Sequence[torch.Tensor], total_hours: float
    ) -> float:
        """negative_streams: list of (T,V) log-prob tensors that contain NO
        keyword. Returns false accepts per hour at the current threshold."""
        fa = 0
        for lp in negative_streams:
            if self.detect(lp):
                fa += 1
        return fa / max(total_hours, 1e-9)


def _contains_subseq(seq: List[int], sub: List[int]) -> bool:
    if not sub:
        return False
    it = iter(seq)
    return all(any(x == s for x in it) for s in sub)
