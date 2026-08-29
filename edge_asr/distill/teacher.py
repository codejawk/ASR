"""Teacher wrapper for distillation.

Two roles:
  * **pseudo-labeler** — transcribe unlabeled audio to text targets
    (sequence-level KD); this is the standard path where a large teacher
    like Whisper/Parakeet labels thousands of hours (scripts/02_pseudo_label.sh).
  * **soft-target provider** — emit CTC logits / encoder features for
    frame-level KD (`distill/kd_loss.py`).

In this repo the teacher is any trained (larger) `Transducer`, so the whole
distillation loop is self-contained and testable on synthetic data. To use
a *real* foundation-model teacher, produce a pseudo-labelled manifest with
`scripts/02_pseudo_label.sh` and train the student on it (sequence-KD); wire
a real teacher's CTC posteriors here for frame-KD.
"""
from __future__ import annotations

from typing import List

import torch

from ..decode import greedy_search


class Teacher:
    def __init__(self, model, tokenizer):
        self.model = model.eval()
        self.tok = tokenizer
        for p in self.model.parameters():
            p.requires_grad_(False)

    @property
    def device(self):
        return next(self.model.parameters()).device

    @torch.no_grad()
    def transcribe(self, feats: torch.Tensor) -> str:
        """Pseudo-label a single utterance (T, C) -> text."""
        return self.tok.decode(greedy_search(self.model, feats.to(self.device)))

    @torch.no_grad()
    def soft_targets(self, feats, feat_lens, targets, target_lens):
        """Return teacher CTC logits + encoder features for frame-KD."""
        d = self.device
        out = self.model(feats.to(d), feat_lens.to(d), targets.to(d), target_lens.to(d),
                         return_features=True)
        return {"ctc_logits": out["ctc_logits"], "enc": out["enc"], "enc_lens": out["enc_lens"]}
