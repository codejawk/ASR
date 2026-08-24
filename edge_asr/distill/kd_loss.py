"""Knowledge-distillation losses for the streaming transducer student.

At ~10 M params, distillation from a large teacher is the single biggest
accuracy lever — bigger than any encoder-architecture change. We implement
the two forms that actually work for on-device transducers:

  * **CTC-KD** (frame-level soft targets): KL divergence between the
    teacher's and student's CTC posteriors, frame-aligned. Both models run
    at the same frame rate (same subsampling), so their time axes match.
    Temperature `tau` softens the distribution (Hinton et al. 2015).

  * **Feature-KD** (optional): MSE between teacher and student encoder
    features after a learned projection (dims differ). Cheap regularizer.

Sequence-level KD (training the student on the teacher's *hypotheses*) is
handled separately by pseudo-labeling: see `distill/teacher.py` and
`training/train_distill.py`.

Refs: Knowledge Distillation for Neural Transducers from SSL models
(arXiv:2110.03334); Fast Streaming Transducer via KD with Whisper
(arXiv:2409.13499).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def _length_mask(lengths: torch.Tensor, T: int) -> torch.Tensor:
    ar = torch.arange(T, device=lengths.device).unsqueeze(0)  # (1,T)
    return (ar < lengths.unsqueeze(1)).float()                # (B,T)


def ctc_kd_loss(
    student_logits: torch.Tensor,  # (B, T, V) raw
    teacher_logits: torch.Tensor,  # (B, T, V) raw
    enc_lens: torch.Tensor,        # (B,)
    tau: float = 2.0,
) -> torch.Tensor:
    """Temperature-scaled KL(teacher || student), masked to valid frames."""
    assert student_logits.shape == teacher_logits.shape, "frame rates must match for CTC-KD"
    B, T, V = student_logits.shape
    s_logp = F.log_softmax(student_logits / tau, dim=-1)
    t_p = F.softmax(teacher_logits / tau, dim=-1)
    t_logp = F.log_softmax(teacher_logits / tau, dim=-1)
    kl = (t_p * (t_logp - s_logp)).sum(-1)  # (B, T)
    mask = _length_mask(enc_lens, T)
    denom = mask.sum().clamp_min(1.0)
    return (kl * mask).sum() / denom * (tau * tau)


class FeatureProjector(torch.nn.Module):
    """Projects student encoder dim -> teacher encoder dim for feature-KD."""

    def __init__(self, student_dim: int, teacher_dim: int):
        super().__init__()
        self.proj = torch.nn.Linear(student_dim, teacher_dim)

    def forward(self, student_enc: torch.Tensor) -> torch.Tensor:
        return self.proj(student_enc)


def feature_kd_loss(
    student_enc: torch.Tensor,      # (B, T, Ds)
    teacher_enc: torch.Tensor,      # (B, T, Dt)
    enc_lens: torch.Tensor,
    projector: FeatureProjector,
) -> torch.Tensor:
    T = student_enc.size(1)
    proj = projector(student_enc)
    mse = ((proj - teacher_enc) ** 2).mean(-1)  # (B, T)
    mask = _length_mask(enc_lens, T)
    return (mse * mask).sum() / mask.sum().clamp_min(1.0)
