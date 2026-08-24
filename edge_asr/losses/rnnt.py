"""RNN-Transducer loss.

Uses `torchaudio.functional.rnnt_loss` when torchaudio is installed (fast,
CUDA kernels). Otherwise falls back to a numerically-stable pure-PyTorch
log-domain forward algorithm so the whole project runs with only `torch`.

The pure fallback is O(T*U) sequential tensor ops per batch — fine for
unit tests, small pilots and understanding the lattice. For multi-thousand
hour training, install torchaudio (or warp-rnnt) to get the CUDA path.

Lattice recurrence (log domain):
    a[0,0] = 0
    a[t,u] = logaddexp( a[t-1,u] + logp_blank[t-1,u],
                        a[t,u-1] + logp_label[t,u-1] )
    loss   = -( a[T-1,U] + logp_blank[T-1,U] )
"""
from __future__ import annotations

import torch

try:  # prefer the optimized kernel when available
    import torchaudio.functional as _AF

    _HAS_TORCHAUDIO = True
except Exception:  # pragma: no cover
    _HAS_TORCHAUDIO = False


def _pure_rnnt_loss(
    logits: torch.Tensor,  # (B, T, U+1, V) raw logits
    targets: torch.Tensor,  # (B, U)
    logit_lengths: torch.Tensor,  # (B,)
    target_lengths: torch.Tensor,  # (B,)
    blank: int,
) -> torch.Tensor:
    log_probs = torch.log_softmax(logits, dim=-1)
    B, T, Up1, V = log_probs.shape
    device = log_probs.device
    losses = []
    neg_inf = float("-inf")

    for b in range(B):
        Tb = int(logit_lengths[b].item())
        Ub = int(target_lengths[b].item())
        lp = log_probs[b, :Tb, : Ub + 1, :]  # (Tb, Ub+1, V)
        tgt = targets[b, :Ub]

        logp_blank = lp[:, :, blank]  # (Tb, Ub+1)
        if Ub > 0:
            idx = tgt.view(1, Ub, 1).expand(Tb, Ub, 1)
            logp_label = lp[:, :Ub, :].gather(2, idx).squeeze(2)  # (Tb, Ub)
        else:
            logp_label = lp.new_zeros(Tb, 0)

        alpha = lp.new_full((Tb, Ub + 1), neg_inf)
        alpha[0, 0] = 0.0
        for t in range(Tb):
            for u in range(Ub + 1):
                if t == 0 and u == 0:
                    continue
                terms = []
                if t > 0:
                    terms.append(alpha[t - 1, u] + logp_blank[t - 1, u])
                if u > 0:
                    terms.append(alpha[t, u - 1] + logp_label[t, u - 1])
                alpha[t, u] = torch.logsumexp(torch.stack(terms), dim=0)

        ll = alpha[Tb - 1, Ub] + logp_blank[Tb - 1, Ub]
        losses.append(-ll)

    return torch.stack(losses).mean()


def rnnt_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    logit_lengths: torch.Tensor,
    target_lengths: torch.Tensor,
    blank: int = 0,
) -> torch.Tensor:
    """Mean RNN-T loss over the batch.

    `logits` are (B, T, U+1, V) **raw** joiner outputs (not normalized).
    Both backends handle their own log-softmax internally.
    """
    if _HAS_TORCHAUDIO:
        return _AF.rnnt_loss(
            logits=logits.float(),
            targets=targets.int(),
            logit_lengths=logit_lengths.int(),
            target_lengths=target_lengths.int(),
            blank=blank,
            reduction="mean",
        )
    return _pure_rnnt_loss(logits, targets, logit_lengths, target_lengths, blank)
