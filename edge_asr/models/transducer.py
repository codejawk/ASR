r"""Model 1: streaming RNN-T with an auxiliary CTC head.

    feats -> encoder -> (transducer head: predictor + joiner)
                     \-> (aux CTC head: linear -> vocab)

The aux CTC head costs ~50 KB, speeds convergence, and gives a cheap
non-autoregressive fallback decode path. `ctc_loss_scale` weights it.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .decoder import StatelessDecoder
from .joiner import Joiner
from .streaming_encoder import EncoderConfig, StreamingConformerEncoder
from .ssm_encoder import SSMEncoderConfig, StreamingMambaEncoder
from ..losses.rnnt import rnnt_loss


@dataclass
class TransducerConfig:
    vocab_size: int = 500
    blank: int = 0
    encoder: EncoderConfig = None
    decoder_dim: int = 256
    joiner_dim: int = 256
    context: int = 2
    ctc_loss_scale: float = 0.2

    def __post_init__(self):
        if self.encoder is None:
            self.encoder = EncoderConfig()


class Transducer(nn.Module):
    def __init__(self, cfg: TransducerConfig):
        super().__init__()
        self.cfg = cfg
        if getattr(cfg.encoder, "encoder_type", "conformer") == "mamba":
            self.encoder = StreamingMambaEncoder(cfg.encoder)
        else:
            self.encoder = StreamingConformerEncoder(cfg.encoder)
        self.decoder = StatelessDecoder(cfg.vocab_size, cfg.decoder_dim, cfg.context, cfg.blank)
        self.joiner = Joiner(self.encoder.out_dim, cfg.decoder_dim, cfg.joiner_dim, cfg.vocab_size)
        self.ctc_head = nn.Linear(self.encoder.out_dim, cfg.vocab_size)

    # -------------------------------------------------------------- training
    def forward(
        self,
        feats: torch.Tensor,
        feat_lens: torch.Tensor,
        targets: torch.Tensor,
        target_lens: torch.Tensor,
        return_features: bool = False,
    ) -> dict:
        enc = self.encoder(feats)  # (B, T', D)
        enc_lens = torch.clamp(
            (feat_lens // self.cfg.encoder.subsampling_factor), max=enc.size(1)
        )

        # ---- transducer branch
        blank = self.cfg.blank
        pad = torch.full((targets.size(0), 1), blank, dtype=targets.dtype, device=targets.device)
        dec_in = torch.cat([pad, targets], dim=1)  # prepend blank (SOS)
        pred = self.decoder(dec_in)  # (B, U+1, Dp)
        logits = self.joiner(enc, pred)  # (B, T, U+1, V)
        rnnt = rnnt_loss(logits, targets.int(), enc_lens.int(), target_lens.int(), blank)

        # ---- aux CTC branch
        ctc_logits = self.ctc_head(enc)  # (B, T, V) raw
        ctc_logp = F.log_softmax(ctc_logits, dim=-1).transpose(0, 1)  # (T, B, V)
        ctc = F.ctc_loss(
            ctc_logp, targets, enc_lens, target_lens, blank=blank, zero_infinity=True
        )

        loss = rnnt + self.cfg.ctc_loss_scale * ctc
        out = {"loss": loss, "rnnt": rnnt.detach(), "ctc": ctc.detach()}
        if return_features:
            out["enc"] = enc              # (B, T, D)  encoder features (for feature-KD)
            out["ctc_logits"] = ctc_logits  # (B, T, V) raw logits (for CTC-KD)
            out["enc_lens"] = enc_lens
        return out

    # ------------------------------------------------------------- streaming
    def init_state(self, batch: int = 1, device="cpu"):
        return self.encoder.init_state(batch, device)

    @torch.no_grad()
    def encode_chunk(self, feats_chunk: torch.Tensor, state) -> Tuple[torch.Tensor, list]:
        return self.encoder.forward_chunk(feats_chunk, state)

    def num_params(self) -> dict:
        def count(m):
            return sum(p.numel() for p in m.parameters())

        return {
            "encoder": count(self.encoder),
            "decoder": count(self.decoder),
            "joiner": count(self.joiner),
            "ctc_head": count(self.ctc_head),
            "total": count(self),
        }
