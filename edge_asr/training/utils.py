from __future__ import annotations

import random
from typing import Any, Dict

import numpy as np
import torch
import yaml

from ..models import (
    BCResNet,
    BCResNetConfig,
    EncoderConfig,
    Transducer,
    TransducerConfig,
)
from ..models.ssm_encoder import SSMEncoderConfig


def set_seed(seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_config(path: str) -> Dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def _build_encoder_config(enc_cfg: Dict[str, Any]):
    etype = enc_cfg.get("encoder_type", "conformer")
    if etype == "mamba":
        fields = SSMEncoderConfig.__dataclass_fields__
        return SSMEncoderConfig(**{k: v for k, v in enc_cfg.items() if k in fields})
    fields = EncoderConfig.__dataclass_fields__
    return EncoderConfig(**{k: v for k, v in enc_cfg.items() if k in fields})


def build_model1(cfg: Dict[str, Any], vocab_size: int) -> Transducer:
    enc = _build_encoder_config(cfg["encoder"])
    tcfg = TransducerConfig(
        vocab_size=vocab_size,
        blank=cfg.get("blank", 0),
        encoder=enc,
        decoder_dim=cfg["decoder_dim"],
        joiner_dim=cfg["joiner_dim"],
        context=cfg.get("context", 2),
        ctc_loss_scale=cfg.get("ctc_loss_scale", 0.2),
    )
    return Transducer(tcfg)


def build_model2(cfg: Dict[str, Any]) -> BCResNet:
    return BCResNet(BCResNetConfig(**cfg["bcresnet"]))


def configure_optimizer(model, lr: float, weight_decay: float = 1e-2):
    """AdamW with weight-decay correctly excluded from parameters that must
    not be decayed: all 1-D params (biases, norm gains, the SSM `D`) and the
    SSM state matrix `A_log` / `dt` bias. Decaying these corrupts Mamba
    training (the model gets stuck emitting all-blank). Standard practice.
    """
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim <= 1 or name.endswith("A_log") or name.endswith(".D") or "dt_proj.bias" in name:
            no_decay.append(p)
        else:
            decay.append(p)
    groups = [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=lr, betas=(0.9, 0.98))
