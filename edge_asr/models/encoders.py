"""Encoder factory — pick the Model-1 encoder by `encoder_type`.

Both encoders expose the identical interface so the transducer, decoder,
streaming session and ONNX export are encoder-agnostic:

    .forward(feats) -> (B, T', d)          # full-seq training
    .init_state(batch, device) -> State    # flat list of per-layer tensors
    .forward_chunk(chunk, state) -> (enc, new_state)
    .out_dim : int

  * "conformer" : streaming cache-aware Conformer-lite (attention).
  * "mamba"     : streaming selective-SSM (O(1) state, battery-friendly).
"""
from __future__ import annotations

from typing import Any, Dict

from .streaming_encoder import EncoderConfig, StreamingConformerEncoder
from .ssm_encoder import SSMEncoderConfig, StreamingMambaEncoder


def build_encoder(cfg: Dict[str, Any]):
    """cfg is the `encoder:` dict from a Model-1 yaml."""
    etype = cfg.get("encoder_type", "conformer")
    if etype == "conformer":
        fields = EncoderConfig.__dataclass_fields__
        kwargs = {k: v for k, v in cfg.items() if k in fields}
        return StreamingConformerEncoder(EncoderConfig(**kwargs))
    elif etype == "mamba":
        fields = SSMEncoderConfig.__dataclass_fields__
        kwargs = {k: v for k, v in cfg.items() if k in fields}
        return StreamingMambaEncoder(SSMEncoderConfig(**kwargs))
    raise ValueError(f"unknown encoder_type: {etype}")


def encoder_out_dim(cfg: Dict[str, Any]) -> int:
    return cfg["d_model"]
