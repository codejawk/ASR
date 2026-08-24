"""Cache-aware streaming encoder for Model 1.

This is a *Conformer-lite* streaming encoder that captures the mechanics
that actually determine whether an ASR model fits and streams on a watch:

  * early convolutional **subsampling** (100 Hz -> 25 Hz) so downstream
    self-attention runs on 4x fewer frames;
  * **cache-aware chunked** processing: attention key/value context and
    the depthwise-conv left context are carried forward between chunks as
    explicit state tensors, so every frame is encoded exactly once;
  * **causal** depthwise convolutions (left padding only) and bounded
    left attention context — no lookahead beyond the current chunk.

Production note
---------------
The reference production encoder is icefall's streaming **Zipformer**
(U-Net variable frame rate, BiasNorm, ScaledAdam), which has better
accuracy-per-parameter. `EncoderConfig` deliberately mirrors the knobs
you would scale in that recipe (see configs/model1_general.yaml and
docs/ARCHITECTURE.md 3.3). This implementation is a faithful, quantizable
stand-in that trains and exports through ONNX/QNN so you can de-risk the
*toolchain* before committing GPU-weeks to a Zipformer run.

Streaming state contract (also the ONNX/QNN I/O contract)
--------------------------------------------------------
`forward_chunk(x, state)` consumes one fixed-size chunk plus a state
tuple and returns (encoded_chunk, new_state). Shapes are static, which
is exactly what the HTP backend needs.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

State = List[torch.Tensor]  # per-layer: [attn_kv_cache, conv_cache], flattened


@dataclass
class EncoderConfig:
    encoder_type: str = "conformer"
    input_dim: int = 80
    # Per-block model dim. Kept modest so the whole encoder lands ~11 M params
    # at int8 -> ~11 MB. Mirrors Zipformer's per-stack `encoder-dim`.
    d_model: int = 256
    n_layers: int = 12
    n_heads: int = 4
    ff_dim: int = 512
    conv_kernel: int = 15
    subsampling_factor: int = 4  # 100 Hz -> 25 Hz
    # streaming geometry (frames are *pre-subsampling* 100 Hz frames)
    chunk_frames: int = 32  # 320 ms acoustic chunk
    left_context_chunks: int = 4  # attention memory horizon
    dropout: float = 0.1

    @property
    def out_frames_per_chunk(self) -> int:
        return self.chunk_frames // self.subsampling_factor


class Conv2dSubsampling(nn.Module):
    """Two stride-2 convs -> /4 in time and freq. 100 Hz -> 25 Hz."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, out_dim, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_dim, out_dim, 3, stride=2, padding=1),
            nn.ReLU(),
        )
        freq_after = ((in_dim + 1) // 2 + 1) // 2
        self.out = nn.Linear(out_dim * freq_after, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F)
        x = x.unsqueeze(1)  # (B, 1, T, F)
        x = self.conv(x)  # (B, C, T', F')
        b, c, t, f = x.shape
        x = x.transpose(1, 2).contiguous().view(b, t, c * f)
        return self.out(x)  # (B, T', d_model)


class FeedForward(nn.Module):
    def __init__(self, d_model: int, ff_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, ff_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return x + 0.5 * self.net(x)


class CachedSelfAttention(nn.Module):
    """MHA where each chunk attends to a fixed left KV-cache + itself."""

    def __init__(self, d_model: int, n_heads: int, dropout: float, left_cache: int):
        super().__init__()
        assert d_model % n_heads == 0
        self.h = n_heads
        self.dk = d_model // n_heads
        self.left_cache = left_cache
        self.norm = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)

    def _split(self, x):
        b, t, _ = x.shape
        return x.view(b, t, self.h, self.dk).transpose(1, 2)  # (B,h,t,dk)

    def forward(
        self, x: torch.Tensor, kv_cache: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: (B, T, d). kv_cache: (B, 2, Lc, d) holding cached K and V inputs.
        residual = x
        xn = self.norm(x)
        qkv = self.qkv(xn)
        q, k, v = qkv.chunk(3, dim=-1)

        if kv_cache is not None and kv_cache.size(2) > 0:
            k_cat = torch.cat([kv_cache[:, 0], k], dim=1)
            v_cat = torch.cat([kv_cache[:, 1], v], dim=1)
        else:
            k_cat, v_cat = k, v

        qs, ks, vs = self._split(q), self._split(k_cat), self._split(v_cat)
        attn = torch.matmul(qs, ks.transpose(-2, -1)) / (self.dk ** 0.5)
        attn = attn.softmax(dim=-1)
        out = torch.matmul(attn, vs)  # (B,h,Tq,dk)
        b, h, tq, dk = out.shape
        out = out.transpose(1, 2).contiguous().view(b, tq, h * dk)
        out = residual + self.drop(self.proj(out))

        # new cache = last `left_cache` frames of the *inputs* k,v
        new_k = k_cat[:, -self.left_cache :]
        new_v = v_cat[:, -self.left_cache :]
        new_cache = torch.stack([new_k, new_v], dim=1)  # (B,2,Lc,d)
        return out, new_cache


class CausalConvModule(nn.Module):
    """Conformer conv module with left-only padding + carried conv cache."""

    def __init__(self, d_model: int, kernel: int, dropout: float):
        super().__init__()
        self.kernel = kernel
        self.norm = nn.LayerNorm(d_model)
        self.pw1 = nn.Conv1d(d_model, 2 * d_model, 1)
        self.dw = nn.Conv1d(d_model, d_model, kernel, groups=d_model)  # no padding; we pad w/ cache
        self.bn = nn.BatchNorm1d(d_model)
        self.pw2 = nn.Conv1d(d_model, d_model, 1)
        self.drop = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, conv_cache: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: (B, T, d)
        residual = x
        x = self.norm(x).transpose(1, 2)  # (B, d, T)
        x = self.pw1(x)
        x = F.glu(x, dim=1)

        pad = self.kernel - 1
        if conv_cache is not None and conv_cache.size(2) > 0:
            x = torch.cat([conv_cache, x], dim=2)
        else:
            x = F.pad(x, (pad, 0))
        new_cache = x[:, :, -pad:] if pad > 0 else x[:, :, :0]

        x = self.dw(x)
        x = self.bn(x)
        x = F.silu(x)
        x = self.pw2(x)
        x = x.transpose(1, 2)
        return residual + self.drop(x), new_cache


class ConformerBlock(nn.Module):
    def __init__(self, cfg: EncoderConfig, left_cache_frames: int):
        super().__init__()
        self.ff1 = FeedForward(cfg.d_model, cfg.ff_dim, cfg.dropout)
        self.attn = CachedSelfAttention(cfg.d_model, cfg.n_heads, cfg.dropout, left_cache_frames)
        self.conv = CausalConvModule(cfg.d_model, cfg.conv_kernel, cfg.dropout)
        self.ff2 = FeedForward(cfg.d_model, cfg.ff_dim, cfg.dropout)
        self.norm = nn.LayerNorm(cfg.d_model)

    def forward(self, x, kv_cache, conv_cache):
        x = self.ff1(x)
        x, kv_cache = self.attn(x, kv_cache)
        x, conv_cache = self.conv(x, conv_cache)
        x = self.ff2(x)
        return self.norm(x), kv_cache, conv_cache


class StreamingConformerEncoder(nn.Module):
    def __init__(self, cfg: EncoderConfig):
        super().__init__()
        self.cfg = cfg
        self.subsample = Conv2dSubsampling(cfg.input_dim, cfg.d_model)
        # cache measured in *subsampled* frames
        self.left_cache_frames = cfg.left_context_chunks * cfg.out_frames_per_chunk
        self.blocks = nn.ModuleList(
            ConformerBlock(cfg, self.left_cache_frames) for _ in range(cfg.n_layers)
        )
        self.out_dim = cfg.d_model

    # ---- non-streaming (training) path -------------------------------------
    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        """feats: (B, T, F) -> (B, T', d). Full-context conv but causal attn
        cache is empty here; used for parallel training over full utterances."""
        x = self.subsample(feats)
        kv = [None] * self.cfg.n_layers
        cc = [None] * self.cfg.n_layers
        for i, blk in enumerate(self.blocks):
            x, kv[i], cc[i] = blk(x, kv[i], cc[i])
        return x

    # ---- streaming (inference / export) path -------------------------------
    def init_state(self, batch: int = 1, device="cpu") -> State:
        st: State = []
        d = self.cfg.d_model
        pad = self.cfg.conv_kernel - 1
        for _ in range(self.cfg.n_layers):
            st.append(torch.zeros(batch, 2, self.left_cache_frames, d, device=device))
            st.append(torch.zeros(batch, d, pad, device=device))
        return st

    def forward_chunk(self, feats_chunk: torch.Tensor, state: State) -> Tuple[torch.Tensor, State]:
        """feats_chunk: (B, chunk_frames, F). Returns (B, out_frames, d)."""
        x = self.subsample(feats_chunk)
        new_state: State = []
        for i, blk in enumerate(self.blocks):
            kv, cc = state[2 * i], state[2 * i + 1]
            x, kv, cc = blk(x, kv, cc)
            new_state.append(kv)
            new_state.append(cc)
        return x, new_state
