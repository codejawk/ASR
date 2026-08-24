"""Streaming SSM / Mamba encoder (selective state-space, S6).

Why an SSM encoder for a watch (the competition angle)
------------------------------------------------------
Self-attention carries a KV cache that grows with context, so streaming
attention pays O(context) memory/compute per frame. A **selective SSM
(Mamba)** instead carries a *fixed-size recurrent state* `h ∈ R^{Di×N}`,
giving **O(1) memory and compute per frame** — exactly what an always-on,
battery-bound streaming recognizer wants. Recent work (Samba-ASR 2025,
Mamba-for-streaming-ASR 2025, ConMamba) shows SSM encoders match or beat
Conformers at lower memory/latency.

This is a faithful, quantizable pure-PyTorch implementation of the S6
selective scan. The scan is a sequential recurrence here (works on CPU, no
`mamba-ssm` CUDA kernel needed); production swaps in the parallel
associative-scan kernel. Crucially, the **streaming state contract is the
same** as the Conformer encoder — a flat list of per-layer tensors — so it
is a drop-in Model-1 encoder and exports the same way.

References: Gu & Dao, *Mamba: Linear-Time Sequence Modeling with Selective
State Spaces* (2023/2024); Samba-ASR (arXiv:2501.02832); Mamba for Streaming
ASR (arXiv:2410.00070).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .streaming_encoder import Conv2dSubsampling

State = List[torch.Tensor]


@dataclass
class SSMEncoderConfig:
    input_dim: int = 80
    d_model: int = 224
    n_layers: int = 11
    d_state: int = 16          # N: SSM state size
    d_conv: int = 4            # short causal conv in the Mamba block
    expand: int = 2            # Di = expand * d_model
    dt_rank: Optional[int] = None  # defaults to ceil(d_model/16)
    subsampling_factor: int = 4
    chunk_frames: int = 32
    dropout: float = 0.1
    encoder_type: str = "mamba"  # marker

    @property
    def d_inner(self) -> int:
        return self.expand * self.d_model

    @property
    def dt_rank_(self) -> int:
        return self.dt_rank or max(1, (self.d_model + 15) // 16)

    @property
    def out_frames_per_chunk(self) -> int:
        return self.chunk_frames // self.subsampling_factor


class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x):
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return norm * self.weight


def selective_scan(
    u: torch.Tensor,      # (B, L, Di)
    delta: torch.Tensor,  # (B, L, Di)
    A: torch.Tensor,      # (Di, N)
    B: torch.Tensor,      # (B, L, N)
    C: torch.Tensor,      # (B, L, N)
    D: torch.Tensor,      # (Di,)
    h0: torch.Tensor,     # (B, Di, N)
) -> Tuple[torch.Tensor, torch.Tensor]:
    """S6 selective scan (ZOH discretization), sequential over L.

    h_t = exp(Δ_t·A)·h_{t-1} + (Δ_t·B_t)·u_t ; y_t = <C_t, h_t> + D·u_t
    Returns (y: (B,L,Di), h_L: (B,Di,N)).
    """
    dA = torch.exp(delta.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0))      # (B,L,Di,N)
    dB_u = delta.unsqueeze(-1) * B.unsqueeze(2) * u.unsqueeze(-1)          # (B,L,Di,N)
    h = h0
    ys = []
    L = u.size(1)
    for t in range(L):
        h = dA[:, t] * h + dB_u[:, t]           # (B,Di,N)
        y = (h * C[:, t].unsqueeze(1)).sum(-1)  # (B,Di)
        ys.append(y)
    y = torch.stack(ys, dim=1) + u * D.view(1, 1, -1)
    return y, h


class MambaBlock(nn.Module):
    def __init__(self, cfg: SSMEncoderConfig):
        super().__init__()
        self.cfg = cfg
        d, di, N = cfg.d_model, cfg.d_inner, cfg.d_state
        self.norm = RMSNorm(d)
        self.in_proj = nn.Linear(d, 2 * di, bias=False)
        self.conv1d = nn.Conv1d(di, di, cfg.d_conv, groups=di, padding=0, bias=True)
        self.x_proj = nn.Linear(di, cfg.dt_rank_ + 2 * N, bias=False)
        self.dt_proj = nn.Linear(cfg.dt_rank_, di, bias=True)
        # A stored in log form for stability: A = -exp(A_log), negative real.
        A = torch.arange(1, N + 1, dtype=torch.float32).repeat(di, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(di))
        self.out_proj = nn.Linear(di, d, bias=False)
        self._init_dt(di)

    def _init_dt(self, di: int, dt_min: float = 1e-3, dt_max: float = 1e-1, dt_scale: float = 1.0):
        """Standard Mamba dt initialization — critical for training.

        dt_proj weights get a scaled init, and the bias is set so that the
        initial timestep softplus(bias) is spread uniformly (in log space)
        over [dt_min, dt_max]. Without this the SSM either forgets instantly
        or never integrates, and RNN-T gets stuck emitting all-blank.
        """
        rank = self.cfg.dt_rank_
        nn.init.uniform_(self.dt_proj.weight, -rank ** -0.5 * dt_scale, rank ** -0.5 * dt_scale)
        import math

        dt = torch.exp(
            torch.rand(di) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        ).clamp_min(1e-4)
        inv_softplus = dt + torch.log(-torch.expm1(-dt))  # inverse softplus
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_softplus)

    def _params_from(self, xin: torch.Tensor):
        cfg = self.cfg
        dbl = self.x_proj(xin)  # (B,L, dt_rank+2N)
        dt, B, C = torch.split(dbl, [cfg.dt_rank_, cfg.d_state, cfg.d_state], dim=-1)
        delta = F.softplus(self.dt_proj(dt))  # (B,L,Di)
        A = -torch.exp(self.A_log)            # (Di,N)
        return delta, A, B, C

    def forward(
        self,
        x: torch.Tensor,                       # (B, L, d)
        conv_cache: Optional[torch.Tensor],    # (B, Di, d_conv-1)
        ssm_state: Optional[torch.Tensor],     # (B, Di, N)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        cfg = self.cfg
        B_, L, _ = x.shape
        res = x
        xz = self.in_proj(self.norm(x))         # (B,L,2Di)
        xin, z = xz.chunk(2, dim=-1)            # each (B,L,Di)

        # causal short conv with carried cache
        xt = xin.transpose(1, 2)                # (B,Di,L)
        pad = cfg.d_conv - 1
        if conv_cache is not None and conv_cache.size(2) > 0:
            xt = torch.cat([conv_cache, xt], dim=2)
        else:
            xt = F.pad(xt, (pad, 0))
        new_conv_cache = xt[:, :, -pad:] if pad > 0 else xt[:, :, :0]
        xconv = self.conv1d(xt)                 # (B,Di,L)
        xin = F.silu(xconv.transpose(1, 2))     # (B,L,Di)

        delta, A, Bp, Cp = self._params_from(xin)
        if ssm_state is None:
            ssm_state = x.new_zeros(B_, cfg.d_inner, cfg.d_state)
        y, new_state = selective_scan(xin, delta, A, Bp, Cp, self.D, ssm_state)
        y = y * F.silu(z)
        out = self.out_proj(y)
        return res + out, new_conv_cache, new_state


class StreamingMambaEncoder(nn.Module):
    def __init__(self, cfg: SSMEncoderConfig):
        super().__init__()
        self.cfg = cfg
        self.subsample = Conv2dSubsampling(cfg.input_dim, cfg.d_model)
        self.blocks = nn.ModuleList(MambaBlock(cfg) for _ in range(cfg.n_layers))
        self.out_dim = cfg.d_model

    # ---- training (full-seq) ----
    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        x = self.subsample(feats)
        for blk in self.blocks:
            x, _, _ = blk(x, None, None)
        return x

    # ---- streaming ----
    def init_state(self, batch: int = 1, device="cpu") -> State:
        st: State = []
        di, N, pad = self.cfg.d_inner, self.cfg.d_state, self.cfg.d_conv - 1
        for _ in range(self.cfg.n_layers):
            st.append(torch.zeros(batch, di, pad, device=device))
            st.append(torch.zeros(batch, di, N, device=device))
        return st

    def forward_chunk(self, feats_chunk: torch.Tensor, state: State) -> Tuple[torch.Tensor, State]:
        x = self.subsample(feats_chunk)
        new_state: State = []
        for i, blk in enumerate(self.blocks):
            cc, ss = state[2 * i], state[2 * i + 1]
            x, cc, ss = blk(x, cc, ss)
            new_state.append(cc)
            new_state.append(ss)
        return x, new_state

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
