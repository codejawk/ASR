"""Model 2: BC-ResNet-style command / keyword classifier.

Broadcasted Residual Learning (Kim et al., Qualcomm AI Research): a
resource-constrained KWS backbone that does most work with cheap 1D
temporal convs, then broadcasts a 2D frequency-conv branch across time.
This lands ~0.1-0.4 M params -> well under 1 MB int8 and cheap enough to
run continuously on the Wear-Elite eNPU.

This is the *closed-set* design (fixed command list + an "unknown/filler"
class). For a runtime-extensible command set, use the phoneme-CTC + FST
path in edge_asr/decode/keyword_ctc.py instead (see docs/ARCHITECTURE.md 4).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class BCResNetConfig:
    n_mels: int = 40  # KWS uses fewer mels than dictation ASR
    num_classes: int = 12  # e.g. 10 commands + "unknown" + "silence"
    base_channels: int = 16
    width_mult: float = 1.0


class SubSpectralNorm(nn.Module):
    """Split frequency into sub-bands and normalize each separately — helps
    KWS convergence. Freq-agnostic: a single BatchNorm(channels) is shared
    across sub-bands by folding the band index into the batch dim, so it
    works for any frequency size (falls back to 1 band when not divisible)."""

    def __init__(self, channels: int, groups: int = 4):
        super().__init__()
        self.groups = groups
        self.bn = nn.BatchNorm2d(channels)

    def forward(self, x):
        b, c, f, t = x.shape
        g = self.groups if f % self.groups == 0 else 1
        x = x.view(b, c, g, f // g, t)          # split freq into g bands
        x = x.permute(0, 2, 1, 3, 4).reshape(b * g, c, f // g, t)
        x = self.bn(x)
        x = x.view(b, g, c, f // g, t).permute(0, 2, 1, 3, 4).reshape(b, c, f, t)
        return x


class BroadcastedBlock(nn.Module):
    def __init__(self, channels: int, stride=(1, 1), dilation=1):
        super().__init__()
        # 2D frequency-depthwise conv branch
        self.freq_dw = nn.Conv2d(
            channels, channels, (3, 1), stride=stride, padding=(1, 0), groups=channels, bias=False
        )
        self.ssn = SubSpectralNorm(channels, groups=4 if channels % 4 == 0 else 1)
        # 1D temporal branch (broadcast across frequency)
        self.temp_dw = nn.Conv2d(
            channels, channels, (1, 3), padding=(0, dilation), dilation=(1, dilation),
            groups=channels, bias=False,
        )
        self.bn = nn.BatchNorm2d(channels)
        self.pw = nn.Conv2d(channels, channels, 1, bias=False)

    def forward(self, x):
        identity = x
        x = self.freq_dw(x)
        x = self.ssn(x)
        # broadcast: average over frequency, run temporal conv, add back
        aux = x.mean(dim=2, keepdim=True)
        aux = self.temp_dw(aux)
        aux = self.bn(aux)
        aux = F.silu(aux)
        aux = self.pw(aux)
        x = x + aux
        if identity.shape == x.shape:
            x = x + identity
        return F.silu(x)


class BCResNet(nn.Module):
    def __init__(self, cfg: BCResNetConfig):
        super().__init__()
        self.cfg = cfg
        c = int(cfg.base_channels * cfg.width_mult)
        self.stem = nn.Sequential(
            nn.Conv2d(1, c, (5, 5), stride=(2, 1), padding=(2, 2), bias=False),
            nn.BatchNorm2d(c),
            nn.SiLU(),
        )
        self.blocks = nn.Sequential(
            BroadcastedBlock(c, stride=(2, 1), dilation=1),
            BroadcastedBlock(c, dilation=2),
            BroadcastedBlock(c, stride=(2, 1), dilation=4),
            BroadcastedBlock(c, dilation=8),
        )
        self.head = nn.Sequential(
            nn.Conv2d(c, c * 2, 1, bias=False),
            nn.BatchNorm2d(c * 2),
            nn.SiLU(),
        )
        self.classifier = nn.Linear(c * 2, cfg.num_classes)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        """feats: (B, T, n_mels) -> logits (B, num_classes)."""
        x = feats.transpose(1, 2).unsqueeze(1)  # (B, 1, n_mels, T)
        x = self.stem(x)
        x = self.blocks(x)
        x = self.head(x)
        x = x.mean(dim=(2, 3))  # global pool
        return self.classifier(x)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
