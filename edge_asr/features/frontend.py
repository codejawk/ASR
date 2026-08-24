"""Acoustic frontend: 16 kHz PCM -> log-mel filterbank.

Design decisions that are *load-bearing* for on-device streaming
(see docs/ARCHITECTURE.md 1.1):

  * 80-dim log-mel, 25 ms window / 10 ms hop  -> 100 frames/s.
  * The frame rate set here is the compute unit downstream. Everything
    is O(frames); the encoder subsamples *after* this.
  * CMVN must be **causal + online** for streaming. A global mean/var
    over the full utterance silently breaks the streaming deployment,
    so `OnlineCMVN` keeps a running estimate with a warmup floor.

We build the mel filterbank by hand (no torchaudio dependency) so the
exact same features are produced on the training host and inside the
exported ONNX graph.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn


def _hz_to_mel(hz: torch.Tensor) -> torch.Tensor:
    return 2595.0 * torch.log10(1.0 + hz / 700.0)


def _mel_to_hz(mel: torch.Tensor) -> torch.Tensor:
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def mel_filterbank(
    n_mels: int,
    n_fft: int,
    sample_rate: int,
    f_min: float = 20.0,
    f_max: Optional[float] = None,
) -> torch.Tensor:
    """Return a (n_mels, n_fft//2 + 1) triangular filterbank matrix."""
    f_max = f_max or sample_rate / 2.0
    n_freqs = n_fft // 2 + 1
    all_freqs = torch.linspace(0, sample_rate / 2, n_freqs)

    m_min, m_max = _hz_to_mel(torch.tensor(f_min)), _hz_to_mel(torch.tensor(f_max))
    m_pts = torch.linspace(m_min.item(), m_max.item(), n_mels + 2)
    f_pts = _mel_to_hz(m_pts)

    fb = torch.zeros(n_mels, n_freqs)
    for m in range(1, n_mels + 1):
        left, center, right = f_pts[m - 1], f_pts[m], f_pts[m + 1]
        up = (all_freqs - left) / (center - left + 1e-8)
        down = (right - all_freqs) / (right - center + 1e-8)
        fb[m - 1] = torch.clamp(torch.minimum(up, down), min=0.0)
    return fb


class LogMelFrontend(nn.Module):
    """Waveform (B, num_samples) -> log-mel (B, T, n_mels)."""

    def __init__(
        self,
        sample_rate: int = 16000,
        n_mels: int = 80,
        frame_length_ms: float = 25.0,
        frame_shift_ms: float = 10.0,
        f_min: float = 20.0,
        f_max: Optional[float] = None,
        preemphasis: float = 0.97,
        log_floor: float = 1e-6,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.win_length = int(round(sample_rate * frame_length_ms / 1000.0))
        self.hop_length = int(round(sample_rate * frame_shift_ms / 1000.0))
        self.n_fft = 1
        while self.n_fft < self.win_length:
            self.n_fft *= 2  # next pow-2 >= win_length, FFT-friendly
        self.preemphasis = preemphasis
        self.log_floor = log_floor

        self.register_buffer("window", torch.hann_window(self.win_length), persistent=False)
        self.register_buffer(
            "fb", mel_filterbank(n_mels, self.n_fft, sample_rate, f_min, f_max), persistent=False
        )

    @property
    def frames_per_second(self) -> float:
        return self.sample_rate / self.hop_length

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)
        if self.preemphasis > 0:
            wav = torch.cat([wav[:, :1], wav[:, 1:] - self.preemphasis * wav[:, :-1]], dim=1)

        spec = torch.stft(
            wav,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=True,
            pad_mode="reflect",
            return_complex=True,
        )  # (B, n_freqs, T)
        power = spec.real.pow(2) + spec.imag.pow(2)
        mel = torch.matmul(self.fb, power)  # (B, n_mels, T)
        logmel = torch.log(torch.clamp(mel, min=self.log_floor))
        return logmel.transpose(1, 2).contiguous()  # (B, T, n_mels)


class OnlineCMVN(nn.Module):
    """Causal cepstral mean/variance normalization with a warmup floor.

    Keeps a running count/sum/sumsq so the statistics at frame t depend
    only on frames <= t. This is what makes the exact same normalization
    reproducible chunk-by-chunk at inference time. During training we run
    it over the whole (padded) utterance, which is equivalent because it
    is strictly causal.
    """

    def __init__(self, n_mels: int = 80, warmup_frames: int = 50, eps: float = 1e-5):
        super().__init__()
        self.warmup_frames = warmup_frames
        self.eps = eps
        # global priors, updated by `fit_global_stats`, used during warmup
        self.register_buffer("global_mean", torch.zeros(n_mels))
        self.register_buffer("global_std", torch.ones(n_mels))

    @torch.no_grad()
    def fit_global_stats(self, mean: torch.Tensor, std: torch.Tensor):
        self.global_mean.copy_(mean)
        self.global_std.copy_(std.clamp_min(self.eps))

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        # feats: (B, T, C). cumulative causal statistics.
        B, T, C = feats.shape
        csum = feats.cumsum(dim=1)
        csumsq = (feats * feats).cumsum(dim=1)
        counts = torch.arange(1, T + 1, device=feats.device).view(1, T, 1).float()
        run_mean = csum / counts
        run_var = (csumsq / counts) - run_mean.pow(2)
        run_std = run_var.clamp_min(self.eps).sqrt()

        # blend toward global priors during the warmup window
        w = torch.clamp(counts / max(self.warmup_frames, 1), max=1.0)
        mean = w * run_mean + (1 - w) * self.global_mean
        std = w * run_std + (1 - w) * self.global_std
        return (feats - mean) / std
