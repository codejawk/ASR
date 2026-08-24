"""Data augmentation for the wrist domain.

At this model scale, augmentation moves WER more than architecture. The
two that matter most on a watch: additive noise (MUSAN-style + your own
in-the-wild recordings) and reverberation (RIR convolution). SpecAugment
regularizes the encoder.

`NoiseInjection` reads a directory of noise wavs when given one; with no
corpus it synthesizes coloured noise so the pipeline still runs.
"""
from __future__ import annotations

import glob
import os
import random
from typing import Optional

import torch


class SpecAugment:
    def __init__(self, freq_masks=2, freq_width=15, time_masks=2, time_width=25, prob=0.8):
        self.fm, self.fw = freq_masks, freq_width
        self.tm, self.tw = time_masks, time_width
        self.prob = prob

    def __call__(self, feats: torch.Tensor) -> torch.Tensor:
        # feats: (T, C)
        if random.random() > self.prob:
            return feats
        feats = feats.clone()
        T, C = feats.shape
        for _ in range(self.fm):
            w = random.randint(0, min(self.fw, C))
            if w == 0:
                continue
            f0 = random.randint(0, C - w)
            feats[:, f0 : f0 + w] = 0.0
        for _ in range(self.tm):
            w = random.randint(0, min(self.tw, T))
            if w == 0:
                continue
            t0 = random.randint(0, T - w)
            feats[t0 : t0 + w, :] = 0.0
        return feats


class NoiseInjection:
    def __init__(self, noise_dir: Optional[str] = None, snr_db_range=(5, 20), prob=0.5, sr=16000):
        self.noise_files = sorted(glob.glob(os.path.join(noise_dir, "**/*.wav"), recursive=True)) \
            if noise_dir and os.path.isdir(noise_dir) else []
        self.snr_range = snr_db_range
        self.prob = prob
        self.sr = sr

    def _sample_noise(self, n: int, device) -> torch.Tensor:
        if self.noise_files:
            import wave

            path = random.choice(self.noise_files)
            with wave.open(path, "rb") as w:
                frames = w.readframes(w.getnframes())
            data = torch.frombuffer(bytearray(frames), dtype=torch.int16).float() / 32768.0
            if data.numel() < n:
                reps = (n // max(data.numel(), 1)) + 1
                data = data.repeat(reps)
            start = random.randint(0, data.numel() - n)
            return data[start : start + n].to(device)
        # fallback: coloured noise
        white = torch.randn(n, device=device)
        # simple 1-pole lowpass to make it less white/harsh
        out = torch.empty_like(white)
        a = 0.9
        prev = 0.0
        for i in range(n):
            prev = a * prev + (1 - a) * white[i]
            out[i] = prev
        return out

    def __call__(self, wav: torch.Tensor) -> torch.Tensor:
        if random.random() > self.prob:
            return wav
        noise = self._sample_noise(wav.numel(), wav.device)
        sig_p = wav.pow(2).mean().clamp_min(1e-8)
        noise_p = noise.pow(2).mean().clamp_min(1e-8)
        snr = random.uniform(*self.snr_range)
        scale = (sig_p / (noise_p * (10 ** (snr / 10)))).sqrt()
        return (wav + scale * noise).clamp(-1.0, 1.0)
