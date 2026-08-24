"""Synthetic, learnable audio for smoke tests and CI.

No downloads, no external audio. Each character maps to a distinct tone
frequency; a word/utterance is the concatenation of its characters' tones
(with light noise and envelope). This produces an acoustic->text mapping a
tiny model can actually learn in a few hundred steps, so the smoke test
verifies the *whole* pipeline (loss goes down, decode emits text, export
round-trips) rather than just that code runs.

KWS classes are distinct multi-tone "chirps".
"""
from __future__ import annotations

import json
import math
import random
from typing import List

import torch

SR = 16000
_ALPHABET = " abcdefghijklmnopqrstuvwxyz'"


def _char_freq(c: str) -> float:
    idx = _ALPHABET.index(c) if c in _ALPHABET else 0
    return 200.0 + idx * 90.0  # 200..2600 Hz, distinct per char


def _tone(freq: float, dur: float, sr: int = SR) -> torch.Tensor:
    n = int(dur * sr)
    t = torch.arange(n) / sr
    env = torch.hann_window(n) if n > 1 else torch.ones(n)
    wav = 0.6 * torch.sin(2 * math.pi * freq * t) * env
    # a little harmonic content so log-mel has structure
    wav += 0.2 * torch.sin(2 * math.pi * 2 * freq * t) * env
    return wav


def text_to_wav(text: str, char_dur: float = 0.08, noise: float = 0.02) -> torch.Tensor:
    parts = [_tone(_char_freq(c), char_dur) for c in text.lower()]
    wav = torch.cat(parts) if parts else torch.zeros(int(0.1 * SR))
    wav = wav + noise * torch.randn_like(wav)
    return wav.clamp(-1.0, 1.0)


def make_synthetic_asr_manifest(path: str, n: int = 48, seed: int = 0) -> List[str]:
    rng = random.Random(seed)
    words = ["turn on the light", "call mom", "set a timer", "what time is it",
             "play music", "stop", "send a message", "start a run", "weather today",
             "open maps", "answer call", "cancel"]
    lines = []
    with open(path, "w") as f:
        for i in range(n):
            text = rng.choice(words)
            wav = text_to_wav(text)
            rec = {"array": wav.tolist(), "text": text, "duration": wav.numel() / SR}
            f.write(json.dumps(rec) + "\n")
            lines.append(text)
    return lines


def _class_chirp(label: int, rng: random.Random) -> torch.Tensor:
    base = 300.0 + label * 150.0
    dur = 0.5
    nsamp = int(dur * SR)
    t = torch.arange(nsamp) / SR
    env = torch.hann_window(nsamp)
    wav = torch.zeros(nsamp)
    for k in range(1, 4):  # class-specific chord
        wav += (0.4 / k) * torch.sin(2 * math.pi * base * k * t)
    return (wav * env + 0.03 * torch.randn(nsamp)).clamp(-1, 1)


def make_synthetic_kws_manifest(path: str, classes: List[str], n_per: int = 12, seed: int = 0):
    rng = random.Random(seed)
    with open(path, "w") as f:
        for label, name in enumerate(classes):
            for _ in range(n_per):
                wav = _class_chirp(label, rng)
                rec = {"array": wav.tolist(), "label": label, "duration": 0.5}
                f.write(json.dumps(rec) + "\n")


def make_synthetic_command_manifest(
    path: str, commands: List[str], domains: List[int], n_per: int = 12, seed: int = 0
):
    """Each command has a name (text) and a domain id. Audio is a class-
    specific chirp. Used to train the open-vocab hypernetwork + router:
    positives pair a chirp with its own name, negatives with another name."""
    rng = random.Random(seed)
    with open(path, "w") as f:
        for label, (name, dom) in enumerate(zip(commands, domains)):
            for _ in range(n_per):
                wav = _class_chirp(label, rng)
                rec = {"array": wav.tolist(), "label": label, "keyword": name,
                       "domain": dom, "duration": 0.5}
                f.write(json.dumps(rec) + "\n")
