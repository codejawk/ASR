"""Manifest-driven datasets.

A manifest is JSONL, one utterance per line:

    {"audio": "/path/a.wav", "text": "turn on the light", "duration": 1.8}

For KWS the field is "label" (an int class id) instead of "text".
Audio is loaded with the stdlib `wave` module (16 kHz mono PCM) so there
is no torchaudio dependency; a synthetic in-memory path is supported for
smoke tests via the "wave_tensor" key (a torch.save'd tensor path) or an
"array" inlined list.
"""
from __future__ import annotations

import json
import wave
from typing import Callable, List, Optional

import torch
from torch.utils.data import Dataset

from ..features import LogMelFrontend, OnlineCMVN


def load_wav(path: str, sr: int = 16000) -> torch.Tensor:
    if path.endswith(".pt"):
        return torch.load(path)
    with wave.open(path, "rb") as w:
        assert w.getframerate() == sr, f"expected {sr} Hz, got {w.getframerate()}"
        frames = w.readframes(w.getnframes())
    return torch.frombuffer(bytearray(frames), dtype=torch.int16).float() / 32768.0


class ManifestDataset(Dataset):
    def __init__(
        self,
        manifest: str,
        frontend: LogMelFrontend,
        cmvn: Optional[OnlineCMVN] = None,
        tokenizer=None,
        task: str = "asr",  # "asr" or "kws"
        wav_augment: Optional[Callable] = None,
        spec_augment: Optional[Callable] = None,
        sr: int = 16000,
    ):
        self.items = [json.loads(l) for l in open(manifest) if l.strip()]
        self.frontend = frontend
        self.cmvn = cmvn
        self.tokenizer = tokenizer
        self.task = task
        self.wav_augment = wav_augment
        self.spec_augment = spec_augment
        self.sr = sr

    def __len__(self):
        return len(self.items)

    def _wav(self, item) -> torch.Tensor:
        if "array" in item:
            return torch.tensor(item["array"], dtype=torch.float32)
        return load_wav(item["audio"], self.sr)

    def __getitem__(self, i):
        item = self.items[i]
        wav = self._wav(item)
        if self.wav_augment is not None:
            wav = self.wav_augment(wav)
        feats = self.frontend(wav).squeeze(0)  # (T, C)
        if self.cmvn is not None:
            feats = self.cmvn(feats.unsqueeze(0)).squeeze(0)
        if self.spec_augment is not None:
            feats = self.spec_augment(feats)

        if self.task == "asr":
            tokens = torch.tensor(self.tokenizer.encode(item["text"]), dtype=torch.long)
            return feats, tokens
        else:
            return feats, torch.tensor(item["label"], dtype=torch.long)


def collate_asr(batch):
    feats, tokens = zip(*batch)
    feat_lens = torch.tensor([f.size(0) for f in feats], dtype=torch.long)
    tok_lens = torch.tensor([t.size(0) for t in tokens], dtype=torch.long)
    C = feats[0].size(1)
    Tmax = int(feat_lens.max())
    Umax = int(tok_lens.max()) if int(tok_lens.max()) > 0 else 1
    padded_f = torch.zeros(len(feats), Tmax, C)
    padded_t = torch.zeros(len(feats), Umax, dtype=torch.long)
    for i, (f, t) in enumerate(zip(feats, tokens)):
        padded_f[i, : f.size(0)] = f
        padded_t[i, : t.size(0)] = t
    return padded_f, feat_lens, padded_t, tok_lens


def collate_kws(batch):
    feats, labels = zip(*batch)
    feat_lens = torch.tensor([f.size(0) for f in feats], dtype=torch.long)
    C = feats[0].size(1)
    Tmax = int(feat_lens.max())
    padded_f = torch.zeros(len(feats), Tmax, C)
    for i, f in enumerate(feats):
        padded_f[i, : f.size(0)] = f
    return padded_f, torch.stack(labels)
