"""Host-side streaming ASR session — the reference for the on-device loop.

This is the code that would live in the watch app (C++/Kotlin on-device).
The model is a pure single-chunk function; *this* loop owns the audio ring
buffer, encoder state, endpointing and partial results. Kept in PyTorch
here so it is testable, but the control flow maps 1:1 to the on-device
host code that drives encoder.onnx / decoder.onnx / joiner.onnx on QNN.
"""
from __future__ import annotations

from typing import List, Optional

import torch

from ..features import LogMelFrontend, OnlineCMVN


class StreamingASRSession:
    def __init__(self, model, tokenizer, frontend: Optional[LogMelFrontend] = None,
                 cmvn: Optional[OnlineCMVN] = None, max_sym_per_frame: int = 3,
                 endpoint_silence_frames: int = 40):
        self.model = model.eval()
        self.tok = tokenizer
        self.frontend = frontend or LogMelFrontend(n_mels=model.cfg.encoder.input_dim)
        self.cmvn = cmvn or OnlineCMVN(n_mels=model.cfg.encoder.input_dim)
        self.max_sym = max_sym_per_frame
        self.endpoint_silence_frames = endpoint_silence_frames
        self.reset()

    def reset(self):
        self.state = self.model.init_state(batch=1)
        self.blank = self.model.cfg.blank
        self.context = self.model.cfg.context
        self.ctx = [self.blank] * self.context
        self.hyp: List[int] = []
        self._blank_run = 0
        self._feat_carry = torch.zeros(0, self.model.cfg.encoder.input_dim)

    @property
    def text(self) -> str:
        return self.tok.decode(self.hyp)

    @torch.no_grad()
    def accept_waveform(self, wav_chunk: torch.Tensor) -> str:
        """Feed raw 16 kHz PCM (any length). Returns current partial text.
        Emits chunks to the encoder whenever a full acoustic chunk fills."""
        feats = self.frontend(wav_chunk).squeeze(0)  # (t, C)
        feats = self.cmvn(feats.unsqueeze(0)).squeeze(0)
        self._feat_carry = torch.cat([self._feat_carry, feats], dim=0)

        chunk = self.model.cfg.encoder.chunk_frames
        while self._feat_carry.size(0) >= chunk:
            block = self._feat_carry[:chunk]
            self._feat_carry = self._feat_carry[chunk:]
            self._process_block(block)
        return self.text

    def _process_block(self, block: torch.Tensor):
        enc_chunk, self.state = self.model.encode_chunk(block.unsqueeze(0), self.state)
        enc_chunk = enc_chunk.squeeze(0)
        for t in range(enc_chunk.size(0)):
            enc_t = enc_chunk[t : t + 1]
            emitted = 0
            frame_blank = True
            while emitted < self.max_sym:
                prev = torch.tensor([self.ctx[-self.context :]])
                pred = self.model.decoder.step(prev)
                logit = self.model.joiner(enc_t, pred)
                tok = int(logit.argmax(-1).item())
                if tok == self.blank:
                    break
                frame_blank = False
                self.hyp.append(tok)
                self.ctx.append(tok)
                emitted += 1
            self._blank_run = self._blank_run + 1 if frame_blank else 0

    @property
    def endpointed(self) -> bool:
        """True once trailing silence exceeds the threshold — the signal to
        finalize the utterance and hand the result to the app."""
        return self._blank_run >= self.endpoint_silence_frames
