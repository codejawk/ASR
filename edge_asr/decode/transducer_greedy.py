"""Greedy transducer decoding (what ships on the watch).

Two entry points:
  * greedy_search      — full (non-streaming) encoder output in one shot.
  * streaming_greedy_search — chunk-by-chunk, carrying encoder state; this
    mirrors the on-device host loop where the streaming loop lives in host
    code and the model is a pure single-chunk function.

Both use the stateless predictor's `step()` and a max-symbols-per-frame
guard to prevent runaway emission.
"""
from __future__ import annotations

from typing import List

import torch


def _decode_from_encoder(model, enc: torch.Tensor, max_sym_per_frame: int = 3) -> List[int]:
    device = enc.device
    blank = model.cfg.blank
    context = model.cfg.context
    hyp: List[int] = []
    ctx = [blank] * context

    T = enc.size(0)
    for t in range(T):
        enc_t = enc[t : t + 1]  # (1, D)
        emitted = 0
        while emitted < max_sym_per_frame:
            prev = torch.tensor([ctx[-context:]], device=device)
            pred = model.decoder.step(prev)  # (1, Dp)
            logit = model.joiner(enc_t, pred)  # (1, V)
            tok = int(logit.argmax(-1).item())
            if tok == blank:
                break
            hyp.append(tok)
            ctx.append(tok)
            emitted += 1
    return hyp


@torch.no_grad()
def greedy_search(model, feats: torch.Tensor, max_sym_per_frame: int = 3) -> List[int]:
    """feats: (T, C) single utterance. Returns token ids (no blanks)."""
    model.eval()
    enc = model.encoder(feats.unsqueeze(0)).squeeze(0)  # (T', D)
    return _decode_from_encoder(model, enc, max_sym_per_frame)


@torch.no_grad()
def streaming_greedy_search(
    model, feats: torch.Tensor, max_sym_per_frame: int = 3
) -> List[int]:
    """Chunked decode over a full utterance, exercising the state contract."""
    model.eval()
    cfg = model.cfg.encoder
    chunk = cfg.chunk_frames
    state = model.init_state(batch=1, device=feats.device)
    blank = model.cfg.blank
    context = model.cfg.context
    hyp: List[int] = []
    ctx = [blank] * context

    T = feats.size(0)
    for start in range(0, T, chunk):
        block = feats[start : start + chunk]
        if block.size(0) < chunk:  # pad final short chunk to static shape
            block = torch.nn.functional.pad(block, (0, 0, 0, chunk - block.size(0)))
        enc_chunk, state = model.encode_chunk(block.unsqueeze(0), state)
        enc_chunk = enc_chunk.squeeze(0)  # (out_frames, D)
        for t in range(enc_chunk.size(0)):
            enc_t = enc_chunk[t : t + 1]
            emitted = 0
            while emitted < max_sym_per_frame:
                prev = torch.tensor([ctx[-context:]], device=feats.device)
                pred = model.decoder.step(prev)
                logit = model.joiner(enc_t, pred)
                tok = int(logit.argmax(-1).item())
                if tok == blank:
                    break
                hyp.append(tok)
                ctx.append(tok)
                emitted += 1
    return hyp
