"""ONNX export with **static shapes** — the HTP/QNN contract.

Model 1 exports as three graphs (the sherpa-onnx pattern), because QNN's
EP does not support Loop/If: the streaming loop lives in host code and each
graph is a pure single-step function.

    encoder.onnx : (feats_chunk, *state_in)  -> (enc_chunk, *state_out)
    decoder.onnx : (prev_tokens)             -> (pred_vec)
    joiner.onnx  : (enc_t, pred_vec)         -> (logits)

Model 2 exports as a single classifier graph.

Requires the `onnx` package for `torch.onnx.export`. If it is missing the
functions raise a clear, actionable error instead of failing obscurely.
"""
from __future__ import annotations

import os
from typing import List

import torch
import torch.nn as nn


def _require_onnx():
    try:
        import onnx  # noqa: F401
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "ONNX export needs the `onnx` package. Install with:\n"
            "    pip install onnx\n"
            "(onnxruntime alone is not enough for torch.onnx.export)."
        ) from e


def _export(module, args, path, input_names, output_names, opset):
    """Export via the legacy TorchScript exporter when available.

    The newer dynamo exporter emits graphs whose shape metadata ORT's int8
    quantizer rejects (shape-inference mismatch). The legacy exporter
    (`dynamo=False`) produces clean, quantizable graphs. We fall back to the
    default path only if the legacy one is unavailable.
    """
    try:
        torch.onnx.export(
            module, args, path,
            input_names=input_names, output_names=output_names,
            opset_version=opset, do_constant_folding=True, dynamo=False,
        )
    except TypeError:
        torch.onnx.export(
            module, args, path,
            input_names=input_names, output_names=output_names,
            opset_version=opset, do_constant_folding=True,
        )


class _EncoderChunk(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.n_layers = model.cfg.encoder.n_layers

    def forward(self, feats_chunk, *state):
        enc, new_state = self.model.encoder.forward_chunk(feats_chunk, list(state))
        return (enc, *new_state)


class _DecoderStep(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, prev_tokens):
        return self.model.decoder.step(prev_tokens)


class _JoinerStep(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, enc_t, pred_vec):
        return self.model.joiner(enc_t, pred_vec)


def export_model1_streaming(model, out_dir: str, opset: int = 17) -> List[str]:
    _require_onnx()
    os.makedirs(out_dir, exist_ok=True)
    model.eval()
    cfg = model.cfg
    ecfg = cfg.encoder
    F = ecfg.input_dim

    # --- encoder chunk (static shapes) ---
    state = model.init_state(batch=1)
    feats_chunk = torch.zeros(1, ecfg.chunk_frames, F)
    enc_wrap = _EncoderChunk(model)
    state_in_names = []
    state_out_names = []
    for i in range(ecfg.n_layers):
        state_in_names += [f"kv_in_{i}", f"cc_in_{i}"]
        state_out_names += [f"kv_out_{i}", f"cc_out_{i}"]
    enc_path = os.path.join(out_dir, "encoder.onnx")
    _export(enc_wrap, (feats_chunk, *state), enc_path,
            ["feats_chunk", *state_in_names], ["enc_chunk", *state_out_names], opset)

    # --- decoder step ---
    prev = torch.zeros(1, cfg.context, dtype=torch.long)
    dec_path = os.path.join(out_dir, "decoder.onnx")
    _export(_DecoderStep(model), (prev,), dec_path, ["prev_tokens"], ["pred_vec"], opset)

    # --- joiner step ---
    enc_t = torch.zeros(1, model.encoder.out_dim)
    pred_vec = torch.zeros(1, cfg.decoder_dim)
    joi_path = os.path.join(out_dir, "joiner.onnx")
    _export(_JoinerStep(model), (enc_t, pred_vec), joi_path, ["enc_t", "pred_vec"], ["logits"], opset)
    return [enc_path, dec_path, joi_path]


def export_model2(model, n_mels: int, num_frames: int, out_dir: str, opset: int = 17) -> str:
    _require_onnx()
    os.makedirs(out_dir, exist_ok=True)
    model.eval()
    feats = torch.zeros(1, num_frames, n_mels)
    path = os.path.join(out_dir, "command.onnx")
    _export(model, (feats,), path, ["feats"], ["logits"], opset)
    return path
