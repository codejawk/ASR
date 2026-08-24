"""End-to-end smoke test: the whole toolchain on synthetic audio.

Verifies, with no downloads:
  1. features produce sane log-mel
  2. Model 1 trains (loss decreases) on synthetic speech-like audio
  3. greedy + streaming decode both emit text and agree in structure
  4. streaming session (host-loop reference) runs
  5. ONNX export of the 3 graphs (skipped cleanly if `onnx` missing)

Run:  python -m pytest tests/ -q      (or)   python tests/test_smoke_end_to_end.py
"""
from __future__ import annotations

import os
import tempfile

import torch

from edge_asr.data import make_synthetic_asr_manifest, ManifestDataset, load_tokenizer, collate_asr
from edge_asr.decode import greedy_search, streaming_greedy_search
from edge_asr.features import LogMelFrontend, OnlineCMVN
from edge_asr.models import EncoderConfig, Transducer, TransducerConfig
from edge_asr.runtime import StreamingASRSession
from torch.utils.data import DataLoader


def _tiny_model(vocab):
    enc = EncoderConfig(input_dim=40, d_model=64, n_layers=2, n_heads=2, ff_dim=128,
                        conv_kernel=7, subsampling_factor=4, chunk_frames=16, left_context_chunks=2)
    return Transducer(TransducerConfig(vocab_size=vocab, encoder=enc, decoder_dim=64,
                                       joiner_dim=64, ctc_loss_scale=0.2))


def test_end_to_end():
    torch.manual_seed(0)
    tmp = tempfile.mkdtemp()
    manifest = os.path.join(tmp, "asr.jsonl")
    make_synthetic_asr_manifest(manifest, n=32, seed=0)

    tok = load_tokenizer("char")
    fe = LogMelFrontend(n_mels=40)
    cmvn = OnlineCMVN(n_mels=40)
    ds = ManifestDataset(manifest, fe, cmvn, tok, task="asr")
    dl = DataLoader(ds, batch_size=8, shuffle=True, collate_fn=collate_asr)

    # feature sanity
    feats0, toks0 = ds[0]
    assert feats0.dim() == 2 and feats0.size(1) == 40
    assert torch.isfinite(feats0).all()

    model = _tiny_model(tok.vocab_size)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

    losses = []
    model.train()
    it = iter(dl)
    for step in range(60):
        try:
            f, fl, t, tl = next(it)
        except StopIteration:
            it = iter(dl)
            f, fl, t, tl = next(it)
        out = model(f, fl, t, tl)
        opt.zero_grad(); out["loss"].backward(); opt.step()
        losses.append(out["loss"].item())

    assert losses[-1] < losses[0], f"loss did not decrease: {losses[0]:.2f} -> {losses[-1]:.2f}"

    # decode paths run and return strings
    model.eval()
    h_full = tok.decode(greedy_search(model, feats0))
    h_stream = tok.decode(streaming_greedy_search(model, feats0))
    assert isinstance(h_full, str) and isinstance(h_stream, str)

    # streaming host session runs on raw waveform
    wav = torch.tensor(ds.items[0]["array"], dtype=torch.float32)
    sess = StreamingASRSession(model, tok, fe, cmvn)
    partial = sess.accept_waveform(wav)
    assert isinstance(partial, str)

    print(f"loss {losses[0]:.2f} -> {losses[-1]:.2f} | full='{h_full}' stream='{h_stream}'")


def test_onnx_export_optional():
    tok = load_tokenizer("char")
    model = _tiny_model(tok.vocab_size)
    try:
        import onnx  # noqa
    except Exception:
        print("onnx not installed; export path skipped (expected on this box)")
        return
    from edge_asr.export import export_model1_streaming
    tmp = tempfile.mkdtemp()
    paths = export_model1_streaming(model, tmp)
    assert all(os.path.exists(p) for p in paths)


if __name__ == "__main__":
    test_end_to_end()
    test_onnx_export_optional()
    print("OK")
