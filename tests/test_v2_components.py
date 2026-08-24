"""Smoke tests for the competition-grade v2 components:
  * Mamba/SSM streaming encoder (trains + streams)
  * open-vocab hypernetwork command model + router
  * flash-paged Mixture-of-Specialists

Run: python tests/test_v2_components.py   (or pytest tests/)
"""
from __future__ import annotations

import os
import tempfile

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from edge_asr.data import (ManifestDataset, collate_asr, load_tokenizer,
                           make_synthetic_asr_manifest)
from edge_asr.features import LogMelFrontend, OnlineCMVN
from edge_asr.models import (CommandModel, CommandModelConfig, SSMEncoderConfig,
                             Transducer, TransducerConfig, encode_keyword)
from edge_asr.moe import ExpertPager, MixtureOfSpecialists
from edge_asr.training import configure_optimizer


def _mamba_transducer(vocab):
    enc = SSMEncoderConfig(input_dim=40, d_model=96, n_layers=3, d_state=8, chunk_frames=16)
    return Transducer(TransducerConfig(vocab_size=vocab, encoder=enc, decoder_dim=64, joiner_dim=64))


def test_mamba_encoder_trains_and_streams():
    torch.manual_seed(0)
    tmp = tempfile.mkdtemp()
    man = os.path.join(tmp, "a.jsonl")
    make_synthetic_asr_manifest(man, n=32, seed=0)
    tok = load_tokenizer("char")
    fe, cmvn = LogMelFrontend(n_mels=40), OnlineCMVN(n_mels=40)
    ds = ManifestDataset(man, fe, cmvn, tok, task="asr")
    dl = DataLoader(ds, batch_size=8, shuffle=True, collate_fn=collate_asr)
    m = _mamba_transducer(tok.vocab_size)
    opt = configure_optimizer(m, lr=2e-3)   # excludes wd from SSM params
    losses = []
    it = iter(dl)
    for _ in range(120):
        try:
            f, fl, t, tl = next(it)
        except StopIteration:
            it = iter(dl); f, fl, t, tl = next(it)
        out = m(f, fl, t, tl)
        opt.zero_grad(); out["loss"].backward(); opt.step()
        losses.append(out["loss"].item())
    assert losses[-1] < 0.5 * losses[0], f"mamba did not learn: {losses[0]:.1f}->{losses[-1]:.1f}"
    # streaming state contract
    st = m.init_state(1)
    enc, st2 = m.encode_chunk(torch.zeros(1, 16, 40), st)
    assert enc.shape[0] == 1 and len(st2) == 2 * 3
    print(f"mamba OK: loss {losses[0]:.1f} -> {losses[-1]:.2f}")


def test_command_model_hypernet_and_router():
    torch.manual_seed(0)
    cfg = CommandModelConfig(n_mels=40, num_domains=3, embed_dim=96, enc_channels=48)
    m = CommandModel(cfg)
    B, T = 6, 50
    feats = torch.randn(B, T, 40)
    kw = torch.tensor([encode_keyword("call") for _ in range(B)])
    out = m(feats, kw)
    assert out["detect"].shape == (B,)
    assert out["router"].shape == (B, 3)
    assert out["speaker"].shape[0] == B
    # embeddings are unit norm
    assert torch.allclose(out["embed"].norm(dim=-1), torch.ones(B), atol=1e-4)
    print(f"command model OK: {m.num_params()/1e6:.3f} M params")


def test_flash_paged_moe():
    torch.manual_seed(0)
    tok = load_tokenizer("char")
    tmp = tempfile.mkdtemp()
    model = _mamba_transducer(tok.vocab_size)
    paths = {}
    for i, name in enumerate(["comms", "media", "info", "control"]):
        p = os.path.join(tmp, f"e{i}.pt")
        torch.save({"model": model.state_dict()}, p)
        paths[i] = p
    pager = ExpertPager(paths, builder=lambda: _mamba_transducer(tok.vocab_size),
                        resident_capacity=1)
    moe = MixtureOfSpecialists(pager=pager, tokenizer=tok,
                               domain_names=["comms", "media", "info", "control"])
    feats = torch.randn(20, 40)
    r0 = moe.transcribe(feats, 0)  # miss
    r0b = moe.transcribe(feats, 0)  # hit
    r1 = moe.transcribe(feats, 1)  # miss (evicts 0)
    assert r0["page_latency_ms"] > 0 and r0b["page_latency_ms"] == 0.0
    rep = pager.report()
    assert rep["experts"] == 4
    assert rep["flash_footprint_mb"] > rep["resident_footprint_mb"]  # capacity > resident
    print(f"moe OK: {rep['flash_footprint_mb']:.2f} MB flash / "
          f"{rep['resident_footprint_mb']:.2f} MB resident, hit_rate {rep['hit_rate']}")


if __name__ == "__main__":
    test_mamba_encoder_trains_and_streams()
    test_command_model_hypernet_and_router()
    test_flash_paged_moe()
    print("V2 OK")
