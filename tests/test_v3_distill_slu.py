"""Smoke tests for the v3 pieces: distillation (CTC-KD + pseudo-labels) and
the joint intent+slot SLU model.

Run: python tests/test_v3_distill_slu.py
"""
from __future__ import annotations

import os
import tempfile

import torch
from torch.utils.data import DataLoader

from edge_asr.data import (ManifestDataset, collate_asr, load_tokenizer,
                           make_synthetic_asr_manifest)
from edge_asr.distill import Teacher, ctc_kd_loss
from edge_asr.features import LogMelFrontend, OnlineCMVN
from edge_asr.models import SSMEncoderConfig, Transducer, TransducerConfig
from edge_asr.slu import SLUConfig, SLUModel, SLUParser
from edge_asr.slu.data import build_vocab, make_dataset
from edge_asr.training import configure_optimizer


def _tiny(vocab, d=64, layers=2):
    enc = SSMEncoderConfig(input_dim=40, d_model=d, n_layers=layers, d_state=8, chunk_frames=16)
    return Transducer(TransducerConfig(vocab_size=vocab, encoder=enc, decoder_dim=48, joiner_dim=48))


def test_ctc_kd_loss_and_distill_step():
    torch.manual_seed(0)
    tmp = tempfile.mkdtemp()
    man = os.path.join(tmp, "a.jsonl")
    make_synthetic_asr_manifest(man, n=24, seed=0)
    tok = load_tokenizer("char")
    fe, cmvn = LogMelFrontend(n_mels=40), OnlineCMVN(n_mels=40)
    ds = ManifestDataset(man, fe, cmvn, tok, task="asr")
    dl = DataLoader(ds, batch_size=6, shuffle=True, collate_fn=collate_asr)

    teacher_model = _tiny(tok.vocab_size, d=96, layers=3)
    teacher = Teacher(teacher_model, tok)
    student = _tiny(tok.vocab_size, d=48, layers=2)
    opt = configure_optimizer(student, lr=2e-3)

    f, fl, t, tl = next(iter(dl))
    soft = teacher.soft_targets(f, fl, t, tl)
    out = student(f, fl, t, tl, return_features=True)
    # teacher and student share frame rate -> KD shapes align
    assert out["ctc_logits"].shape == soft["ctc_logits"].shape
    kd = ctc_kd_loss(out["ctc_logits"], soft["ctc_logits"], out["enc_lens"])
    assert torch.isfinite(kd) and kd.item() >= 0

    losses = []
    it = iter(dl)
    for _ in range(40):
        try:
            f, fl, t, tl = next(it)
        except StopIteration:
            it = iter(dl); f, fl, t, tl = next(it)
        soft = teacher.soft_targets(f, fl, t, tl)
        out = student(f, fl, t, tl, return_features=True)
        loss = out["loss"] + ctc_kd_loss(out["ctc_logits"], soft["ctc_logits"], out["enc_lens"])
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0], f"distill did not reduce loss: {losses[0]:.1f}->{losses[-1]:.1f}"
    print(f"distill OK: kd={kd.item():.3f}, loss {losses[0]:.1f} -> {losses[-1]:.2f}")


def test_pseudo_label_transcribe():
    torch.manual_seed(0)
    tmp = tempfile.mkdtemp()
    man = os.path.join(tmp, "a.jsonl")
    make_synthetic_asr_manifest(man, n=8, seed=0)
    tok = load_tokenizer("char")
    fe, cmvn = LogMelFrontend(n_mels=40), OnlineCMVN(n_mels=40)
    ds = ManifestDataset(man, fe, cmvn, tok, task="asr")
    teacher = Teacher(_tiny(tok.vocab_size), tok)
    hyp = teacher.transcribe(ds[0][0])  # untrained teacher -> some string (maybe empty)
    assert isinstance(hyp, str)
    print(f"pseudo-label OK: teacher hyp type ok ('{hyp}')")


def test_slu_trains_and_parses():
    torch.manual_seed(0)
    word2id, intents, tags = build_vocab(seed=0)
    train = make_dataset(300, word2id, seed=0)
    cfg = SLUConfig(word_vocab=len(word2id), num_intents=len(intents), num_slot_tags=len(tags))
    model = SLUModel(cfg)
    opt = configure_optimizer(model, lr=2e-3)
    import random
    rng = random.Random(0)
    import torch.nn.functional as F
    for _ in range(250):
        batch = [rng.choice(train) for _ in range(32)]
        L = max(len(b["token_ids"]) for b in batch)
        ids = torch.zeros(32, L, dtype=torch.long)
        tag_t = torch.full((32, L), -100, dtype=torch.long)
        lens = torch.zeros(32, dtype=torch.long)
        it = torch.zeros(32, dtype=torch.long)
        for i, b in enumerate(batch):
            n = len(b["token_ids"])
            ids[i, :n] = torch.tensor(b["token_ids"]); tag_t[i, :n] = torch.tensor(b["tags"])
            lens[i] = n; it[i] = b["intent"]
        out = model(ids, lens)
        loss = F.cross_entropy(out["intent"], it) + F.cross_entropy(
            out["slots"].reshape(-1, cfg.num_slot_tags), tag_t.reshape(-1), ignore_index=-100)
        opt.zero_grad(); loss.backward(); opt.step()

    parser = SLUParser(intents, tags)
    model.eval()
    words = "set a timer for five minutes".split()
    ids = [word2id.get(w, 0) for w in words]
    res = parser.parse(model, ids, words)
    assert res.intent == "timer"
    assert res.slots.get("number") == "five" and res.slots.get("unit") == "minutes"
    print(f"slu OK: {res.as_dict()}  ({model.num_params()/1e3:.1f} K params)")


if __name__ == "__main__":
    test_ctc_kd_loss_and_distill_step()
    test_pseudo_label_transcribe()
    test_slu_trains_and_parses()
    print("V3 OK")
