"""Train Model 1 (streaming RNN-T + aux CTC).

Runnable end-to-end on synthetic data:

    python -m edge_asr.training.train_model1 --config configs/model1_general.yaml \
        --manifest data/synth_asr.jsonl --steps 300 --out runs/model1

For real training, point --manifest at a Loquacious/pseudo-labelled
manifest (see docs/DATA_LICENSING.md) and scale --config. Distillation
(hard teacher targets) enters here simply as the manifest text being the
teacher's hypotheses — see docs/ARCHITECTURE.md 3.2.
"""
from __future__ import annotations

import argparse
import os

import torch
from torch.utils.data import DataLoader

from ..data import (
    ManifestDataset,
    NoiseInjection,
    SpecAugment,
    collate_asr,
    load_tokenizer,
    make_synthetic_asr_manifest,
)
from ..decode import greedy_search
from ..eval import wer
from ..features import LogMelFrontend, OnlineCMVN
from .utils import build_model1, configure_optimizer, load_config, pick_device, set_seed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--manifest", default=None, help="JSONL; if absent, synthesize")
    ap.add_argument("--tokenizer", default="char")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--noise-dir", default=None)
    ap.add_argument("--device", default="cpu", help="cpu | cuda | mps | auto")
    ap.add_argument("--out", default="runs/model1")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    set_seed(args.seed)
    device = pick_device(args.device)
    os.makedirs(args.out, exist_ok=True)
    cfg = load_config(args.config)

    if args.manifest is None:
        args.manifest = os.path.join(args.out, "synth_asr.jsonl")
        make_synthetic_asr_manifest(args.manifest, n=64, seed=args.seed)
        print(f"[data] synthesized {args.manifest}")

    tok = load_tokenizer(args.tokenizer)
    frontend = LogMelFrontend(n_mels=cfg["encoder"]["input_dim"])
    cmvn = OnlineCMVN(n_mels=cfg["encoder"]["input_dim"])

    ds = ManifestDataset(
        args.manifest, frontend, cmvn, tok, task="asr",
        wav_augment=NoiseInjection(args.noise_dir, prob=0.5),
        spec_augment=SpecAugment(),
    )
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_asr)

    model = build_model1(cfg, tok.vocab_size).to(device)
    print("[params]", model.num_params(), "| encoder:", cfg["encoder"].get("encoder_type", "conformer"),
          "| device:", device)
    opt = configure_optimizer(model, lr=args.lr, weight_decay=1e-2)

    model.train()
    step = 0
    it = iter(dl)
    while step < args.steps:
        try:
            feats, flen, toks, tlen = next(it)
        except StopIteration:
            it = iter(dl)
            continue
        feats, flen, toks, tlen = (feats.to(device), flen.to(device),
                                   toks.to(device), tlen.to(device))
        out = model(feats, flen, toks, tlen)
        opt.zero_grad()
        out["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        step += 1
        if step % 25 == 0 or step == 1:
            print(f"step {step:4d}  loss {out['loss'].item():.3f}  "
                  f"rnnt {out['rnnt'].item():.3f}  ctc {out['ctc'].item():.3f}")

    ckpt = os.path.join(args.out, "model1.pt")
    torch.save({"model": model.state_dict(), "config": cfg,
                "tokenizer": args.tokenizer, "vocab_size": tok.vocab_size}, ckpt)
    print(f"[save] {ckpt}")

    # quick sanity decode
    feats0, toks0 = ds[0]
    hyp = tok.decode(greedy_search(model, feats0.to(device)))
    ref = tok.decode(toks0.tolist())
    print(f"[decode] ref='{ref}'  hyp='{hyp}'  wer={wer([ref],[hyp]):.2f}")


if __name__ == "__main__":
    main()
