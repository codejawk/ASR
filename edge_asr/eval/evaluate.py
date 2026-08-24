"""Evaluate a trained Model 1 checkpoint over a manifest.

    python -m edge_asr.eval.evaluate --ckpt runs/model1/model1.pt \
        --manifest data/eval_wrist.jsonl --streaming

Reports WER/CER for both the full-context and streaming decode paths so
you can see the streaming penalty directly.
"""
from __future__ import annotations

import argparse
import json

import torch

from ..data import ManifestDataset, load_tokenizer
from ..decode import greedy_search, streaming_greedy_search
from ..features import LogMelFrontend, OnlineCMVN
from ..training.utils import build_model1
from .metrics import cer, wer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--streaming", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    blob = torch.load(args.ckpt, weights_only=False)
    cfg = blob["config"]
    tok = load_tokenizer(blob.get("tokenizer", "char"))
    model = build_model1(cfg, blob["vocab_size"])
    model.load_state_dict(blob["model"])
    model.eval()

    frontend = LogMelFrontend(n_mels=cfg["encoder"]["input_dim"])
    cmvn = OnlineCMVN(n_mels=cfg["encoder"]["input_dim"])
    ds = ManifestDataset(args.manifest, frontend, cmvn, tok, task="asr")

    refs, hyps = [], []
    n = len(ds) if args.limit == 0 else min(args.limit, len(ds))
    for i in range(n):
        feats, toks = ds[i]
        ids = streaming_greedy_search(model, feats) if args.streaming else greedy_search(model, feats)
        hyps.append(tok.decode(ids))
        refs.append(tok.decode(toks.tolist()))

    print(json.dumps({
        "n": n,
        "mode": "streaming" if args.streaming else "full",
        "wer": round(wer(refs, hyps), 4),
        "cer": round(cer(refs, hyps), 4),
    }, indent=2))


if __name__ == "__main__":
    main()
