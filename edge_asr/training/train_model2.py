"""Train Model 2 (BC-ResNet closed-set command classifier).

    python -m edge_asr.training.train_model2 --config configs/model2_command.yaml \
        --steps 300 --out runs/model2

Uses cross-entropy with an explicit "unknown" and "silence" class. The
production gate is FA/hour at fixed FRR (docs/ARCHITECTURE.md 4), tuned on
real ambient audio — not accuracy on Speech Commands.
"""
from __future__ import annotations

import argparse
import os

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..data import ManifestDataset, collate_kws, make_synthetic_kws_manifest
from ..features import LogMelFrontend, OnlineCMVN
from .utils import build_model2, load_config, set_seed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", default="runs/model2")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    set_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    cfg = load_config(args.config)
    classes = cfg["classes"]

    if args.manifest is None:
        args.manifest = os.path.join(args.out, "synth_kws.jsonl")
        make_synthetic_kws_manifest(args.manifest, classes, n_per=20, seed=args.seed)
        print(f"[data] synthesized {args.manifest}")

    frontend = LogMelFrontend(n_mels=cfg["bcresnet"]["n_mels"])
    cmvn = OnlineCMVN(n_mels=cfg["bcresnet"]["n_mels"])
    ds = ManifestDataset(args.manifest, frontend, cmvn, task="kws")
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_kws)

    model = build_model2(cfg)
    print("[params] bcresnet", model.num_params())
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)

    model.train()
    step, it = 0, iter(dl)
    while step < args.steps:
        try:
            feats, labels = next(it)
        except StopIteration:
            it = iter(dl)
            continue
        logits = model(feats)
        loss = F.cross_entropy(logits, labels)
        opt.zero_grad()
        loss.backward()
        opt.step()
        step += 1
        if step % 25 == 0 or step == 1:
            acc = (logits.argmax(-1) == labels).float().mean().item()
            print(f"step {step:4d}  loss {loss.item():.3f}  acc {acc:.3f}")

    ckpt = os.path.join(args.out, "model2.pt")
    torch.save({"model": model.state_dict(), "config": cfg, "classes": classes}, ckpt)
    print(f"[save] {ckpt}")


if __name__ == "__main__":
    main()
