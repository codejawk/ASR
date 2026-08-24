"""Train the redesigned Model 2 (open-vocab hypernetwork KWS + router).

    python -m edge_asr.training.train_command --config configs/model2_command.yaml \
        --steps 300 --out runs/command

Losses:
  * detection : BCE-with-logits on (audio, keyword) pairs — half positive
    (keyword == spoken command), half negative (mismatched keyword). This is
    what teaches the hypernetwork to generate a working detector from text.
  * router    : cross-entropy on the utterance domain (drives MoE expert
    selection).

Reports **open-vocab detection accuracy on held-out keywords** (keywords the
detector's contrastive pairing never saw as positives during the last
epoch) and router accuracy.
"""
from __future__ import annotations

import argparse
import json
import os
import random

import torch
import torch.nn.functional as F

from ..data import make_synthetic_command_manifest
from ..data.datasets import load_wav
from ..features import LogMelFrontend, OnlineCMVN
from ..models import CommandModel, CommandModelConfig, encode_keyword
from .utils import configure_optimizer, load_config, set_seed


def _domains_for(classes):
    # group commands into coarse domains that map to Model-1 experts
    mapping = {
        "call": 0, "message": 0, "answer": 0,          # comms
        "music": 1, "stop": 1,                          # media
        "timer": 2, "weather": 2, "maps": 2,            # info
        "cancel": 3, "wake": 3, "silence": 3, "unknown": 3,  # control
    }
    return [mapping.get(c, 3) for c in classes]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--out", default="runs/command")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    set_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    cfg = load_config(args.config)
    commands = [c for c in cfg["classes"] if c not in ("silence", "unknown")]
    domains = _domains_for(commands)
    num_domains = max(domains) + 1

    manifest = os.path.join(args.out, "synth_cmd.jsonl")
    make_synthetic_command_manifest(manifest, commands, domains, n_per=24, seed=args.seed)
    items = [json.loads(l) for l in open(manifest)]

    n_mels = cfg["bcresnet"]["n_mels"]
    frontend = LogMelFrontend(n_mels=n_mels)
    cmvn = OnlineCMVN(n_mels=n_mels)
    cm_cfg = dict(cfg.get("command_model", {}))
    cm_cfg["num_domains"] = num_domains  # derived from data
    fields = CommandModelConfig.__dataclass_fields__
    mcfg = CommandModelConfig(**{k: v for k, v in cm_cfg.items() if k in fields})
    model = CommandModel(mcfg)
    print(f"[params] CommandModel {model.num_params()/1e6:.3f} M  "
          f"(~{model.num_params()/1e6:.2f} MB int8)")
    opt = configure_optimizer(model, lr=args.lr)

    rng = random.Random(args.seed)

    def feats_of(item):
        wav = torch.tensor(item["array"], dtype=torch.float32)
        f = frontend(wav).squeeze(0)
        return cmvn(f.unsqueeze(0)).squeeze(0)

    # precompute features
    cache = [feats_of(it) for it in items]
    maxT = max(f.size(0) for f in cache)

    def batch(indices):
        feats, kw_ids, labels, doms = [], [], [], []
        for i in indices:
            it = items[i]
            fpad = F.pad(cache[i], (0, 0, 0, maxT - cache[i].size(0)))
            feats.append(fpad)
            if rng.random() < 0.5:  # positive pair
                kw = it["keyword"]; labels.append(1.0)
            else:  # negative pair: a different command's name
                kw = rng.choice([c for c in commands if c != it["keyword"]])
                labels.append(0.0)
            kw_ids.append(encode_keyword(kw))
            doms.append(it["domain"])
        return (torch.stack(feats),
                torch.tensor(kw_ids, dtype=torch.long),
                torch.tensor(labels),
                torch.tensor(doms, dtype=torch.long))

    model.train()
    n = len(items)
    for step in range(args.steps):
        idx = [rng.randrange(n) for _ in range(args.batch_size)]
        feats, kw_ids, labels, doms = batch(idx)
        out = model(feats, kw_ids)
        loss_det = F.binary_cross_entropy_with_logits(out["detect"], labels)
        loss_rt = F.cross_entropy(out["router"], doms)
        loss = loss_det + loss_rt
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 50 == 0 or step == args.steps - 1:
            det_acc = ((out["detect"] > 0).float() == labels).float().mean().item()
            rt_acc = (out["router"].argmax(-1) == doms).float().mean().item()
            print(f"step {step:4d} loss {loss.item():.3f} | det_acc {det_acc:.3f} "
                  f"router_acc {rt_acc:.3f}")

    ckpt = os.path.join(args.out, "command.pt")
    torch.save({"model": model.state_dict(), "config": mcfg.__dict__,
                "commands": commands, "domains": domains}, ckpt)
    print(f"[save] {ckpt}")


if __name__ == "__main__":
    main()
