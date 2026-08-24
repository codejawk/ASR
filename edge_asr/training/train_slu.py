"""Train the joint intent+slot SLU head for Model 2, then demo parsing.

    python -m edge_asr.training.train_slu --steps 400 --out runs/slu

Reports intent accuracy and slot-tag accuracy, then parses example commands
into structured actions.
"""
from __future__ import annotations

import argparse
import os

import torch
import torch.nn.functional as F

from ..slu import SLUConfig, SLUModel, SLUParser
from ..slu.data import BIO_TAGS, INTENTS, build_vocab, make_dataset
from .utils import configure_optimizer, set_seed


def _pad(batch, pad_id=0):
    L = max(len(b["token_ids"]) for b in batch)
    ids = torch.zeros(len(batch), L, dtype=torch.long)
    tags = torch.full((len(batch), L), -100, dtype=torch.long)  # ignore_index
    lens = torch.zeros(len(batch), dtype=torch.long)
    intents = torch.zeros(len(batch), dtype=torch.long)
    for i, b in enumerate(batch):
        n = len(b["token_ids"])
        ids[i, :n] = torch.tensor(b["token_ids"])
        tags[i, :n] = torch.tensor(b["tags"])
        lens[i] = n
        intents[i] = b["intent"]
    return ids, tags, lens, intents


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--out", default="runs/slu")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    set_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    word2id, intents, tags = build_vocab(seed=args.seed)
    train = make_dataset(400, word2id, seed=args.seed)

    cfg = SLUConfig(word_vocab=len(word2id), num_intents=len(intents), num_slot_tags=len(tags))
    model = SLUModel(cfg)
    print(f"[params] SLU {model.num_params()/1e3:.1f} K  (~{model.num_params()/1e6:.3f} MB int8)")
    opt = configure_optimizer(model, lr=args.lr)

    import random
    rng = random.Random(args.seed)
    model.train()
    for step in range(args.steps):
        batch = [rng.choice(train) for _ in range(args.batch_size)]
        ids, tag_t, lens, intent_t = _pad(batch)
        out = model(ids, lens)
        loss_i = F.cross_entropy(out["intent"], intent_t)
        loss_s = F.cross_entropy(out["slots"].reshape(-1, cfg.num_slot_tags),
                                 tag_t.reshape(-1), ignore_index=-100)
        loss = loss_i + loss_s
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 100 == 0 or step == args.steps - 1:
            i_acc = (out["intent"].argmax(-1) == intent_t).float().mean().item()
            m = tag_t != -100
            s_acc = ((out["slots"].argmax(-1) == tag_t) & m).float().sum().item() / m.sum().clamp_min(1).item()
            print(f"step {step:4d} loss {loss.item():.3f} | intent_acc {i_acc:.3f} slot_acc {s_acc:.3f}")

    torch.save({"model": model.state_dict(), "config": cfg.__dict__,
                "word2id": word2id, "intents": intents, "tags": tags},
               os.path.join(args.out, "slu.pt"))
    print(f"[save] {os.path.join(args.out, 'slu.pt')}")

    # ---- demo parse ----
    parser = SLUParser(intents, tags)
    model.eval()
    examples = ["set a timer for five minutes", "call mom", "weather in tokyo",
                "send a message to boss", "play some music"]
    print("\n[demo] on-device SLU parsing:")
    for text in examples:
        words = text.split()
        ids = [word2id.get(w, 0) for w in words]
        res = parser.parse(model, ids, words)
        print(f"  '{text}'  ->  {res.as_dict()}")


if __name__ == "__main__":
    main()
