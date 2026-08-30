"""Build ASR train/eval manifests from SLURP (real spoken-assistant commands).

SLURP: ~72k real recordings of home-assistant commands, annotated with
scenario/action/entities. Real human voices + real command phrases -> a model
trained on it recognizes real speech (unlike the TTS demo).

    # 1) get SLURP metadata + audio (audio download is several GB)
    git clone https://github.com/pswietojanski/slurp
    cd slurp/scripts && ./download_audio.sh && cd ../..
    # 2) build manifests (subset with --max for a quick run)
    python scripts/prep_slurp.py --slurp ./slurp --split real --max 8000 --out data/slurp

LICENSE: SLURP audio is CC BY-NC 4.0 (NON-COMMERCIAL). Fine for research/demo;
for a product obtain a commercial license (info@emotech.co).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from edge_asr.data.text_norm import normalize


def find_meta(slurp_root):
    for cand in ["dataset/slurp", "dataset", "."]:
        d = os.path.join(slurp_root, cand)
        if os.path.exists(os.path.join(d, "train.jsonl")):
            return d
    hits = glob.glob(os.path.join(slurp_root, "**", "train.jsonl"), recursive=True)
    if hits:
        return os.path.dirname(hits[0])
    sys.exit(f"Could not find train.jsonl under {slurp_root}. Did you clone SLURP?")


def resolve_audio(slurp_root, fname, split):
    subdirs = {"real": ["slurp_real"], "synth": ["slurp_synth"], "all": ["slurp_real", "slurp_synth"]}[split]
    for base in [os.path.join(slurp_root, "audio"), slurp_root]:
        for sd in subdirs:
            p = os.path.join(base, sd, fname)
            if os.path.exists(p):
                return os.path.abspath(p)
    return None


def build(meta_dir, slurp_root, jsonl, split, cap, rng):
    rows, missing = [], 0
    for line in open(os.path.join(meta_dir, jsonl)):
        e = json.loads(line)
        text = normalize(e.get("sentence", ""))
        if not text:
            continue
        for rec in e.get("recordings", []):
            fname = rec.get("file")
            if not fname:
                continue
            ap = resolve_audio(slurp_root, fname, split)
            if ap is None:
                missing += 1
                continue
            rows.append({"audio": ap, "text": text})
    rng.shuffle(rows)
    if cap:
        rows = rows[:cap]
    return rows, missing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slurp", required=True, help="path to the cloned slurp/ repo")
    ap.add_argument("--split", choices=["real", "synth", "all"], default="real")
    ap.add_argument("--max", type=int, default=8000, help="cap training rows (subset)")
    ap.add_argument("--out", default="data/slurp")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    meta = find_meta(args.slurp)
    rng = random.Random(args.seed)

    train, m1 = build(meta, args.slurp, "train.jsonl", args.split, args.max, rng)
    dev_name = "devel.jsonl" if os.path.exists(os.path.join(meta, "devel.jsonl")) else "test.jsonl"
    ev, m2 = build(meta, args.slurp, dev_name, args.split, max(300, args.max // 20), rng)

    with open(os.path.join(args.out, "train.jsonl"), "w") as f:
        f.write("\n".join(json.dumps(r) for r in train))
    with open(os.path.join(args.out, "eval.jsonl"), "w") as f:
        f.write("\n".join(json.dumps(r) for r in ev))
    with open(os.path.join(args.out, "train_text.txt"), "w") as f:
        f.write("\n".join(r["text"] for r in train))
    print(f"[slurp] split={args.split}  train {len(train)}  eval {len(ev)}  "
          f"(missing audio: {m1 + m2})")
    if m1 + m2 > 0:
        print("  [warn] some audio not found — run slurp/scripts/download_audio.sh first")


if __name__ == "__main__":
    main()
