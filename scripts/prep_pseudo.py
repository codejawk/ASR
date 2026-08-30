"""Rebuild NORMALIZED student-training files from cached Whisper labels.

Run from the repo root after data/pseudo.jsonl, data/train_gt.jsonl and
data/eval.jsonl already exist (the Colab notebook creates them). Writes
data/train_pseudo.jsonl (student targets) and data/train_text.txt (tokenizer
corpus), both normalized so the tokenizer never emits UNK.

    python scripts/prep_pseudo.py
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from edge_asr.data.text_norm import normalize
from edge_asr.eval import wer


def main():
    pseudo = {
        os.path.abspath(json.loads(l)["audio"]): normalize(json.loads(l)["text"])
        for l in open("data/pseudo.jsonl")
    }
    tr = [json.loads(l) for l in open("data/train_gt.jsonl")]
    n = 0
    with open("data/train_pseudo.jsonl", "w") as f, open("data/train_text.txt", "w") as t:
        for r in tr:
            a = r["audio"]
            if pseudo.get(a):
                f.write(json.dumps({"audio": a, "text": pseudo[a], "duration": r["duration"]}) + "\n")
                t.write(pseudo[a] + "\n")
                n += 1
    ev = [json.loads(l) for l in open("data/eval.jsonl")]
    refs = [normalize(r["text"]) for r in ev]
    hyps = [pseudo.get(r["audio"], "") for r in ev]
    print("teacher WER on eval:", round(wer(refs, hyps), 3))
    print(f"student training lines: {n}  -> data/train_pseudo.jsonl, data/train_text.txt")


if __name__ == "__main__":
    main()
