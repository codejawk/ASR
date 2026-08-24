"""Synthetic SLU data — a small command grammar with intents and slot spans.

Generates (words, intent, BIO-tags) triples, and builds the word / intent /
tag vocabularies. No downloads. Realistic enough to train the joint
intent+slot model to convergence and demo structured parsing.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

# each template: (intent, list of tokens where a token can be a literal or
# a ("<slot_type>", [candidate values]) placeholder)
NUMBERS = ["one", "two", "three", "five", "ten", "fifteen", "thirty"]
UNITS = ["minutes", "seconds", "hours"]
CONTACTS = ["mom", "dad", "alex", "sam", "boss"]
CITIES = ["london", "paris", "tokyo", "delhi"]

TEMPLATES = [
    ("timer", ["set", "a", "timer", "for", ("number", NUMBERS), ("unit", UNITS)]),
    ("alarm", ["set", "an", "alarm", "for", ("number", NUMBERS), ("unit", UNITS)]),
    ("call", ["call", ("contact", CONTACTS)]),
    ("message", ["send", "a", "message", "to", ("contact", CONTACTS)]),
    ("weather", ["weather", "in", ("city", CITIES)]),
    ("music", ["play", "some", "music"]),
]

INTENTS = [t[0] for t in TEMPLATES]
SLOT_TYPES = ["number", "unit", "contact", "city"]


def _bio_tags():
    tags = ["O"]
    for s in SLOT_TYPES:
        tags += [f"B-{s}", f"I-{s}"]
    return tags


BIO_TAGS = _bio_tags()


def sample(rng: random.Random) -> Tuple[List[str], str, List[str]]:
    intent, template = rng.choice(TEMPLATES)
    words, tags = [], []
    for tok in template:
        if isinstance(tok, tuple):
            stype, cands = tok
            val = rng.choice(cands).split()
            for j, w in enumerate(val):
                words.append(w)
                tags.append(("B-" if j == 0 else "I-") + stype)
        else:
            words.append(tok)
            tags.append("O")
    return words, intent, tags


def build_vocab(n_samples: int = 400, seed: int = 0):
    rng = random.Random(seed)
    words = set()
    for _ in range(n_samples):
        w, _, _ = sample(rng)
        words.update(w)
    word_list = ["<pad>"] + sorted(words)
    word2id = {w: i for i, w in enumerate(word_list)}
    return word2id, INTENTS, BIO_TAGS


def make_dataset(n: int, word2id: Dict[str, int], seed: int = 0):
    rng = random.Random(seed)
    intent2id = {v: i for i, v in enumerate(INTENTS)}
    tag2id = {v: i for i, v in enumerate(BIO_TAGS)}
    data = []
    for _ in range(n):
        words, intent, tags = sample(rng)
        ids = [word2id.get(w, 0) for w in words]
        data.append({
            "words": words,
            "token_ids": ids,
            "intent": intent2id[intent],
            "tags": [tag2id[t] for t in tags],
        })
    return data
