"""Text normalization for ASR targets.

Teacher models (Whisper) emit punctuation and casing; ASR training targets
should be a small, consistent character set or the tokenizer produces UNK
tokens for every comma/period — which makes the targets unlearnable. This
lowercases, drops punctuation (keeping apostrophes and digits), and collapses
whitespace. Use the SAME normalization for (a) the tokenizer training text,
(b) the student's training targets, and (c) the eval references.
"""
from __future__ import annotations

import re

_KEEP = re.compile(r"[^a-z0-9' ]+")
_WS = re.compile(r"\s+")


def normalize(text: str) -> str:
    text = text.lower()
    text = _KEEP.sub(" ", text)
    text = _WS.sub(" ", text).strip()
    return text
