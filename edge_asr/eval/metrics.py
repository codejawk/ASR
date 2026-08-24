"""Evaluation metrics: WER / CER (Levenshtein) and FA/hour.

Ship-gate reminder (docs/ARCHITECTURE.md, Part 3.6): report WER *and*
semantic task-success, latency percentiles (p50/p95 — not means), and
battery per active minute. This module gives you the text-side numbers.
"""
from __future__ import annotations

from typing import List, Sequence


def edit_distance(ref: Sequence, hyp: Sequence) -> int:
    n, m = len(ref), len(hyp)
    if n == 0:
        return m
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        cur = [i] + [0] * m
        for j in range(1, m + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[m]


def wer(refs: List[str], hyps: List[str]) -> float:
    total_err, total_words = 0, 0
    for r, h in zip(refs, hyps):
        rw, hw = r.split(), h.split()
        total_err += edit_distance(rw, hw)
        total_words += len(rw)
    return total_err / max(total_words, 1)


def cer(refs: List[str], hyps: List[str]) -> float:
    total_err, total_chars = 0, 0
    for r, h in zip(refs, hyps):
        total_err += edit_distance(list(r), list(h))
        total_chars += len(r)
    return total_err / max(total_chars, 1)


def false_accepts_per_hour(num_false_accepts: int, hours: float) -> float:
    return num_false_accepts / max(hours, 1e-9)
