"""Tokenizers. BPE (SentencePiece) for real training; a char tokenizer
for smoke tests and tiny command sets.

Convention: id 0 is always the transducer/CTC **blank**. SentencePiece is
trained with the real text; the char tokenizer builds its vocab from a
fixed alphabet. `tokens.txt` (one token per line, id == line number) is the
artifact shipped next to the model.
"""
from __future__ import annotations

import os
from typing import List, Optional

BLANK = "<blk>"


class CharTokenizer:
    def __init__(self, alphabet: Optional[str] = None):
        if alphabet is None:
            alphabet = " abcdefghijklmnopqrstuvwxyz'"
        # id 0 = blank, then chars
        self.tokens = [BLANK] + list(alphabet)
        self.tok2id = {t: i for i, t in enumerate(self.tokens)}
        self.blank_id = 0

    @property
    def vocab_size(self) -> int:
        return len(self.tokens)

    def encode(self, text: str) -> List[int]:
        text = text.lower()
        return [self.tok2id[c] for c in text if c in self.tok2id]

    def decode(self, ids: List[int]) -> str:
        return "".join(self.tokens[i] for i in ids if i != self.blank_id)

    def save(self, path: str):
        with open(path, "w") as f:
            for t in self.tokens:
                f.write(t + "\n")


class SentencePieceTokenizer:
    def __init__(self, model_path: str):
        import sentencepiece as spm

        self.sp = spm.SentencePieceProcessor(model_file=model_path)
        self.blank_id = 0  # we reserve 0 for blank; sp ids are shifted by +1

    @property
    def vocab_size(self) -> int:
        return self.sp.get_piece_size() + 1  # +1 for blank at id 0

    def encode(self, text: str) -> List[int]:
        return [i + 1 for i in self.sp.encode(text.lower(), out_type=int)]

    def decode(self, ids: List[int]) -> str:
        pieces = [i - 1 for i in ids if i != self.blank_id]
        return self.sp.decode(pieces)

    @staticmethod
    def train(text_file: str, model_prefix: str, vocab_size: int = 500):
        import sentencepiece as spm

        spm.SentencePieceTrainer.train(
            input=text_file,
            model_prefix=model_prefix,
            vocab_size=vocab_size,
            model_type="bpe",
            character_coverage=1.0,
            bos_id=-1,
            eos_id=-1,
            unk_id=0,
            pad_id=-1,
        )
        return model_prefix + ".model"


def load_tokenizer(spec: str):
    """spec = "char" or a path to a .model SentencePiece file."""
    if spec == "char":
        return CharTokenizer()
    if os.path.exists(spec):
        return SentencePieceTokenizer(spec)
    raise FileNotFoundError(f"tokenizer spec not understood: {spec}")
