#!/usr/bin/env bash
# Data preparation skeleton. Fill in the corpus download once you have
# cleared licensing (see docs/DATA_LICENSING.md). Do NOT add GigaSpeech.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data

echo "[data] This is a skeleton. Recommended commercially-clean English set:"
echo "       Loquacious (25k h). Then Common Voice (CC0) + MLS (CC-BY)."
echo
echo "Example (HuggingFace datasets), AFTER legal sign-off:"
cat <<'PY'
# python -c "
# from datasets import load_dataset
# ds = load_dataset('speechbrain/loquacious', split='train', streaming=True)
# # -> write manifest lines: {'audio': path, 'text': ..., 'duration': ...}
# "
PY
echo
echo "Then build a SentencePiece BPE-500 model from the transcripts:"
cat <<'PY'
# python -c "
# from edge_asr.data.tokenizer import SentencePieceTokenizer
# SentencePieceTokenizer.train('data/train_text.txt','data/bpe500',vocab_size=500)
# "
PY
