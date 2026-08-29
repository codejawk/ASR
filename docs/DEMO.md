# Demo & roadmap — from your Mac to a real distilled model

A staged path. Phases 0–1 run on your **Mac (CPU)** with a *real* Whisper
teacher; Phases 2–4 scale up on a **GPU**. Nothing here is throwaway — the
Whisper you run in Phase 0 is the same teacher used for real distillation.

| Phase | Where | What you get |
|-------|-------|--------------|
| 0. Feel it work | Mac CPU | real Whisper transcribing your audio |
| 1. Real mini-distillation | Mac CPU | Whisper labels clips → small student learns from them |
| 2. Scale up | Colab GPU | real dataset + Whisper-large teacher, full training |
| 3. Compress | GPU/box | int4 QAT + pruning → ~10 MB |
| 4. Deploy | SW6100 | QNN on-device |

---

## Phase 0 — hear it work (Mac, ~5 min)

```bash
source .venv/bin/activate
pip install faster-whisper            # fast on Mac CPU (int8)

# transcribe any audio file (convert to 16 kHz mono first if needed):
ffmpeg -i myclip.m4a -ac 1 -ar 16000 myclip.wav      # if you have ffmpeg
python scripts/transcribe_whisper.py myclip.wav --model base
# -> [transcript] your words here
```
Record a few seconds on your Mac (Voice Memos / QuickTime), export, convert,
and watch a real model recognize your speech. **This is the teacher.**

---

## Phase 1 — real mini-distillation (Mac, ~15–30 min)

Prove the *real* pipeline (not synthetic) end-to-end on CPU:

```bash
# 1) collect ~20-100 short 16 kHz mono WAVs into data/clips/
#    (your voice, podcasts you can use, LibriSpeech samples, etc.)

# 2) Whisper labels them -> a training manifest
python scripts/pseudo_label_whisper.py --audio-dir data/clips \
    --out data/pseudo.jsonl --model small

# 3) train the small STUDENT on real audio + Whisper labels
python -m edge_asr.training.train_model1 \
    --config configs/model1_general.yaml --tokenizer char \
    --manifest data/pseudo.jsonl --steps 2000 --device cpu --out runs/student

# 4) see the student transcribe (its own decode)
python -m edge_asr.eval.evaluate --ckpt runs/student/model1.pt \
    --manifest data/pseudo.jsonl --limit 10
```
With only tens of clips the student will be rough — the point is the **real
pipeline runs**: real teacher → real labels → small streaming student. Accuracy
comes with data + a GPU (Phase 2).

> Tip: for a cleaner text target use a **BPE tokenizer** instead of `char`.
> Train one on your transcripts:
> `python -c "from edge_asr.data.tokenizer import SentencePieceTokenizer as S; S.train('data/train_text.txt','data/bpe500',500)"`
> then pass `--tokenizer data/bpe500.model`.

---

## Phase 2 — scale up on Colab GPU

See **docs/RUNNING.md** (§B). In short: GPU runtime, `pip install` deps
*except torch*, use `whisper large-v3` as the teacher on a real dataset
(Loquacious / Common Voice — clear licensing in **docs/DATA_LICENSING.md**),
train the student with `--device auto`.

```python
!python scripts/pseudo_label_whisper.py --audio-dir data/train_audio \
    --out data/pseudo.jsonl --model large-v3 --device cuda --compute-type float16
!PYTHONPATH=. python -m edge_asr.training.train_model1 \
    --config configs/model1_general.yaml --tokenizer data/bpe500.model \
    --manifest data/pseudo.jsonl --steps 40000 --device auto --out runs/student
```

---

## Phase 3 — compress to ~10 MB

```bash
python -m edge_asr.tools.count_params configs/model1_general.yaml 500   # budget
python scripts/export_pipeline.py --ckpt runs/student/model1.pt \
    --out runs/student/onnx --mode mixed                                 # int4-mixed
```
Ship path is int4 **QAT** (planned) — PTQ here reports the size; QAT recovers
the accuracy. Add structured pruning + re-distill for a further 1.3–2×.

---

## Phase 4 — deploy on SW6100

3-graph static ONNX → QNN context binary for the Wear-Elite Hexagon (Conformer
path; Mamba stays research). See **docs/DEPLOYMENT_SW6100.md**.

---

### What needs a GPU

| Runs on Mac CPU | Needs GPU |
|-----------------|-----------|
| Phase 0 (Whisper base/small), Phase 1 (tiny distillation), export/quantize | Phase 2 (Whisper-large teacher + full student training), int4 QAT |
