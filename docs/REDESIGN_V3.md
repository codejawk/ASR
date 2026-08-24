# Redesign v3 — distillation-first, two specialized siblings

This supersedes the "make the encoder fancier" instinct. The honest finding:
at 5–10 MB the biggest accuracy lever is **not architecture** — it's
**distillation from a large teacher** plus **the right head for each job**.
v3 keeps the two models *separate* (the always-on power budget demands it)
but makes them **siblings distilled from one teacher**.

## Why the two models stay separate (the load-bearing reason)

It is **power/duty-cycle**, not accuracy:

- **Model 2 (5 MB, commands)** must run **always-on** at eNPU power
  (~mW). You cannot run a 10 MB general encoder continuously without wrecking
  battery — so the command model must be independently runnable and tiny.
- **Model 1 (10 MB, general NLR)** runs **duty-cycled** on the Hexagon,
  waking only to transcribe.

So we share **knowledge (distillation)**, not **runtime weights**.

```
        [large teacher: Whisper-large / Parakeet-0.6B]
                 │  distillation + pseudo-labels
      ┌──────────┴───────────┐
      ▼                      ▼
 Model 1 (10 MB)        Model 2 (5 MB)
 general NLR            commands + SLU, always-on
 Zipformer TRANSDUCER   Conformer/Zipformer CTC + FST + intent/slot
 int4-QAT               int4-QAT (harder), eNPU-resident
```

## Right head for each job

| | Model 2 (5 MB) | Model 1 (10 MB) |
|---|---|---|
| Job | specific commands + intent/slot | general NLR / dictation |
| Head | **CTC + FST + intent/slot SLU** | **transducer** |
| Vocabulary | command grammar (open-vocab via FST) | full BPE-500 |
| Runs | always-on (eNPU) | duty-cycled (Hexagon) |
| Metric | FA/hour @ fixed FRR + intent/slot F1 | WER |

Note this is exactly "Conformer with the decoder removed" (= Conformer-CTC)
for Model 2, and the transducer kept for Model 1 — the principled version of
that earlier question.

## What is implemented in this repo (v3)

### 1. Distillation pipeline — `edge_asr/distill/`, `training/train_distill.py`
- **CTC-KD** (`kd_loss.ctc_kd_loss`): temperature-scaled KL between teacher
  and student CTC posteriors, frame-aligned (same subsampling → same frame
  rate). The transducer now exposes raw CTC logits + encoder features via
  `forward(..., return_features=True)`.
- **Feature-KD** (`kd_loss.feature_kd_loss`): optional MSE with a learned
  projection between student/teacher encoder dims.
- **Sequence-KD / pseudo-labelling** (`--pseudo-label`): the teacher
  transcribes unlabeled audio; the student trains on those hypotheses. This
  is the "teacher labels thousands of hours" path — plug a real
  Whisper/Parakeet teacher via `scripts/02_pseudo_label.sh`.
- Verified: teacher (1.35 M) → student (0.18 M), KD loss active, student
  converges. Refs: arXiv:2110.03334, arXiv:2409.13499.

### 2. On-device SLU (Model 2 "used wisely") — `edge_asr/slu/`, `training/train_slu.py`
- Joint **intent + slot** model (word-embed → BiGRU → intent head + BIO slot
  head), ~0.1 MB. Structured parsing, no cloud NLU:
  ```
  "set a timer for five minutes" -> {intent: timer, slots: {number: five, unit: minutes}}
  "call mom"                     -> {intent: call,  slots: {contact: mom}}
  ```
- Verified: intent acc 1.00, slot acc 1.00 on the synthetic grammar; parser
  emits structured actions.

## What to do with a real teacher

1. `scripts/02_pseudo_label.sh` → pseudo-label your unlabeled + wrist audio
   with Whisper-large / Parakeet-0.6B (sequence-KD data).
2. `train_distill --teacher <big.pt or wrap a real teacher> --pseudo-label`
   for Model 1; distill Model 2 the same way, specialized to the command
   grammar.
3. int4-QAT both (encoder int4, joiner int8, embedding fp16).
4. Export → QNN context binary for SW6100 (Conformer/Zipformer only; keep
   Mamba as the research arm).

## Selling points (for the pitch)

- **Deploys today** — standard ops, no exotic kernels.
- **Distillation-first** — a 10 M model that punches like a 100 M one.
- **Right tool per job** — CTC+FST for commands, transducer for dictation.
- **On-device SLU** — intent+slot on the wrist, a feature users feel.
- **Battery-honest** — always-on / duty-cycled two-tier split.
- **Open-vocab commands** — add by text, no retraining.
- **Private/offline** for the common paths.
