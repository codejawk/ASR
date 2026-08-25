# WristVoice — On-Device ASR for Smartwatches under 10 + 5 MB

### Technical Report

**Target hardware:** Snapdragon Wear Elite / SW6100 (dual-NPU).
**Repository:** https://github.com/codejawk/ASR — 76 files, ~3,000 LOC, runs on a laptop with no downloads.
**Status:** all components implemented and verified on synthetic data; real-speech training is the documented next phase.
**Date:** 2026-08-25.

---

## Abstract

We present **WristVoice**, a complete, runnable system for two on-device
speech models on a smartwatch under hard memory budgets: **Model 1** for
general natural-language recognition (≤ 10 MB) and **Model 2** for commands
(≤ 5 MB). The central design decision is not a new architecture but a
**training strategy**: the two models are **specialized siblings distilled
from one large teacher**, kept physically separate because the always-on
power budget demands it. Model 1 is a streaming RNN-T transducer with a
selectable Conformer-lite or Mamba/SSM encoder; Model 2 is a Conformer-CTC
keyword model extended with an **open-vocabulary hypernetwork** and a
**joint intent+slot spoken-language-understanding (SLU)** head. We add a
**flash-paged Mixture-of-Specialists** for the multilingual case and an
**int4 mixed-precision** export path. Every claim in §8 is measured; the
numbers verify the *machinery* on synthetic data — real word-error-rate
requires a large teacher and real wrist audio, which we scope honestly.

---

## 1. Problem statement

Two speech tasks must run on a watch under three constraints:

| Constraint | Reality |
|------------|---------|
| **Memory** | 10 + 5 MB *resident* — **not** a compute limit (the chip runs 2 B-param models). It is RAM residency, OTA/APK size, and a shared Wear-OS pool. |
| **Power** | Commands must be recognized **always-on** at eNPU milliwatts. A 10 MB general encoder cannot run continuously without wrecking battery. |
| **Deployment** | int4/int8 on the **QNN/HTP** backend: standard operators only, static shapes, no dynamic control flow in the graph. |

**Contributions (all implemented in the repo):**
1. A two-model, two-tier system respecting the power/duty-cycle split (§3).
2. A streaming transducer with **both** a deployable Conformer-lite encoder and a research **Mamba/SSM** encoder behind one interface (§4).
3. A command model that does **open-vocab detection + on-device SLU** (§5).
4. A **distillation pipeline** — CTC-KD, sequence-KD (pseudo-labels), feature-KD (§6.1).
5. **int4 mixed-precision** export following the on-device recipe (§6.3).
6. A **flash-paged Mixture-of-Specialists** for the multilingual case (§6.5).
7. A **self-contained, dependency-light** implementation (pure-PyTorch RNN-T loss, hand-built log-mel) with a synthetic learnable-audio harness that verifies the whole pipeline in CI (§7).

---

## 2. Background: ASR as a stack

Every modern system is `frontend → encoder → head`. What determines
on-device viability is the **head**:

| Head | Streaming? | Cost | Used here |
|------|-----------|------|-----------|
| **CTC** | trivially | cheapest (a linear layer) | Model 2, aux head |
| **RNN-T / transducer** | natively | +predictor +joiner | **Model 1 (primary)** |
| AED (Whisper/Moonshine) | with tricks | full decoder | teacher only |
| LLM-decoder | no | GB-scale | teacher only |

"Streaming vs offline" is architectural, not tunable. The frontend sets the
frame rate (100 Hz here) that all downstream compute scales with; the
encoder subsamples early (→ 25 Hz) to cut cost. Compression levers, ranked
by real impact: **distillation ≫ int8 QAT > int4 mixed-precision >
structured pruning / low-rank / weight-sharing**.

---

## 3. System architecture

```
 mic → log-mel (80-d, 100 Hz)
        │
        ▼
 ┌─────────────────────────────────────────────┐
 │ Model 2  (eNPU, always-on, ~mW)             │
 │   wake stub → open-vocab KWS → router →     │──► direct action / SLU
 │   intent+slot SLU → speaker gate            │
 └───────────────────────┬─────────────────────┘
                         │ wake (gate)
                         ▼
 ┌─────────────────────────────────────────────┐
 │ Model 1  (Hexagon, duty-cycled)             │
 │   streaming transducer + aux CTC            │──► text
 └─────────────────────────────────────────────┘
```

**Why two separate models (the load-bearing reason):** *power*, not
accuracy. Model 2 must listen continuously at eNPU power; Model 1 is a heavy
Hexagon workload that wakes on demand. So the two share **knowledge
(distillation)**, not **runtime weights** — they are siblings from one
teacher (§6.1). Model 2 also acts as the **gate**: the battery-hungry
recognizer only wakes when there is something to transcribe.

---

## 4. Model 1 — general natural-language recognition (≤ 10 MB)

Streaming **RNN-T transducer** (stateless predictor + small joiner) with an
**auxiliary CTC head** (loss weight 0.2) that speeds convergence and gives a
cheap non-autoregressive fallback path. `edge_asr/models/transducer.py`.

**Encoders (one interface, two implementations):**
- **Conformer-lite, cache-aware streaming** (`streaming_encoder.py`) — the
  *deployable* baseline. Conv subsampling 100→25 Hz, macaron FFN + cached
  MHSA + causal depthwise conv. Standard ops → runs on QNN today. Config
  knobs mirror icefall's **Zipformer** for a mechanical swap.
- **Mamba/SSM, selective-scan** (`ssm_encoder.py`) — the *research* arm.
  **O(1) recurrent state per frame** (no growing KV cache) → constant
  streaming memory/compute, battery-relevant. Faithful S6 implementation
  with the Mamba `dt` init; requires excluding weight-decay from `A_log`/
  `D`/`dt`-bias (`training/utils.configure_optimizer`), otherwise RNN-T gets
  stuck emitting all-blank (a real gotcha we hit and fixed).

**Streaming state contract = ONNX/QNN I/O contract.** `forward_chunk(chunk,
state) → (enc, new_state)` with static shapes; the streaming loop lives in
host code because QNN has no `Loop`/`If`.

**Parameter budget** (verified by `tools/count_params.py`):

| Config | encoder | total | int8 | int4-mixed |
|--------|--------:|------:|-----:|-----------:|
| Conformer (d=224, 11 L) | 10.47 M | 10.96 M | 10.96 MB | **6.51 MB** |
| Mamba (d=256, 14 L)     | 8.04 M  | 8.55 M  | 8.55 MB  | **5.17 MB** |

int4-mixed = encoder int4 + joiner int8 + decoder embedding fp16. Both fit
under 10 MB; the SSM encoder is more parameter-efficient.

---

## 5. Model 2 — commands + on-device SLU (≤ 5 MB)

The command model does more than pick a label from a fixed list — it does
**spoken-language understanding** on the wrist.

- **Open-vocab hypernetwork KWS** (`command_model.py`, HyperSpotter-style):
  a keyword-*text* encoder generates a matched-filter detector for an
  arbitrary command string at runtime → add commands by typing, no
  retraining. Verified: P(detect | correct)=**0.99** vs P(| mismatched)=**0.00**.
- **Router head** → domain/language, drives expert selection (§6.5).
- **Speaker embedding** → optional owner-gating.
- **Joint intent + slot SLU** (`slu/intent_slot.py`): word-embed → BiGRU →
  intent classifier + per-token BIO slot head (~0.1 MB). Structured parsing:
  ```
  "set a timer for five minutes" → {intent: timer, number: five, unit: minutes}
  "call mom"                     → {intent: call, contact: mom}
  ```
- **Head choice: CTC + FST**, not a transducer — commands don't need heavy
  language modeling; CTC streams trivially, decodes with one argmax/frame,
  and quantizes/exports cleanly (the principled version of "Conformer with
  the decoder removed"; see Appendix A).
- **Two-tier power design:** a ~0.1 MB always-on wake stub gates the full
  5 MB stack, so the capacity never costs always-on power.
- **Metric:** false-accepts/hour at fixed false-reject-rate (not accuracy),
  tuned on real ambient audio (`decode/keyword_ctc.py`).

---

## 6. Methods

### 6.1 Distillation (the primary accuracy lever) — implemented
At ~10 M params a large teacher is worth ~20–40 % relative WER; architecture
is worth ~2 %. `edge_asr/distill/`, `training/train_distill.py`:
- **CTC-KD** — temperature-scaled KL between teacher and student CTC
  posteriors, frame-aligned (same subsampling → same frame rate).
- **Sequence-KD / pseudo-labelling** (`--pseudo-label`) — the teacher
  transcribes unlabeled + wrist audio; the student trains on those
  hypotheses. Plug a real Whisper/Parakeet teacher via `02_pseudo_label.sh`.
- **Feature-KD** — projected MSE between encoder features.
Verified: teacher (1.35 M) → student (0.18 M), KD loss active, student
converges. Refs [KD-transducer], [Whisper-KD].

### 6.2 Self-supervised pretraining (BEST-RQ) — planned
Pretrain the small encoder on unlabeled + wrist audio before distillation to
punch above its parameter weight and adapt to the device mic. Ref [BEARD].

### 6.3 Quantization — implemented (PTQ), QAT scoped
- `export/quantize.py`: int8 dynamic PTQ (a size *ceiling*) and **int4
  mixed-precision** via ORT's `MatMulNBitsQuantizer` (k-quant preferred).
- Ship path is **int8 QAT** (encoder int4 possible): W8A8 per-channel,
  decoder embedding fp16. `count_params.py` reports the int4-mixed *floor*
  the model targets. Ref [Microsoft-OnDevice].

### 6.4 Elastic / Matryoshka width — planned
Train the encoder so a thin width-slice is itself valid → one model serves a
low-battery/short-utterance operating point and full dictation from shared
weights. Refs [MoME], [DynamicEncoderSize].

### 6.5 Flash-paged Mixture-of-Specialists — implemented (multilingual case)
Classical MoE is backwards for a watch: it saves compute (abundant) and
costs resident memory (scarce). We **invert** it — experts live in flash,
one resident in RAM, a per-utterance router selects and pages one in.
`edge_asr/moe/`. Verified: 4 experts = **6.2 MB flash / 1.5 MB resident (4×
capacity)**, ~2.6 ms page-in. Best reserved for multi-*language* experts.

---

## 7. Implementation

- **Dependency-light & self-contained:** a numerically-stable **pure-PyTorch
  RNN-T loss** (auto-uses torchaudio's CUDA kernel when present), a
  hand-built **log-mel frontend** with causal online CMVN — the whole
  train → decode → export → quantize pipeline runs with only
  `torch`, `numpy`, `onnxruntime`, `sentencepiece`.
- **Synthetic learnable-audio harness** (`data/synth.py`): each character
  maps to a tone, so a tiny model actually converges to WER 0 — verifying
  the *whole* pipeline in CI with no dataset.
- **Repo:** 49 Python modules (all import cleanly), configs, docs, tests;
  `bash scripts/smoke_test.sh` runs v1 + v2 + v3 green.

---

## 8. Experimental verification

**Setup:** synthetic speech-like audio, laptop CPU, no downloads. These
verify the machinery, not real-speech accuracy.

| Result | Value | Source |
|--------|-------|--------|
| Model 1 (Mamba) synthetic decode | loss 85 → 0.48; **full WER 0.00**, streaming WER 0.14 | `test_v2_components.py`, training runs |
| Model 1 (Conformer) synthetic decode | loss 95 → 1.2; **WER 0.00** | `train_model1` |
| Model 1 budget | int4-mixed **5.17 MB** (Mamba) / 6.51 MB (Conformer) | `count_params.py` |
| Model 2 open-vocab KWS | P(detect\|correct)=**0.99**, P(\|mismatch)=**0.00**; router acc **1.00** | `train_command` |
| SLU intent + slot | **1.00 / 1.00**; ~0.1 MB | `train_slu` |
| Distillation | teacher 1.35 M → student 0.18 M, CTC-KD active | `train_distill` |
| Flash-paged MoE | 6.2 MB flash / 1.5 MB resident (**4×**), ~2.6 ms page-in | `moe_demo.py` |
| Deployability | 100 % standard ops (Conformer path) → QNN | export |
| Integrity | 49/49 modules import; full smoke green | CI |

**Honest scope.** Every number above is real, but the task is synthetic and
labels are ground-truth, so the student learns with or without KD — these
prove *correctness of the machinery*. The accuracy *benefit* of distillation
rests on the cited results, and real WER requires a large teacher + real
wrist audio + GPU training (§11).

---

## 9. Deployment on SW6100

- **Dual NPU:** eNPU runs Model 2 (always-on); Hexagon runs Model 1
  (duty-cycled).
- **QNN EP:** no `Loop`/`If` → 3-graph static-shape export
  (encoder-chunk / decoder-step / joiner-step), streaming loop in host code.
- **Quantize on x86_64, infer on arm64.** Precompile a **QNN context
  binary** for the Wear-Elite Hexagon — do not reuse SM88xx prebuilts.
- **Custom-op watchlist** if swapping to real Zipformer: BiasNorm, Swoosh.
- **The 10 MB is flash or resident RAM?** — resolve this first; it decides
  whether to ship a larger paged model or keep shrinking the resident one.

---

## 10. Limitations & honesty

1. **Synthetic data only so far** — no real WER.
2. **Mamba is not NPU-deployable today** — the selective-scan op has no QNN
   kernel; ship Conformer/Zipformer, keep Mamba as research.
3. **Dynamic PTQ overshoots the budget** (~14 MB); the int4/int8 *floor*
   (~5–6.5 MB) needs QAT/static.
4. **Elastic-width for a streaming transducer is a research bet**, not a
   bolted-down published result for audio-only streaming.
5. **Router errors propagate** in the MoE path (wrong expert → wrong text).
6. **Evaluate on real wrist audio**, not LibriSpeech, before any WER claim.

---

## 11. Roadmap

1. Pseudo-label unlabeled + wrist audio with a large teacher (Whisper/Parakeet).
2. Distill Model 1 (transducer) and Model 2 (CTC + SLU) from that teacher.
3. int4 QAT both; validate on a real wrist eval set (WER + FA/hour + p95 latency).
4. QNN bring-up on SW6100 silicon; regenerate context binaries.

---

## 12. Related work & references

> Foundational references are well-established; the recent (2024–26) items
> come from a literature scan and their exact IDs should be verified before
> external citation.

**Foundations.** CTC — Graves et al., ICML 2006. RNN-T — Graves, 2012
(arXiv:1211.3711). Conformer — Gulati et al., Interspeech 2020
(arXiv:2005.08100). Zipformer — Yao et al., ICLR 2024 (arXiv:2310.11230).
Stateless predictor — Ghodsi et al., ICASSP 2020. Pruned RNN-T — Kuang et
al., Interspeech 2022. FastConformer — Rekesh et al., ASRU 2023. BC-ResNet —
Kim et al. (Qualcomm), Interspeech 2021 (arXiv:2106.04140). SubSpectralNorm —
Chang et al., ICASSP 2021. SpecAugment — Park et al., 2019
(arXiv:1904.08779). Distillation — Hinton et al., 2015 (arXiv:1503.02531).
int8 QAT — Jacob et al., CVPR 2018 (arXiv:1712.05877).

**Recent (2024–26).** Samba-ASR (SSM) — arXiv:2501.02832. Mamba for
streaming ASR — arXiv:2410.00070. Multilingual Mamba — arXiv:2510.18684.
MoME (Matryoshka + MoE) — NeurIPS 2025. Adaptive AVSR via Matryoshka LLMs —
arXiv:2503.06362. Dynamic Encoder Size — arXiv:2407.18930. HyperSpotter
(open-vocab KWS) — arXiv:2508.04857. U2-KWS — arXiv:2312.09760. On-device
streaming ASR, int4 k-quant (Microsoft) — arXiv:2604.14493. KD for neural
transducers from SSL — arXiv:2110.03334. Fast streaming transducer via
Whisper-KD — arXiv:2409.13499. BEARD (BEST-RQ + distillation) —
arXiv:2510.24570. Flavors of Moonshine (tiny edge ASR) — arXiv:2509.02523.

---

## Appendix A — Conformer-CTC vs transducer

Removing the decoder from a Conformer yields **Conformer-CTC** — a real,
widely-shipped design. **Pros:** simplest head, fastest decoding (one
argmax/frame, no autoregression → NPU-friendly), easiest to quantize/export,
lowest latency. **Cons:** the conditional-independence assumption models
language weakly → higher WER than RNN-T at equal encoder size on free
dictation; usually needs an external n-gram/FST LM. **Size myth:** the
predictor+joiner removed is only ~0.4 M of ~11 M (~3–4 %), so it barely
shrinks the model — the win is simplicity/latency, not size. **Verdict:**
CTC for Model 2 (commands), transducer for Model 1 (dictation) — exactly the
split adopted here.

## Appendix B — Reproduce

```bash
git clone https://github.com/codejawk/ASR.git && cd ASR
pip install -r requirements.txt
bash scripts/smoke_test.sh                                   # v1+v2+v3 green
python -m edge_asr.tools.count_params configs/model1_mamba.yaml 500
python scripts/moe_demo.py                                   # flash-paged experts
python -m edge_asr.training.train_slu --steps 500 --out runs/slu
```
