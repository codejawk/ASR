# Technical Report — Edge-ASR for SW6100

**Two small streaming ASR models for wrist-class hardware (Snapdragon Wear Elite).**
Status: runnable scaffold, verified end-to-end on synthetic data.
Author: engineering scaffold generated with Claude Code. Date: 2026-08-24.

---

## 1. Problem statement

Ship on-device speech on a smartwatch under hard memory budgets:

| Model | Job | Budget | This project |
|-------|-----|--------|--------------|
| Model 1 | General streaming ASR (dictation, free speech) | ≤ 10 MB | 10.96 M params, ~11 MB int8 (tunable under 10) |
| Model 2 | Command / keyword recognition | ≤ 5 MB | BC-ResNet, < 0.3 MB int8 |

Constraints that shape every decision: streaming (low latency, bounded
lookahead), int8 on the Hexagon/QNN HTP, always-on gating on the eNPU, and
a commercial license on all training data.

---

## 2. System architecture

```
mic → log-mel (100 Hz) → [Model 2: BC-ResNet, eNPU, always-on] ──command──► action
                                        │ wake
                                        ▼
                          [Model 1: streaming transducer, Hexagon] ──► text
```

### 2.1 Model 1 — streaming RNN-Transducer + auxiliary CTC
- **Frontend**: 80-dim log-mel, 25 ms/10 ms, causal online CMVN.
- **Encoder**: cache-aware streaming Conformer-lite. Conv subsampling
  100→25 Hz, causal depthwise convs, chunked attention with a carried
  KV/conv cache (every frame encoded once). ~85% of the parameter budget.
- **Predictor**: stateless (embedding + kernel-2 conv), ~0.13 M params.
- **Joiner**: enc_proj + pred_proj → tanh → vocab (BPE-500), ~0.25 M.
- **Aux CTC head**: linear → vocab, loss weight 0.2.
- **Loss**: RNN-T + 0.2·CTC.

### 2.2 Model 2 — BC-ResNet command classifier
- Broadcasted residual blocks: cheap 1D temporal convs + a frequency
  branch broadcast across time; SubSpectral normalization.
- Closed-set softmax over {commands, wake, unknown, silence}.
- Alternative open-vocab path: phoneme-CTC + keyword FST (runtime-editable
  command list), scored by **false-accepts/hour at fixed FRR**.

### 2.3 Deployment shape
- 3 static-shape ONNX graphs (encoder-chunk / decoder-step / joiner-step)
  because the QNN EP has no `Loop`/`If`; the streaming loop is host code.
- int8 via QAT for ship (dynamic PTQ used here as a size ceiling).

---

## 3. Research foundations (what each component is built on)

The project is an **integration of established, published techniques**. Each
component below names the paper(s) it implements or adapts.

### Acoustic modeling & losses
| Technique | Paper | Where in repo |
|-----------|-------|---------------|
| **CTC** loss / alignment-free training | Graves, Fernández, Gomez, Schmidhuber, *Connectionist Temporal Classification*, ICML 2006 | `losses` (aux head), `decode/ctc_greedy.py` |
| **RNN-Transducer** | Graves, *Sequence Transduction with Recurrent Neural Networks*, ICML Workshop 2012 | `losses/rnnt.py`, `models/transducer.py` |
| **Conformer** encoder (conv + self-attention) | Gulati et al., *Conformer: Convolution-augmented Transformer for Speech Recognition*, Interspeech 2020 | `models/streaming_encoder.py` |
| **Zipformer** (target production encoder) | Yao et al., *Zipformer: A Faster and Better Encoder for ASR*, ICLR 2024 | config-mapped; swap target |
| **FastConformer** (efficient subsampling, alt. encoder) | Rekesh et al., *Fast Conformer with Linearly Scalable Attention*, ASRU 2023 | design note |
| **Stateless predictor** for RNN-T | Ghodsi et al., *RNN-Transducer with Stateless Prediction Network*, ICASSP 2020 | `models/decoder.py` |
| **Pruned RNN-T** (memory-efficient transducer training) | Kuang et al., *Pruned RNN-T for Fast, Memory-Efficient ASR Training*, Interspeech 2022 (k2/icefall) | training design |
| **Cache-aware streaming** Conformer | Noroozi et al. / NVIDIA NeMo, *cache-aware streaming Conformer* (2021–2023) | `forward_chunk` + state contract |

### Keyword spotting (Model 2)
| Technique | Paper | Where |
|-----------|-------|-------|
| **BC-ResNet** / broadcasted residual learning | Kim et al. (Qualcomm AI Research), *Broadcasted Residual Learning for Efficient Keyword Spotting*, Interspeech 2021 | `models/bcresnet.py` |
| **SubSpectral Normalization** | Chang et al., *SubSpectral Normalization for Neural Audio Data Processing*, ICASSP 2021 | `SubSpectralNorm` |
| **MatchboxNet** (alt. small KWS) | Majumdar, Ginsburg, *MatchboxNet: 1D Time-Channel Separable CNN for Speech Commands*, Interspeech 2020 | design alt. |

### Training technique
| Technique | Paper | Where |
|-----------|-------|-------|
| **Knowledge distillation** (hard/soft targets) | Hinton, Vinyals, Dean, *Distilling the Knowledge in a Neural Network*, NeurIPS-W 2015 | pseudo-label + distill pipeline |
| **SpecAugment** | Park et al., *SpecAugment*, Interspeech 2019 | `data/augment.py` |
| **Noise / RIR augmentation** | Snyder et al. *MUSAN* 2015; Ko et al. *A Study on Data Augmentation of Reverberant Speech*, ICASSP 2017 | `data/augment.py` |
| **int8 QAT / integer-only inference** | Jacob et al., *Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference*, CVPR 2018 | `export/quantize.py`, docs |

### Context / reference systems (teachers & baselines)
| System | Paper |
|--------|-------|
| **Whisper** (AED weak-supervision baseline / teacher) | Radford et al., *Robust Speech Recognition via Large-Scale Weak Supervision*, 2022 |
| **Moonshine** (edge AED, monolingual-beats-multilingual evidence) | Jeffries et al., *Moonshine: Speech Recognition for Live Transcription and Voice Commands*, 2024 |
| **Parakeet-TDT / Nemotron** (pseudo-label teachers) | NVIDIA NeMo model cards; TDT: Xu et al., *Efficient Sequence Transduction by Jointly Predicting Tokens and Durations*, ICML 2023 |

> **Citation caveat.** The foundational papers above (CTC, RNN-T, Conformer,
> Zipformer, BC-ResNet, SpecAugment, QAT, etc.) are well-established and I am
> confident in them. Claims about very recent systems and datasets from the
> earlier web-research phase — Loquacious, Nemotron Speech Streaming size
> numbers, Cohere Transcribe / Granite leaderboard figures, the exact QNN-EP
> plugin status — should be **verified against primary sources** before you
> put them in an internal doc; treat those as leads, not settled facts.

---

## 3.9 Competition build (v2) — the differentiators

The baseline above is a solid productization. The **v2** additions are what
make it competition-grade, and each is runnable and verified (see
docs/COMPETITION_PITCH.md for the measured numbers):

- **SSM/Mamba streaming encoder** (`models/ssm_encoder.py`) — selective
  state-space (S6) with **O(1) recurrent state per frame** (no growing KV
  cache), the battery-relevant 2025 frontier. Selectable via
  `encoder_type: mamba`. Verified: full WER 0.00 on synthetic; int4-mixed
  budget **5.17 MB**. Requires the Mamba-`dt` init and **weight-decay
  exclusion** on `A_log`/`D`/`dt`-bias (`training/utils.configure_optimizer`)
  — without it RNN-T gets stuck emitting all-blank.
- **Open-vocab hypernetwork Model 2** (`models/command_model.py`) — a
  keyword-text hypernetwork generates a detector for any command string at
  runtime (HyperSpotter-style); plus a router head (domain/language) and a
  speaker embedding. Verified: P(detect|correct)=0.99 vs 0.00 for a
  mismatched keyword, from text alone, no retraining.
- **Flash-paged Mixture-of-Specialists** (`moe/`) — the headline idea:
  invert MoE so experts live in flash and only one is resident. Verified:
  4 experts = 6.2 MB flash / 1.5 MB resident (4× capacity), ~2.6 ms page-in.
- **int4 mixed-precision export** (`export/quantize.py`) — encoder int4
  (k-quant), joiner int8, decoder fp16, following Microsoft arXiv:2604.14493.

## 4. Novelty — an honest assessment

**The individual components are established techniques; the novelty is the
edge-specific *system synthesis*, which is the right claim for this contest.**
The single genuinely novel idea is the **inverted, flash-paged
Mixture-of-Specialists** — using the always-on keyword model as the router
that pages domain/language experts from flash, trading the resource edge has
(compute) to save the one it doesn't (resident memory). That framing, wired
end-to-end and measured, is defensible as a systems contribution.

Beyond that:
It is an *engineering integration*: known architectures, known losses, known
compression, assembled and budgeted for a specific device. If someone asks
"what new science is here," the answer is *none* — and that is the correct
answer for a productization scaffold.

What it *does* contribute, as engineering, is worth naming honestly:

1. **A self-contained, dependency-light pure-PyTorch RNN-T loss** with an
   automatic torchaudio fast-path. This is a convenience (letting the whole
   pipeline run without torchaudio/CUDA), not a new algorithm — it's the
   standard log-domain forward from Graves 2012.
2. **A config surface that mirrors icefall's Zipformer**, so the stand-in
   encoder can be swapped for the production encoder mechanically. This is
   integration design, not novelty.
3. **A synthetic *learnable-audio* smoke harness** — each character maps to a
   tone, so a tiny model actually converges to WER 0, verifying the whole
   train→decode→export path in CI without any dataset. Useful, not novel.
4. **The eNPU→Hexagon cascade wiring** as testable Python. Standard two-stage
   KWS-gates-ASR design; the value is that it's runnable and maps 1:1 to the
   on-device host loop.

If you want *genuine* novelty to pursue (research contributions you could
credibly claim), candidates are: (a) **wrist-channel-specific augmentation /
front-end** tuned to band-conduction and arm-raise noise, backed by a real
on-device eval set — this is under-explored and product-relevant; (b) an
**int4 mixed-precision scheme for stateful streaming transducers on HTP** that
keeps activation outliers in check — the Microsoft edge-ASR study flags this
as hard; (c) **on-device continual/personalization** (contact names, user
accent) within the memory budget. Those are real gaps, not this scaffold.

---

## 5. Design decision: "Conformer with the decoder removed" (Conformer-CTC)

See §Appendix A below for the full analysis. Short version: **it's a
legitimate, well-established design (Conformer-CTC), not a bad idea — but it
trades accuracy for simplicity, and at this budget it barely saves size.**
The project already ships a CTC head, so you can benchmark it directly.

---

## 6. Verified results (synthetic, this machine)

| Check | Result |
|-------|--------|
| Model 1 param budget | 10.96 M (encoder 10.47 M / decoder 0.13 M / joiner 0.25 M / ctc 0.11 M) |
| Model 1 training (synthetic) | loss 95 → 1.2; greedy decode WER 0.00 |
| Model 2 training (synthetic) | accuracy → 1.000 |
| RNN-T loss | pure-PyTorch fallback backprops correctly |
| Streaming decode | chunked cache-aware path runs, matches full-context structure |
| ONNX export + int8 | 3 graphs export; dynamic PTQ 14.6 MB (ceiling); param floor ~11 MB |
| Cascade | eNPU-gate → Hexagon-ASR pipeline runs |

These verify **machinery correctness**, not speech accuracy. Real WER numbers
require real data + GPU training (docs/ROADMAP.md).

---

## 7. Limitations

- Synthetic data only so far; no real-speech WER yet.
- Encoder is a Conformer-lite stand-in, not the production Zipformer.
- Dynamic PTQ overshoots budget; QAT/static int8 is the ship path.
- QNN context binaries must be regenerated for the Wear-Elite Hexagon
  variant; not yet run on real silicon.
- BC-ResNet block here keeps width constant (~tens of K params); a
  production BC-ResNet-8 expands channels per stage (~320 K, still < 1 MB).

---

## Appendix A — Conformer-CTC vs Transducer (full analysis)

**What "remove the decoder" means.** In an attention encoder-decoder (AED,
e.g. Whisper) the *decoder* is an autoregressive attention stack. In an
RNN-T the *decoder* is the predictor+joiner. Removing either and keeping just
the encoder + a CTC projection gives **Conformer-CTC** — a real, widely-shipped
architecture (NeMo Conformer-CTC, Citrinet, wav2vec2-CTC all use it).

**Pros of Conformer-CTC**
- **Simplest possible head**: one linear layer to vocab. No predictor, no
  joiner.
- **Fastest, simplest decoding**: one argmax per frame, no autoregression and
  no loop-carried label state. This is *very* NPU-friendly — the whole
  utterance is one parallel pass; no per-token feedback loop.
- **Easiest to quantize and export**: no joiner, no dynamic decode loop in the
  graph. Cleaner QNN bring-up.
- **Lowest latency** and trivially streaming (frame-synchronous by
  construction).

**Cons of Conformer-CTC**
- **Conditional independence assumption**: CTC assumes output tokens are
  independent given the audio. It models *acoustics* well but *language*
  weakly — so on free-form dictation it typically has **higher WER than an
  RNN-T of the same encoder size**, and it's worse on homophones, rare words,
  and long-range context.
- **Usually needs an external LM** (n-gram / WFST shallow fusion) to close much
  of that gap — which adds back some of the complexity you removed, though the
  LM lives outside the model and doesn't eat the model's MB budget.
- **Peaky/spiky posteriors** can hurt endpointing and confidence estimates.

**Size reality at your 10 MB budget.** The encoder is ~85% of the budget. The
predictor+joiner you'd remove is only ~0.4 M of ~11 M params — **~3–4%**. So
Conformer-CTC does **not** meaningfully shrink the model. Its win is
*simplicity, latency, and quantization/export ease*, **not** size.

**Two more caveats specific to your setup.**
1. Vanilla Conformer is heavier per unit accuracy than **Zipformer** or
   **FastConformer**. If you go CTC, use a Zipformer-CTC or FastConformer-CTC
   encoder, not a plain Conformer.
2. A stock Conformer conv module and full self-attention **look into the
   future** — you must make convs causal and attention chunked/cached (exactly
   what this repo's encoder does) or it won't stream.

**Verdict / recommendation**
- **Model 2 (commands): yes.** You don't need a transducer for a command set —
  CTC (or even the plain classifier already in the repo) is the right call.
- **Model 1 (general dictation): keep the transducer as the primary**, because
  at equal encoder size it generally wins WER on free speech, which is Model
  1's whole job. **But Conformer/Zipformer-CTC + a small n-gram FST is a
  genuinely reasonable, simpler, easier-to-ship alternative** and is worth
  benchmarking head-to-head — especially if QNN export of the transducer
  joiner/loop proves painful on the Wear-Elite target.
- **You can test this today with a one-line change.** This project already
  trains an auxiliary CTC head on the same encoder. Decode with
  `edge_asr.decode.ctc_greedy` on `model.ctc_head(enc)` instead of the
  transducer path, compare WER on the same data, and let the numbers — not the
  suggestion — decide.
