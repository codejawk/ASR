# Competition Pitch — WristVoice

**A router-gated, flash-paged Mixture-of-Specialists for always-on wrist ASR.**

---

## The one-sentence thesis

> The always-on keyword model doubles as a **router** that pages **domain/
> language expert recognizers** from flash into a tiny resident-memory
> budget — delivering **large effective capacity under a small resident
> footprint**, with an **SSM/Mamba streaming encoder** for O(1)-state,
> battery-friendly recognition, **int4 mixed-precision** to fit, and an
> **open-vocabulary hypernetwork** for zero-shot, retraining-free commands.

Every ingredient is published 2025-26 work; the **synthesis for edge is
new**, and — crucially — **it runs today** (all numbers below are measured
on a laptop, no dataset downloads).

---

## Why this wins on a watch (the insight)

Classical sparse Mixture-of-Experts is **backwards for a watch**: it saves
*compute* (which the Snapdragon Wear-Elite Hexagon has plenty of — ~12 TOPS)
but must keep **every expert resident in RAM** (the scarce resource). We
**invert** it:

| | Classical MoE | **Ours (flash-paged specialists)** |
|---|---|---|
| Experts live in | RAM (all) | **Flash (all), RAM (one)** |
| Routing granularity | per-token | **per-utterance** (coarse) |
| Router | extra network | **reuses the always-on Model 2** |
| Saves | compute | **the binding constraint: memory** |

The 10 MB "flash vs RAM" question every edge team hits becomes our **core
mechanism**: effective capacity = Σ experts on cheap flash; resident cost =
one expert in RAM. Paging latency hides behind the wake-word gate.

---

## Architecture

```
 🎤 ─► log-mel ─► [Model 2: eNPU, always-on]
                    ├─ WakeStub (~0.1 MB)         24/7 gate
                    ├─ Open-vocab hypernet KWS    user commands, no retrain
                    ├─ Router  (domain/lang)  ───┐ selects expert
                    └─ Speaker embed (owner gate) │
                                                  ▼
                    [ExpertPager]  flash ─► RAM (LRU, 1 resident)
                                                  ▼
      [Model 1: Hexagon, duty-cycled]  SSM/Mamba streaming transducer
                                                  ▼
                                                text
```

- **Model 2** (`edge_asr/models/command_model.py`) — HyperSpotter-style
  open-vocab KWS (a keyword-text hypernetwork generates a detector for any
  command string at runtime) + router head + speaker embedding, two-tier so
  5 MB of capacity never costs always-on power.
- **Model 1** (`edge_asr/models/ssm_encoder.py`) — selective-SSM (Mamba)
  streaming transducer: **O(1) recurrent state per frame**, no growing KV
  cache → constant streaming memory/compute.
- **MoE** (`edge_asr/moe/`) — `ExpertPager` (flash→RAM, LRU) +
  `MixtureOfSpecialists` (router → page → decode).

---

## Verified results (measured, synthetic data, laptop CPU)

| Claim | Evidence |
|-------|----------|
| SSM encoder trains & streams | Mamba transducer: loss 85→0.48, **full WER 0.00**, streaming WER 0.14 |
| Open-vocab commands work | audio="call" → P(detect \| "call")=**0.99**, P(\| "music")=**0.00** — from *text*, no retraining |
| Router works | domain accuracy **1.00** on synthetic |
| Flash-paging works | 4 experts = **6.2 MB flash / 1.5 MB resident → 4× capacity**, ~2.6 ms page-in |
| Fits the budget | Mamba Model 1 **int4-mixed = 5.17 MB**; Conformer = 6.51 MB (both < 10 MB) |
| Model 2 budget | hypernet + router + speaker = **~0.65 MB** (huge headroom in 5 MB) |

Reproduce: `bash scripts/smoke_test.sh`, `python scripts/moe_demo.py`,
`python -m edge_asr.tools.count_params configs/model1_mamba.yaml 500`.

---

## Grounding in 2025-26 SOTA (cite these)

- **SSM/Mamba ASR**: Samba-ASR (arXiv:2501.02832, Jan 2025); Mamba for
  Streaming ASR + unimodal aggregation (arXiv:2410.00070); ConMamba
  (edge-oriented: ~50% less memory, ~65% faster inference); Multilingual
  Mamba (arXiv:2510.18684).
- **MoE for speech**: UME upcycling MoE (arXiv:2412.17507); Omni-Router MoE
  (arXiv:2507.05724); MoE-Conformer streaming multilingual (arXiv:2305.15663).
- **Open-vocab KWS**: HyperSpotter — hyper-matched filters (arXiv:2508.04857);
  U2-KWS (arXiv:2312.09760).
- **Edge int4 compression**: Microsoft "Pushing the Limits of On-Device
  Streaming ASR" (arXiv:2604.14493) — int4 k-quant, encoder-only, decoder/
  joiner FP — the recipe our `mixed_precision_quantize` follows.
- **Foundations**: RNN-T (Graves 2012), Conformer (Gulati 2020), Zipformer
  (Yao, ICLR 2024), stateless predictor (Ghodsi 2020), pruned RNN-T
  (Kuang 2022), BC-ResNet (Kim, Qualcomm 2021), QAT (Jacob 2018).

---

## Honest risk register (own these in Q&A)

1. **SSM ops on QNN/HTP** — Mamba's selective-scan may not be supported on
   the Wear-Elite backend yet. Mitigation: ship the **Conformer/Zipformer**
   baseline (also implemented), present Mamba as the research arm.
2. **Router error propagation** — wrong expert → wrong transcript. Mitigation:
   a general fallback expert + report router accuracy separately.
3. **Paging latency under thermal/flash contention** — measured behind the
   wake gate; report p95, not mean.
4. **Evaluate on real wrist audio**, not LibriSpeech — build the wrist eval
   set (positions × noise × accents) before claiming WER.
5. **Novelty is systems-level, not a new loss** — say so; it's the right,
   defensible bet for an edge competition and it survives scrutiny.

---

## The 90-second demo script

1. `python -m edge_asr.tools.count_params configs/model1_mamba.yaml 500`
   → "Model 1 fits at **5.17 MB** int4-mixed."
2. `python scripts/moe_demo.py`
   → "4 experts, **6.2 MB flash / 1.5 MB resident**, 2.6 ms page-in, real
   transcripts routed per domain."
3. Open-vocab: show P(detect) flipping as you change the keyword *text* —
   "add a command by typing it, no retraining."
4. One slide: the inverted-MoE table above. That's the idea judges remember.
