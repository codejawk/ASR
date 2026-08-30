# On-Device ASR for Wearables — From Basics to Our Winning Approach

**Project:** WristVoice — streaming speech recognition on a smartwatch (Snapdragon Wear Elite / SW6100)
**Budget:** Model 1 (general) ≤ 10 MB · Model 2 (commands) ≤ 5 MB · streaming · on-device · no cloud
**Repo:** https://github.com/codejawk/ASR

This single document explains the whole thing from zero: the vocabulary
(with examples), the techniques the field uses today, and exactly how our
approach improves on them. Running example throughout: you say **"call mom."**

---

## Part 1 — The vocabulary (with examples)

### The task
- **ASR (Automatic Speech Recognition)** — turn speech into text. *"call mom" (sound) → `call mom` (text).*
- **Streaming vs offline** — streaming shows words *as you speak* (low latency); offline waits until you finish. A watch needs streaming.

### Sound → numbers (the frontend)
- **Waveform** — raw audio: 16,000 amplitude numbers per second.
- **Log-mel / spectrogram ("features")** — the waveform converted into "how much of each frequency, over time" (like an equalizer). *"80-mel @ 100 Hz" = 80 frequency bands, a snapshot every 10 ms.* "call mom" (~1 s) → an 80 × ~100 grid.
- **Frame** — one 10 ms snapshot.
- **CMVN** — rescale features so loud/quiet, near/far mic don't confuse the model. **Causal/online** = uses only past audio (required for streaming).
- **Subsampling** — squeeze frames 4× (100→25/sec) so the heavy part does 4× less work.

### The network
- **Parameters (weights)** — the model's learned numbers. *"10 M params" = 10 million.* More = bigger, usually smarter.
- **Encoder** — the big network that reads the feature grid and outputs a rich "acoustic meaning" representation. **~85 % of the model.** It does *not* output text.
- **Decoder / head** — the part that turns the encoder output into text.

### Features → words (the two main "heads")
- **Token / vocab / BPE** — the output alphabet. **BPE-500** = ~500 common subword pieces. *"timer" = 1 token; "nestorius" = `nest`+`or`+`ius`.*
- **Blank** — a special "emit nothing here" token used by CTC and transducers.
- **CTC** — the *simplest* head: at each frame guess a token or blank, then collapse repeats/blanks. Cheap, streams trivially. **Weakness: treats each output independently — no sense of word order.** *`c-c-a-l-l-_-m-o-m` → `call mom`.*
- **RNN-T / Transducer** — a *smarter* head that remembers what it wrote:
  - **Predictor** — looks at words emitted so far ("language history"). *After "call", it expects a name.*
  - **Joiner** — combines encoder (acoustics) + predictor (language) → next token.
- **AED / attention decoder** — Whisper's kind: writes a word, looks at it, writes the next (autoregressive). Accurate but **offline**.
- **Beam search / LM / FST** — smarter decoding: keep several candidate sentences (beam), score with a **language model (LM)**, or constrain with an **FST** (a small graph of allowed words). *An FST of your contacts helps "call **Nandhini**" resolve.*

### Training
- **Teacher / Student** — big accurate model (Whisper, ~1.5 B) teaches a small shippable one (~10 M).
- **Distillation (KD)** — train the student to imitate the teacher → it "punches above its weight."
- **Pseudo-labeling** — the teacher auto-transcribes unlabeled audio → cheap training data.
- **SSL pretraining (self-supervised)** — pretrain the encoder on raw audio with no labels first (BEST-RQ, wav2vec2), then fine-tune.
- **WER / CER** — Word/Character Error Rate. Lower = better. *"call mom" → "call tom" = 1 of 2 words wrong = WER 0.5.*

### Making it small
- **fp32 / fp16 / int8 / int4** — bits per number: 4 / 2 / 1 / 0.5 bytes. *10 M params × 1 byte = 10 MB at int8; × 0.5 = 5 MB at int4.*
- **Quantization** — shrinking numbers to fewer bits.
- **PTQ** — quantize *after* training (fast, lossy for tiny models).
- **QAT** — train *with* quantization simulated so the model compensates (near-lossless; what you ship).
- **Pruning** — delete redundant weights.

### Hardware
- **NPU** — the chip's AI accelerator. **eNPU** = ultra-low-power (always-on); **Hexagon** = bigger, used in bursts.
- **QNN** — Qualcomm's toolkit to compile a model onto the NPU.
- **RTFx / latency** — speed (× real-time) and delay.

---

## Part 2 — The techniques that exist today

### A. The four architectures (by "head")

| Head | Streaming? | Cost | Who uses it |
|---|---|---|---|
| **CTC** | trivially | cheapest (a linear layer) | wav2vec2, edge KWS |
| **RNN-T / Transducer** | natively | + predictor + joiner | Google/Apple/Samsung on-device, Zipformer, Parakeet |
| **AED (attention)** | offline | full decoder | Whisper, Moonshine |
| **LLM-decoder** | no | GB-scale | Canary-Qwen, Granite, Cohere (leaderboard tops) |

### B. Encoders (the acoustic model)
- **Conformer** (2020) — conv + attention. The workhorse.
- **Zipformer** (ICLR 2024) — U-Net variable frame rate; **best accuracy-per-param** in open source; standard ops → NPU-deployable.
- **Mamba / SSM** (2023–25) — O(1) recurrent state, battery-friendly, but its core "selective scan" op **has no NPU kernel yet**.

### C. Training paradigms
- **Supervised** — train on (audio, human transcript). Needs lots of labeled data.
- **SSL pretraining** — pretrain on unlabeled audio, then fine-tune. Lets small models do more with less labeled data.
- **Distillation + pseudo-labeling** — a big teacher labels data and supervises a small student. **The dominant way to make a small model accurate.**

### D. Compression
Distillation (makes small viable) · **int8/int4 quantization** (PTQ fast/lossy, QAT slow/near-lossless) · pruning · low-rank factorization · weight sharing.

### E. Decoding
Greedy (fastest) · beam search · LM shallow-fusion · **WFST** (constrain to allowed words / contextual biasing).

### F. State of the art (Aug 2026)

| Model | Params | WER % | On a watch? |
|---|---|---|---|
| IBM Granite Speech 4.1 | 2 B | 5.33 | No |
| Cohere Transcribe | 2 B | 5.42 | No |
| NVIDIA Canary-Qwen | 2.5 B | 5.63 | No |
| Parakeet-TDT-0.6B v3 | 0.6 B | 6.32 | No |
| Whisper-large-v3 (teacher) | 1.55 B | 7.44 | No |
| Nemotron Streaming (int4) | 0.6 B | 8.20 * | ~ (670 MB) |
| Moonshine Tiny (edge) | 27 M | 12.0 * | Yes (27 MB) |
| **WristVoice (our target)** | **~10 M** | **—** | **Yes (≤10 MB)** |

*Source: HF Open ASR Leaderboard, English avg WER (2026); verify current values. `*` = streaming/edge on a different track.*

- **Leaderboard tops** — 2–2.5 B params, GB-scale, offline. Best accuracy, *zero* size/latency discipline. The top is separated by < 1 WER point, and **none fits a watch.**
- **Edge tier** — Nemotron (670 MB), Moonshine (27 MB), Zipformer-small (6–20 MB): smaller & streaming, but each uses only **one or two** levers and none targets the *wrist* domain.
- **The gap** — nobody is genuinely optimizing the **≤10 MB, streaming, on-NPU, wrist** point. That is the opportunity.

---

## Part 3 — Our approach: how we make it better

### 1. Pick a frontier we can win
Not "beat a 2.5 B model on WER" (impossible at 10 MB). Instead:

> **Best streaming WER at ≤ 10 MB, on-device, on a wearable NPU — and best command accuracy + wake latency on real wrist audio.**

The giants physically cannot enter that column. That is the trophy.

### 2. Two specialists — right head per job

**Model 1 — General recognizer (≤ 10 MB): streaming Zipformer transducer**
```
80-mel → conv subsample (100→25 Hz)
      → Zipformer encoder  (U-Net variable frame rate, cache-aware chunked)
              ~13 M params:  layers 2·2·3·4·3·2, dim 128·192·256·320·256·192
      → stateless predictor (embed + kernel-2 conv, ~0.2 M)
      → joiner (dim 256, BPE-500, ~0.5 M)
      + aux CTC head
Loss: pruned RNN-T + 0.2·CTC.   Compress: int4 QAT → ~6 MB (4 MB headroom to spend).
```
Zipformer (not Mamba) because its ops deploy on the Hexagon today; Mamba's selective-scan has no NPU kernel yet (keep it research-only).

**Model 2 — Commands + SLU (≤ 5 MB): Conformer-CTC + FST + intent/slot**
```
80-mel → small streaming Conformer/Zipformer-CTC (~3–4 M)
      → CTC + keyword-FST (open-vocab) + intent/slot head (BiGRU, ~0.1 M)
Two-tier: 0.1 MB always-on wake stub → gates the 5 MB stack.   int4 → ~2 MB.
```
CTC (no transducer decoder) because commands are short: one parallel pass, NPU-friendly, and the **runtime FST** lets you add commands/contacts by text with no retraining.

### 3. Use *all six* levers, not one (the real win)
Most edge models use 1–2 of these. Stacking all six is how you become #1 at the size point:

1. **Distill** from Parakeet-0.6B / Whisper-large — the biggest accuracy lever (20–40 % relative WER).
2. **SSL pretrain** (BEST-RQ) on unlabeled + wrist audio.
3. **int4 QAT + k-quant** — fit the budget near-losslessly (not the lossy PTQ everyone benchmarks).
4. **Multi-latency training** (chunks 160/320/560 ms) — one model, tunable latency.
5. **Wrist-domain data + augmentation** (MUSAN + RIR + real on-device recordings) — win the domain the giants ignore (their WER collapses on far-field wrist audio).
6. **Runtime context-FST** (contacts/apps) — command/entity accuracy up, no retraining.

### 4. The one block that's *ours* — an elastic model

Train **one encoder at two nested widths (Matryoshka)**:
```
        ┌────── full width (~13 M) ──────┐ → dictation (duty-cycled)  = Model 1
encoder │  thin slice (~3 M sub-network) │ → commands (always-on)     = Model 2
        └─────────────────────────────────┘
```
Run the **thin slice always-on** (it *is* the command model, low power) and the **full width on wake** (dictation). You get **10 + 5 from one ~12 MB weight file** — shared, not two separate 15 MB models — with **battery-adaptive quality**.

Why it wins: pure on-device (no external sensors), directly serves 10 + 5 and *beats* it (shared weights < 15 MB), and is genuinely novel for streaming ASR. **Research risk exists**; the safe fallback is two separately-distilled models (still best-in-class via the six levers).

### 5. Why this beats what exists
- vs **giants** — they can't enter ≤10 MB / streaming / on-NPU. Different game.
- vs **edge models** — we combine *all six* levers (they use 1–2), add **wrist-domain data** (they don't), and unify 10 + 5 in **one elastic model** (they ship separate ones).
- **Proof** — a Pareto plot (WER vs MB vs latency) with our point *below* Moonshine/Zipformer-small at equal size, a wrist eval set we dominate, and on-device latency/power on SW6100.

### 6. The honest boundary
The **encoder and losses are standard, cited work — no new core algorithm.** Our contribution is **systems + the elastic-width unification + the domain focus.** That is a real, defensible, ownable claim; pretending we invented a new acoustic model would not survive review.

---

## Part 4 — Proof of concept (built & measured)

All measured on this project (synthetic data + a real Whisper teacher), on a Colab T4 (training) and a MacBook (inference):

| Result | Value | Status |
|---|---|---|
| Whisper teacher, real speech | WER 0.05 | verified |
| Command model (Model 2), held-out | WER 0.20 / CER 0.13 | verified |
| Command model, deployed | 2.55 MB int8 (1.35 MB int4) | verified |
| Command model, in-domain TTS | WER 0.00 | verified |
| Model 1 budget (Mamba) | 5.17 MB int4-mixed | verified |
| Deployable ops (Conformer) | 100 % standard | verified |
| General model on 2 h data | WER 1.0 | data wall (needs scale) |

**Honest finding:** the general model on only 2 h of data learns silence (WER 1.0) — a data-quantity problem, not a bug (the same pipeline hits WER 0 on synthetic; the teacher hits 0.05 on the same audio). General ASR from scratch needs ~100× more data → that is the Phase-2 distillation plan, not a defect.

**Budget proof (Model 1, Mamba encoder, int4-mixed):**

| Component | Params | Precision | Size |
|---|---|---|---|
| Encoder | 8.04 M | int4 + fp16 scales | 4.52 MB |
| Joiner | 0.26 M | int8 | 0.26 MB |
| CTC head | 0.13 M | int8 | 0.13 MB |
| Decoder (embed) | 0.13 M | fp16 | 0.26 MB |
| **Total** | 8.55 M | int4-mixed | **5.17 MB** |

**int8 vs int4:** int8 Conformer = 10.96 MB (over the 10 MB budget); int4 halves the encoder → 6.51 MB. int8 Mamba fits (8.55 MB); int4 gives headroom (5.17 MB).

---

## Part 5 — Roadmap & research-paper path

**Build order (office GPU):**
1. Pilot: distill a 13 M Zipformer-T from Parakeet on LibriSpeech-100 → confirm WER at the size point.
2. Add SSL pretrain + wrist data + augmentation.
3. int4 QAT → verify ≤10 MB with <1 % WER loss.
4. Elastic width (research arm) — nested widths; measure thin-slice commands vs full-width dictation.
5. QNN context binary on SW6100; measure latency + power.

**Research-paper angle (Interspeech / ICASSP):**
*"Personalizable streaming ASR under 10 MB for wearables: a size–latency–accuracy study."* Contribution = the elastic-width unification + the ≤10 MB wrist frontier. Needs: real results on LibriSpeech/Common Voice, honest baselines (Whisper-tiny, Moonshine, Zipformer-small), and ablations (int8 vs int4, distill vs scratch, elastic on/off).

---

## References

**Foundations:** CTC — Graves 2006 · RNN-T — Graves 2012 (1211.3711) · Conformer — Gulati 2020 (2005.08100) · Zipformer — Yao, ICLR 2024 (2310.11230) · Stateless predictor — Ghodsi 2020 · Distillation — Hinton 2015 (1503.02531) · int8 QAT — Jacob 2018 (1712.05877).

**Recent (2024–26):** Samba-ASR (2501.02832) · Mamba streaming (2410.00070) · MoME / Matryoshka (NeurIPS 2025) · Dynamic Encoder Size (2407.18930) · on-device int4 (2604.14493) · HyperSpotter KWS (2508.04857) · KD for transducers (2110.03334) · Whisper-KD streaming (2409.13499) · BEARD / BEST-RQ (2510.24570) · SLURP (2011.13205).

> Foundational references are solid; recent arXiv ids come from a literature scan — verify before external citation.
