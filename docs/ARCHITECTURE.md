# Architecture

This is the engineering reference for both models. It condenses the ASR
technology landscape into the choices that are *load-bearing at 10 MB*, and
maps each choice to the code in this repo.

## 0. ASR is a stack, not a model

```
mic → AEC/NS/beamforming → VAD → feature extraction → encoder → decoder/head → text
      (eNPU DSP/neural)          (log-mel, 100 Hz)   (acoustic)  (CTC/RNN-T/AED)
```

What determines whether it fits and streams on a watch is the **head**:

| Head | Streaming? | Size cost | In this repo |
|------|-----------|-----------|--------------|
| CTC | trivially | cheapest (linear only) | aux head + Model 2 phoneme path |
| **RNN-T / Transducer** | natively | +predictor +joiner | **Model 1 (primary)** |
| AED (Whisper/Moonshine) | only with hacks | full decoder | teacher only |
| LLM-decoder | no | GB-scale | teacher only |

"Streaming/causal vs offline" is **architectural, not tunable** — no amount
of tuning turns an offline encoder-decoder into a low-latency streamer.

## 1. Frontend (`edge_asr/features/frontend.py`)

- 80-dim log-mel, 25 ms window / 10 ms hop → **100 frames/s**. This frame
  rate is the compute unit for everything downstream.
- Filterbank built by hand (no torchaudio) so the *exact same* features run
  on the training host and inside the exported ONNX graph.
- **CMVN must be causal + online** for streaming. `OnlineCMVN` keeps
  cumulative statistics with a warmup blend toward global priors; a global
  full-utterance mean/var silently breaks streaming. `test_features.py`
  asserts causality (a late frame cannot change an early normalized frame).

## 2. Model 1 encoder (`edge_asr/models/streaming_encoder.py`)

A **cache-aware streaming Conformer-lite** that captures the three
mechanics that decide on-device viability:

1. **Early subsampling** 100 → 25 Hz (`Conv2dSubsampling`) so self-attention
   runs on 4× fewer frames.
2. **Cache-aware chunked** processing: attention KV context and depthwise-
   conv left context are carried between chunks as explicit state tensors,
   so every frame is encoded exactly once. These state tensors become the
   ONNX/QNN model inputs/outputs — with **static shapes**, which is what HTP
   needs anyway.
3. **Causal** depthwise convs (left-pad only) + bounded left attention
   context (no lookahead beyond the current chunk).

The **encoder is ~85% of the parameter budget**, so encoder shape *is* the
whole size decision. `EncoderConfig` mirrors the knobs you scale in
icefall's Zipformer recipe; the production path swaps this class for
Zipformer (U-Net variable frame rate, BiasNorm, ScaledAdam) with the same
config surface. This stand-in trains and exports so you de-risk the
*toolchain* before committing GPU-weeks.

### Streaming geometry (config)
- `chunk_frames: 32` → 320 ms acoustic chunk.
- `subsampling_factor: 4` → 8 output frames per chunk at 25 Hz.
- `left_context_chunks: 4` → attention memory horizon.
- Multi-latency training (production): sample chunk sizes `"16,32,64"` so one
  checkpoint serves several latency operating points, selectable post-ship.

## 3. Model 1 heads

- **Stateless predictor** (`decoder.py`): embedding + kernel-2 depthwise
  conv over the last 2 tokens. ~0.13 M params, state = "previous token".
- **Joiner** (`joiner.py`): `enc_proj + pred_proj → tanh → vocab`. The vocab
  projection is the tax paid twice (here + embedding), so `joiner_dim=256`
  and **BPE-500** keep it ~0.25 M.
- **Aux CTC head** (`transducer.py`): ~0.11 M, weight 0.2. Speeds
  convergence and gives a cheap non-autoregressive fallback decode.

### Budget (verified by `tools/count_params.py`)
```
encoder   10.47 M   ← 85%
decoder    0.13 M
joiner     0.25 M
ctc_head   0.11 M
total     10.96 M   → ~11 MB int8, under the 13 MB working target
```
Levers to reach <10 MB: `d_model 224→208`, `n_layers 11→10`, or BPE 500→400.

## 3.1 RNN-T loss (`edge_asr/losses/rnnt.py`)

Uses `torchaudio.functional.rnnt_loss` when installed (CUDA kernels);
otherwise a **numerically-stable pure-PyTorch** log-domain forward so the
project runs with only `torch`. Lattice:
```
a[0,0]=0;  a[t,u]=logaddexp(a[t-1,u]+lp_blank[t-1,u], a[t,u-1]+lp_label[t,u-1])
loss = -(a[T-1,U] + lp_blank[T-1,U])
```
For multi-thousand-hour training, install torchaudio (or warp-rnnt).

## 3.2 Training strategy (`training/train_model1.py`)

- **Pseudo-label at scale**: run a large teacher (Parakeet-TDT-0.6B-v3 or
  Nemotron streaming 0.6B) over unlabeled audio (`scripts/02_pseudo_label.sh`).
  Teacher quality dominates student WER.
- **Distill with hard targets**: the manifest text *is* the teacher's
  hypothesis. Hard targets beat soft when teacher/student architectures
  differ (large teacher, small streaming student).
- **Augment for the wrist**: `data/augment.py` (noise + SpecAugment; add
  MUSAN + RIR dirs). Real on-device mic recordings move WER most.

## 3.4 Quantization

PTQ (`export/quantize.py`, ORT dynamic int8) is for de-risking size/latency
and telling you the real on-disk number. **Ship with int8 QAT** (W8A8,
per-channel weights) — at ~11 M params there is no redundancy to absorb PTQ
error. Keep the predictor embedding fp16 (quantizing it saves ~0.2 MB and
costs accuracy — `quantize.py` skips `decoder.onnx` by default).

## 3.5 Export (`export/export_onnx.py`)

Three static-shape graphs — because QNN's EP has no `Loop`/`If`:
```
encoder.onnx : (feats_chunk, *state_in) → (enc_chunk, *state_out)
decoder.onnx : (prev_tokens)            → (pred_vec)
joiner.onnx  : (enc_t, pred_vec)        → (logits)
```
The streaming loop lives in host code (`runtime/streaming_session.py`).

## 4. Model 2 — command recognition

Pick by one question: **can the command set change after ship?**

| Approach | Params | Open-vocab? | Code |
|----------|--------|-------------|------|
| Closed-set BC-ResNet | 0.1–0.4 M | no | `models/bcresnet.py` |
| Phoneme-CTC + keyword FST | 1–2 M | **yes** | `decode/keyword_ctc.py` |
| Transducer KWS | ~3 M | yes | reuse Model 1 stack, small |

Recommendation: **phoneme-CTC + FST (~1.5 MB)** — add a command by editing a
text file, no retraining, and it sits always-on on the eNPU. The metric is
**false-accepts/hour at fixed false-reject rate**, not accuracy — tune the
threshold on real ambient audio (`KeywordSpotter.score_false_accepts`).

## 5. Runtime cascade (`runtime/cascade.py`)

```
[eNPU, always-on]  VAD → command/KWS (Model 2) ─┬─ direct action (fast path)
                                                 └─ "wake" → [Hexagon] Model 1 → text
```
This three-stage split is what makes two models *correct* rather than
arbitrary on Wear Elite's dual-NPU (Hexagon up to 12 TOPS + ultra-low-power
eNPU for persistent KWS).

## 3.6 Evaluation (`eval/`)

Do **not** ship against LibriSpeech. Build a wrist eval set (positions,
noise, launch-market accents, your command grammar + dictation, long-tail
names/numbers). Report **WER + task-success + latency p50/p95 (not means) +
battery/active-minute**. A model that wins WER and loses p95 loses the
product review.
