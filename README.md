# Edge-ASR — two small streaming ASR models for a watch

> **Competition build (v2):** a **router-gated, flash-paged Mixture-of-
> Specialists** with an **SSM/Mamba streaming encoder**, an **open-vocab
> hypernetwork** command model, and **int4 mixed-precision**. See
> [docs/COMPETITION_PITCH.md](docs/COMPETITION_PITCH.md). Everything below is
> measured on a laptop, no downloads:
> `python scripts/moe_demo.py` · `python tests/test_v2_components.py`

A complete, runnable project for building the two on-device speech models
you specified, targeted at **Snapdragon Wear Elite / SW6100**:

| Model | Job | Budget | This repo lands at |
|-------|-----|--------|--------------------|
| **Model 1** | General streaming ASR (dictation + free speech) | ≤ 10 MB | **~11 MB int8** encoder-dominated transducer (tunable under 10) |
| **Model 2** | Command / keyword recognition | ≤ 5 MB | **< 0.3 MB int8** BC-ResNet (huge headroom) |

The whole toolchain — **train → decode → export → quantize** — runs on a
laptop with only `torch`, `numpy`, `onnxruntime`, `sentencepiece`. No
dataset downloads are required to verify it: a synthetic speech-like data
generator lets the end-to-end smoke test actually *train a model to WER 0*
so you know the pipeline is correct before spending GPU-weeks.

> **Read `docs/` before real training.** The single most important
> practical finding: **GigaSpeech is not commercially usable** — use
> **Loquacious** (25k h, commercial-OK) instead. See `docs/DATA_LICENSING.md`.

## Quick start

```bash
cd edge-asr
pip install -r requirements.txt          # or: make setup
bash scripts/smoke_test.sh               # verifies the whole pipeline, no downloads
```

Expected: unit tests pass, the param-budget check prints ~11 MB, and the
smoke test trains a tiny model and prints a decoded hypothesis.

### Train Model 1 on synthetic data (sanity run, ~3–4 min CPU)

```bash
make train1
# ... loss 95 -> ~1.2, [decode] ref='send a message' hyp='send a message' wer=0.00
```

### Train Model 2 (command model)

```bash
make train2
```

### Export + int8 quantize Model 1 (needs `onnx`)

```bash
pip install onnx
python scripts/export_pipeline.py --ckpt runs/model1/model1.pt --out runs/model1/onnx
# prints the real on-disk int8 size vs the 10 MB budget
```

## What's in here

```
edge_asr/
  features/       log-mel frontend + causal online CMVN (streaming-safe)
  models/         encoders: streaming Conformer-lite (Zipformer-mapped) AND
                  streaming Mamba/SSM (O(1) state); stateless predictor,
                  joiner, transducer; BC-ResNet + CommandModel (hypernet KWS
                  + router + speaker) for Model 2
  moe/            ExpertPager (flash→RAM, LRU) + MixtureOfSpecialists (router→page→decode)
  losses/         RNN-T loss (torchaudio kernel if present, else pure-torch)
  data/           tokenizers, wrist-domain augmentation, datasets, synthetic gen
  decode/         greedy + streaming transducer decode, CTC decode, keyword FST
  training/       train_model1.py (conformer|mamba), train_model2.py,
                  train_command.py; configure_optimizer (Mamba-safe weight decay)
  export/         ONNX export (3-graph, static shapes) + int8 + int4-mixed quant
  eval/           WER/CER, FA/hour, evaluate.py
  runtime/        StreamingASRSession (host loop) + CascadePipeline (eNPU→Hexagon)
  tools/          count_params.py — fp16/int8/int4-mixed budget (run BEFORE training)
configs/          model1_general.yaml (conformer), model1_mamba.yaml, model2_command.yaml
scripts/          smoke_test.sh, moe_demo.py, export_pipeline.py, data/pseudo-label stubs
tests/            end-to-end smoke, feature tests, v2 component tests
docs/             COMPETITION_PITCH, TECHNICAL_REPORT, ARCHITECTURE,
                  DATA_LICENSING, DEPLOYMENT_SW6100, ROADMAP
```

### Competition v2 quickstart

```bash
python tests/test_v2_components.py                          # mamba + hypernet + paging
python scripts/moe_demo.py                                  # routed, flash-paged experts
python -m edge_asr.tools.count_params configs/model1_mamba.yaml 500   # 5.17 MB int4-mixed
python -m edge_asr.training.train_command --config configs/model2_command.yaml --steps 250 --out runs/command
```

## The design in one paragraph

Model 1 is a **cache-aware streaming transducer**: log-mel → conv
subsampling (100 → 25 Hz) → causal Conformer-lite encoder → stateless
predictor + small joiner, trained with **pruned-ish RNN-T + auxiliary CTC**.
It exports as three static-shape ONNX graphs (encoder-chunk / decoder-step /
joiner-step) because the QNN Execution Provider has no `Loop`/`If` — the
streaming loop lives in host code (`runtime/streaming_session.py`). Model 2
is a tiny **BC-ResNet** command classifier that sits always-on on the eNPU
and *gates* the expensive Hexagon ASR path (`runtime/cascade.py`).

The production encoder is icefall's **Zipformer**; the config knobs here
mirror that recipe so the swap is mechanical. See `docs/ARCHITECTURE.md`.

## Status of each piece

- ✅ Runs and trains today (verified): features, encoder/decoder/joiner,
  RNN-T loss, CTC aux, both training loops, greedy + streaming decode,
  streaming host session, cascade, WER/FA metrics, param budgeting.
- ✅ ONNX export code written; runs when `onnx` is installed.
- 🔜 You provide: real corpora (post-licensing), a teacher for
  pseudo-labelling, QAT, and the SW6100 QNN context-binary regeneration.
  These are documented step-by-step in `docs/` and stubbed in `scripts/`.
```
