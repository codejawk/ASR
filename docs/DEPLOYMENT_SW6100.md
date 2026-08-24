# Deployment — Snapdragon Wear Elite / SW6100

## Silicon

Wear Elite has a **dual-NPU** design:
- **Hexagon NPU** (up to ~12 TOPS, 2B-param class) — runs Model 1 in bursts.
- **Ultra-low-power eNPU** in a low-power island — runs persistent ambient
  workloads (KWS, noise suppression, activity). Model 2 lives here.

This is *why* the two-model split is correct rather than arbitrary:

```
[eNPU, always-on, ~mW]   VAD → command/KWS (Model 2)
                                   ├─ direct action (fast path)
                                   └─ "wake" ─┐
                                              ▼
[Hexagon, duty-cycled]             streaming ASR (Model 1) → text
```

## ⚠️ Pin down the budget source first

The chip runs 2B-param models, so a **10 MB cap is not silicon-driven** —
it's coming from resident RAM, APK/OTA size, or a shared Wear OS pool.
**Find out which before committing the architecture:**
- If it's **flash**: ship a larger Model 1 and page it, or make it a
  per-language downloadable asset (see DATA_LICENSING → Multilingual).
- If it's **resident RAM**: the ~11 MB int8 here is the thing to keep
  shrinking (d_model/layers/BPE levers in ARCHITECTURE §3).

## Toolchain: ONNX Runtime + QNN Execution Provider

- Qualcomm now ships a **plugin QNN EP** for ONNX Runtime as a standalone
  package — no rebuild-from-source. Install stock ORT + the QNN EP plugin.
- **QNN EP supports a subset of ONNX ops — no `Loop`/`If`.** So the model is
  a pure single-chunk function and the **streaming loop lives in host code**
  (mirror `runtime/streaming_session.py` in C++/Kotlin). This repo's 3-graph
  export already assumes this.
- **Quantization utilities run on x86_64** — quantize on an x64 box, infer
  with the arm64 package.
- Export with **fully static shapes** (fixed chunk length, fixed cache
  tensor shapes). **Precompile a QNN context binary** to avoid graph-compile
  cost on cold start.
- `sherpa-onnx` has streaming-Zipformer QNN export + an Android demo, but its
  prebuilt context binaries target a **different (larger) Hexagon variant** —
  you must **regenerate context binaries against the Wear-Elite QNN target**.
  Reusing someone else's SM88xx binaries will not work.

## Custom-op watchlist

If/when you swap in real Zipformer, verify these map to supported QNN ops or
decompose cleanly: **BiasNorm**, the **Swoosh** activations, and the custom
normalizations. The Conformer-lite stand-in here uses only standard ops
(LayerNorm, Conv, GLU, SiLU, MatMul, Softmax) to keep the first QNN bring-up
clean.

## Alternative runtime

**ExecuTorch + Qualcomm delegate** (BSD, PyTorch-native) is a viable
fallback if ONNX op coverage bites. Default to ONNX/sherpa-onnx (more proven
ASR mileage); keep ExecuTorch in your back pocket.

## Highest-value first week

1. Confirm whether **10 MB is flash or resident RAM**.
2. Export a **stock streaming-Zipformer** (even 2× over budget) to QNN and
   run it on the **actual SW6100** — de-risk the toolchain before training.
   Toolchain risk on a new SoC > model risk.
3. Run `tools/count_params.py` on your target config; confirm the export
   round-trips through the QNN EP.
4. Only then start the LibriSpeech-960 pilot (2 days, 4 GPUs) to validate
   the recipe, before the full multi-thousand-hour run.

## Expected accuracy at this size

Plan on roughly **6–8% test-clean / 14–18% test-other** on LibriSpeech at
~11–13 M params, and materially worse on real wrist audio. If your product
bar is tighter, the **budget** is what has to move, not the model.
