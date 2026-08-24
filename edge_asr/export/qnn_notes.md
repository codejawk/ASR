# QNN / HTP export notes (quick reference)

See `docs/DEPLOYMENT_SW6100.md` for the full story. Cheat-sheet:

- **No `Loop`/`If`** in QNN EP → 3-graph export, host-side streaming loop.
- **Static shapes only** → fixed chunk length + fixed cache tensor shapes.
  `export_model1_streaming` freezes these.
- **Quantize on x86_64**, infer on arm64.
- **Precompile a context binary** for the Wear-Elite target; do not reuse
  SM88xx binaries.
- Keep the **decoder embedding fp16** (`quantize.py` skips `decoder.onnx`).
- If swapping to real Zipformer, check **BiasNorm / Swoosh** op coverage.

Sanity command once `onnx` + QNN EP are installed:

```bash
python scripts/export_pipeline.py --ckpt runs/model1/model1.pt --out runs/model1/onnx
# then load runs/model1/onnx/int8/*.onnx with ORT + QNN EP on-device
```
