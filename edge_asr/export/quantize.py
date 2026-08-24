"""Post-training int8 quantization via ONNX Runtime.

This is the *PTQ* path — fast, good for de-risking size/latency. For the
final shippable Model 1 at ~13 M params, PTQ leaves accuracy on the table;
do int8 **QAT** in PyTorch instead (docs/ARCHITECTURE.md 3.4). This script
still matters: it tells you the real on-disk size after quantization, which
is the number your 10 MB budget is measured against.

Notes baked in from the deployment research:
  * ORT quantization utilities run on x86_64 — quantize on an x64 box,
    infer with the arm64 package.
  * Keep the transducer decoder embedding in fp16/fp32; quantizing it
    saves ~200 KB and costs real accuracy. We therefore quantize encoder +
    joiner and leave decoder.onnx unquantized by default.
"""
from __future__ import annotations

import os
from typing import List, Optional


def dynamic_quantize(model_paths: List[str], out_dir: str, skip: Optional[List[str]] = None) -> List[str]:
    from onnxruntime.quantization import QuantType, quantize_dynamic

    skip = skip or ["decoder.onnx"]
    os.makedirs(out_dir, exist_ok=True)
    outputs = []
    for p in model_paths:
        base = os.path.basename(p)
        out = os.path.join(out_dir, base.replace(".onnx", ".int8.onnx"))
        if base in skip:
            # copy through unquantized
            import shutil

            shutil.copy(p, os.path.join(out_dir, base))
            outputs.append(os.path.join(out_dir, base))
            print(f"[quant] kept fp {base}")
            continue
        quantize_dynamic(p, out, weight_type=QuantType.QInt8)
        outputs.append(out)
        print(f"[quant] int8 {base} -> {os.path.getsize(out)/1e6:.2f} MB")
    return outputs


def mixed_precision_quantize(model_paths: List[str], out_dir: str,
                             int4_targets=("encoder.onnx",),
                             int8_targets=("joiner.onnx",),
                             keep_fp=("decoder.onnx",),
                             block_size: int = 32) -> List[str]:
    """Mixed int4/int8 quantization following the on-device recipe from
    Microsoft's 2026 study (arXiv:2604.14493):

      * **encoder int4** — it is ~85-95% of params, so this is where the MB
        live; int4 block-wise (k-quant-style) weights.
      * **joiner int8** — small, accuracy-sensitive projection.
      * **decoder FP32** — the predictor embedding; quantizing it saves ~KBs
        and costs accuracy.

    Falls back to int8 for any graph if the int4 quantizer is unavailable in
    the installed onnxruntime.
    """
    import os
    import shutil

    os.makedirs(out_dir, exist_ok=True)
    outputs = []
    for p in model_paths:
        base = os.path.basename(p)
        if base in keep_fp:
            dst = os.path.join(out_dir, base)
            shutil.copy(p, dst); outputs.append(dst)
            print(f"[mixed] fp32  {base}")
        elif base in int4_targets:
            from onnxruntime.quantization import QuantType, quantize_dynamic
            i8 = os.path.join(out_dir, base.replace(".onnx", ".int8.onnx"))
            quantize_dynamic(p, i8, weight_type=QuantType.QInt8)
            i4 = os.path.join(out_dir, base.replace(".onnx", ".int4.onnx"))
            ok = _try_int4(p, i4, block_size)
            # keep int4 only if it is actually smaller than int8 (version-safe)
            if ok and os.path.getsize(i4) < os.path.getsize(i8):
                os.remove(i8)
                print(f"[mixed] int4  {base} -> {os.path.getsize(i4)/1e6:.2f} MB")
                outputs.append(i4)
            else:
                if ok and os.path.exists(i4):
                    os.remove(i4)
                print(f"[mixed] int8  {base} (int4 not smaller here; see count_params for int4 target) "
                      f"-> {os.path.getsize(i8)/1e6:.2f} MB")
                outputs.append(i8)
        else:  # int8
            from onnxruntime.quantization import QuantType, quantize_dynamic
            dst = os.path.join(out_dir, base.replace(".onnx", ".int8.onnx"))
            quantize_dynamic(p, dst, weight_type=QuantType.QInt8)
            outputs.append(dst)
            print(f"[mixed] int8  {base} -> {os.path.getsize(dst)/1e6:.2f} MB")
    return outputs


def _try_int4(src: str, dst: str, block_size: int) -> bool:
    """int4 block-wise weight-only quantization via ORT's MatMulNBits
    quantizer. Prefers **k-quant** (the Microsoft on-device recipe), then
    RTN, then the default config. Returns True on success."""
    try:
        import onnx
        from onnxruntime.quantization import matmul_nbits_quantizer as N

        model = onnx.load(src)
        algo = None
        for name, kwargs in [
            ("KQuantWeightOnlyQuantConfig", {}),
            ("RTNWeightOnlyQuantConfig", {}),
            ("DefaultWeightOnlyQuantConfig", {"block_size": block_size, "bits": 4}),
        ]:
            cfg_cls = getattr(N, name, None)
            if cfg_cls is None:
                continue
            try:
                algo = cfg_cls(**kwargs)
                break
            except TypeError:
                try:
                    algo = cfg_cls()
                    break
                except Exception:
                    continue
        if algo is None:
            return False

        try:
            q = N.MatMulNBitsQuantizer(model, algo_config=algo)
        except TypeError:
            q = N.MatMulNBitsQuantizer(model, block_size=block_size, is_symmetric=True, bits=4)
        q.process()
        q.model.save_model_to_file(dst, use_external_data_format=False)
        return os.path.exists(dst)
    except Exception as e:  # pragma: no cover
        print(f"[mixed] int4 path unavailable: {type(e).__name__}: {e}")
        return False


def report_sizes(paths: List[str]) -> float:
    total = 0.0
    for p in paths:
        mb = os.path.getsize(p) / 1e6
        total += mb
        print(f"  {os.path.basename(p):24s} {mb:6.2f} MB")
    print(f"  {'TOTAL':24s} {total:6.2f} MB")
    return total
