"""Export a trained Model 1 checkpoint to ONNX and int8-quantize it, then
report the real on-disk size against the 10 MB budget.

    python scripts/export_pipeline.py --ckpt runs/model1/model1.pt --out runs/model1/onnx
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from edge_asr.data import load_tokenizer
from edge_asr.export.export_onnx import export_model1_streaming
from edge_asr.export.quantize import dynamic_quantize, mixed_precision_quantize, report_sizes
from edge_asr.training.utils import build_model1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default="runs/model1/onnx")
    ap.add_argument("--mode", choices=["dynamic", "mixed"], default="dynamic",
                    help="dynamic = int8 PTQ (size ceiling); mixed = int4 encoder + int8 joiner + fp decoder")
    args = ap.parse_args()

    blob = torch.load(args.ckpt, weights_only=False)
    model = build_model1(blob["config"], blob["vocab_size"])
    model.load_state_dict(blob["model"])
    model.eval()

    tok = load_tokenizer(blob.get("tokenizer", "char"))
    tokens_path = os.path.join(args.out, "tokens.txt")
    os.makedirs(args.out, exist_ok=True)
    if hasattr(tok, "save"):
        tok.save(tokens_path)

    print("[export] writing fp32 ONNX graphs...")
    fp_paths = export_model1_streaming(model, args.out)
    print("[export] fp32 sizes:")
    report_sizes(fp_paths)

    if args.mode == "mixed":
        print("\n[quant] MIXED int4/int8 (encoder int4, joiner int8, decoder fp):")
        q_dir = os.path.join(args.out, "mixed")
        q_paths = mixed_precision_quantize(fp_paths, q_dir)
    else:
        print("\n[quant] int8 (encoder+joiner; decoder kept fp):")
        q_dir = os.path.join(args.out, "int8")
        q_paths = dynamic_quantize(fp_paths, q_dir)
    print("\n[quant] FINAL model on disk:")
    total = report_sizes(q_paths)
    if os.path.exists(tokens_path):
        tsz = os.path.getsize(tokens_path) / 1e6
        print(f"  {'tokens.txt':24s} {tsz:6.2f} MB")
        total += tsz
    budget = 10.0
    params = build_model1(blob["config"], blob["vocab_size"]).num_params()["total"] / 1e6
    print(f"\n  vs {budget:.0f} MB budget: {'OK' if total <= budget else 'OVER'} ({total:.2f} MB)")
    print(
        "\n  NOTE: this is *dynamic* PTQ — a size CEILING. It weight-quantizes\n"
        f"  MatMul/Gemm only, leaving norms/convs/biases fp32. The parameter\n"
        f"  FLOOR is ~{params:.1f} MB (int8 params). Reach it with static int8\n"
        "  or QAT (docs/ARCHITECTURE.md 3.4), which is what you ship."
    )


if __name__ == "__main__":
    main()
