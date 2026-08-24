"""Print the parameter breakdown and estimated int8 size for a Model 1
config. Run this BEFORE any long training run.

    python -m edge_asr.tools.count_params configs/model1_general.yaml [vocab_size]
"""
from __future__ import annotations

import sys

from ..training.utils import build_model1, load_config


def main():
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else "configs/model1_general.yaml"
    vocab = int(sys.argv[2]) if len(sys.argv) > 2 else 500
    cfg = load_config(cfg_path)
    model = build_model1(cfg, vocab)
    counts = model.num_params()
    print(f"config: {cfg_path}   vocab_size: {vocab}")
    for k, v in counts.items():
        print(f"  {k:10s} {v/1e6:7.3f} M")
    total = counts["total"]
    print(f"\n  fp16 estimate     : ~{2*total/1e6:.2f} MB")
    print(f"  int8 estimate     : ~{total/1e6:.2f} MB (weights) + tokenizer/graph overhead")

    # int4 mixed-precision projection (the Microsoft on-device recipe):
    #   encoder int4 (+ block scales), joiner int8, decoder fp16, ctc int8.
    block = 32
    enc = counts["encoder"]
    enc_int4 = enc * 0.5 + (enc / block) * 2      # 4-bit weights + fp16 block scales
    joiner_int8 = counts["joiner"] * 1.0
    ctc_int8 = counts["ctc_head"] * 1.0
    dec_fp16 = counts["decoder"] * 2.0
    mixed = (enc_int4 + joiner_int8 + ctc_int8 + dec_fp16) / 1e6
    print(f"  int4-mixed (QAT)  : ~{mixed:.2f} MB  [encoder int4 / joiner+ctc int8 / decoder fp16]")

    budget = 10.0
    print(f"\n  vs {budget:.0f} MB ship budget:")
    print(f"    int8        {'OK ' if total/1e6 <= budget else 'OVER'} ({total/1e6:.2f} MB)")
    print(f"    int4-mixed  {'OK ' if mixed <= budget else 'OVER'} ({mixed:.2f} MB)  <- target")


if __name__ == "__main__":
    main()
