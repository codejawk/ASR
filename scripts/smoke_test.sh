#!/usr/bin/env bash
# One-shot verification that the whole toolchain works, no downloads.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=.

echo "== 1. unit tests =="
if python -c "import pytest" 2>/dev/null; then
  python -m pytest -q tests/test_features.py
else
  echo "(pytest not installed; running test file directly)"
  python tests/test_features.py
fi

echo "== 2. param budget =="
python -m edge_asr.tools.count_params configs/model1_general.yaml 500

echo "== 3. end-to-end smoke (train tiny model + decode) =="
python tests/test_smoke_end_to_end.py

echo "== 4. command model forward =="
python -c "from edge_asr.models import BCResNet,BCResNetConfig; import torch; \
m=BCResNet(BCResNetConfig()); print('BCResNet out', tuple(m(torch.randn(2,50,40)).shape), \
m.num_params(),'params')"

echo "== 5. v2 components (mamba encoder / hypernet KWS / flash-paged MoE) =="
python tests/test_v2_components.py

echo "== 6. v3 (distillation CTC-KD + intent/slot SLU) =="
python tests/test_v3_distill_slu.py

echo "== 7. int4-mixed budget (mamba Model 1) =="
python -m edge_asr.tools.count_params configs/model1_mamba.yaml 500 | tail -4

echo "ALL GREEN"
