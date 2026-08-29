# Running Edge-ASR — MacBook (testing) & Colab GPU (training)

**TL;DR:** For *testing*, your MacBook (CPU) is all you need — the whole
project runs on CPU with no downloads. You only need a GPU (Colab) for *real
training on real speech data*, which is optional and comes later.

---

## A. Test on your MacBook (CPU) — start here

### 1. Get the code and a clean Python env
```bash
git clone https://github.com/codejawk/ASR.git
cd ASR
python3 -m venv .venv && source .venv/bin/activate     # Python 3.10–3.13
pip install -r requirements.txt
```

### 2. Verify everything works (~1–2 min)
```bash
bash scripts/smoke_test.sh
```
Expect `ALL GREEN`: unit tests, param budgets, an end-to-end train+decode,
and the v2/v3 component tests.

### 3. Try the individual pieces
```bash
# parameter budgets (fp16 / int8 / int4-mixed) — no training
python -m edge_asr.tools.count_params configs/model1_mamba.yaml 500

# train Model 1 (general ASR) on synthetic data — ~3–4 min CPU
python -m edge_asr.training.train_model1 --config configs/model1_general.yaml \
    --steps 250 --out runs/model1
# -> loss 95 -> ~1, decode 'send a message' WER 0.00

# command model (open-vocab KWS + router)
python -m edge_asr.training.train_command --config configs/model2_command.yaml \
    --steps 300 --out runs/command

# on-device SLU (intent + slot) — trains + demos parsing
python -m edge_asr.training.train_slu --steps 500 --out runs/slu

# flash-paged Mixture-of-Specialists demo
python scripts/moe_demo.py

# distillation (teacher -> student), needs a teacher checkpoint first
python -m edge_asr.training.train_distill --teacher runs/model1/model1.pt \
    --student-config configs/model1_mamba.yaml --steps 400 --out runs/student
```

### 4. Export + quantize (needs onnx)
```bash
pip install onnx onnxscript
python scripts/export_pipeline.py --ckpt runs/model1/model1.pt \
    --out runs/model1/onnx --mode mixed
```

### Notes for macOS
- **CPU is the right choice for these synthetic runs** — they're small and
  finish in minutes. No GPU needed.
- Apple-Silicon **MPS** is possible (`--device mps` on `train_model1` /
  `train_distill`) but not recommended here: the pure-PyTorch RNN-T fallback
  is a slow fit for MPS, and the synthetic runs don't need it.
- `torchaudio` is optional; without it a pure-PyTorch RNN-T loss is used
  automatically.

---

## B. Scale up on Google Colab (GPU) — for real training

Use Colab **only when you move to real speech data** (Loquacious, Common
Voice) and a real teacher (Whisper / Parakeet). The training scripts take
`--device auto` (picks the GPU) and auto-use torchaudio's fast CUDA RNN-T
kernel when installed.

### 1. New Colab notebook → Runtime → Change runtime type → **GPU (T4)**

### 2. Setup cell
```python
!git clone https://github.com/codejawk/ASR.git
%cd ASR
# IMPORTANT: Colab already has a CUDA build of torch — do NOT reinstall it.
# Install only the other deps so you keep the GPU torch:
!pip install -q numpy pyyaml sentencepiece onnx onnxruntime torchaudio
import torch; print("CUDA:", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
```

### 3. Verify on the GPU box
```python
!PYTHONPATH=. python tests/test_v3_distill_slu.py
!PYTHONPATH=. python -m edge_asr.tools.count_params configs/model1_mamba.yaml 500
```

### 4. Train on GPU (synthetic first, to confirm the GPU path)
```python
!PYTHONPATH=. python -m edge_asr.training.train_model1 \
    --config configs/model1_general.yaml --steps 500 \
    --device auto --out runs/model1
```
You should see `device: cuda` in the `[params]` line.

### 5. Real training (the actual reason to use a GPU)
```python
# 1) build a manifest of your real audio (JSONL: {"audio","text","duration"})
#    - clear licensing first (see docs/DATA_LICENSING.md); use Loquacious.
# 2) pseudo-label unlabeled audio with a large teacher (scripts/02_pseudo_label.sh)
# 3) distill on GPU:
!PYTHONPATH=. python -m edge_asr.training.train_distill \
    --teacher runs/teacher/model1.pt --student-config configs/model1_mamba.yaml \
    --manifest data/train.jsonl --pseudo-label --steps 4000 \
    --device auto --out runs/student
```

### Colab gotchas
- **Don't `pip install -r requirements.txt` on Colab** — it can replace the
  CUDA torch with a CPU build. Install the *other* deps only (step 2).
- Colab sessions are ephemeral — save checkpoints to Google Drive:
  ```python
  from google.colab import drive; drive.mount('/content/drive')
  # then pass --out /content/drive/MyDrive/asr_runs/...
  ```
- Free T4 is enough for the pilot; a LibriSpeech-960 sanity run is ~2 days on
  4 GPUs (see docs/ROADMAP.md). Don't attempt the full multi-thousand-hour
  run on free Colab — that needs a real GPU box.

---

## What needs a GPU vs. what doesn't

| Task | Where |
|------|-------|
| Smoke tests, param budgets, MoE demo, SLU, command model | **MacBook CPU** |
| Synthetic Model 1 training, export/quantize | **MacBook CPU** |
| Real distillation from a large teacher on real data | **GPU (Colab)** |
| Full multi-thousand-hour training + int4 QAT | **Real GPU box** (not free Colab) |
