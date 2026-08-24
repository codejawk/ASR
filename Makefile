.PHONY: help setup test smoke count count-mamba train1 train1-mamba train2 command moe export export-mixed clean
PY ?= PYTHONPATH=. python

help:
	@echo "make setup        - pip install -r requirements.txt"
	@echo "make test         - run all tests (features, smoke, v2 components)"
	@echo "make smoke        - full synthetic pipeline (train tiny model1 + decode)"
	@echo "make count        - Model 1 (conformer) budget: fp16/int8/int4-mixed"
	@echo "make count-mamba  - Model 1 (mamba) budget"
	@echo "make train1       - train Model 1 conformer on synthetic (250 steps)"
	@echo "make train1-mamba - train Model 1 mamba encoder on synthetic"
	@echo "make command      - train the hypernetwork command model + router"
	@echo "make slu          - train + demo the intent/slot SLU (Model 2, used wisely)"
	@echo "make distill      - distill a small student from a teacher checkpoint"
	@echo "make moe          - run the flash-paged Mixture-of-Specialists demo"
	@echo "make export       - export + int8-quantize Model 1 (needs onnx)"
	@echo "make export-mixed - export + int4-mixed-quantize Model 1"

setup:
	pip install -r requirements.txt

test:
	$(PY) tests/test_features.py
	$(PY) tests/test_smoke_end_to_end.py
	$(PY) tests/test_v2_components.py
	$(PY) tests/test_v3_distill_slu.py

smoke:
	$(PY) tests/test_smoke_end_to_end.py

count:
	$(PY) -m edge_asr.tools.count_params configs/model1_general.yaml 500

count-mamba:
	$(PY) -m edge_asr.tools.count_params configs/model1_mamba.yaml 500

train1:
	$(PY) -m edge_asr.training.train_model1 --config configs/model1_general.yaml --steps 250 --out runs/model1

train1-mamba:
	$(PY) -m edge_asr.training.train_model1 --config configs/model1_mamba.yaml --steps 250 --out runs/model1_mamba

train2:
	$(PY) -m edge_asr.training.train_model2 --config configs/model2_command.yaml --steps 250 --out runs/model2

command:
	$(PY) -m edge_asr.training.train_command --config configs/model2_command.yaml --steps 300 --out runs/command

slu:
	$(PY) -m edge_asr.training.train_slu --steps 500 --out runs/slu

distill:
	$(PY) -m edge_asr.training.train_distill --teacher runs/model1/model1.pt --student-config configs/model1_mamba.yaml --steps 800 --out runs/student

moe:
	$(PY) scripts/moe_demo.py

export:
	$(PY) scripts/export_pipeline.py --ckpt runs/model1/model1.pt --out runs/model1/onnx

export-mixed:
	$(PY) scripts/export_pipeline.py --ckpt runs/model1/model1.pt --out runs/model1/onnx --mode mixed

clean:
	rm -rf runs __pycache__ */__pycache__ */*/__pycache__ .pytest_cache
