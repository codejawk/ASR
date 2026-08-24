"""Edge-ASR: two small streaming ASR models for wrist-class hardware.

Model 1  — general streaming ASR   (Zipformer/Conformer-lite transducer, ~10-13 MB int8)
Model 2  — command / keyword model (BC-ResNet or phoneme-CTC, ~1.5 MB int8)

The package is intentionally dependency-light so the *whole* toolchain
(train -> decode -> export -> quantize) runs on a laptop with only
`torch`, `numpy`, `onnxruntime` and `sentencepiece`. `torchaudio` and the
QNN Execution Provider are used automatically when present but are not
required to exercise the pipeline end-to-end.
"""

__version__ = "0.1.0"
