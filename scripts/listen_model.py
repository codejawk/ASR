"""Live mic demo for YOUR trained model (the ~2.5 MB Model-2 command recognizer).

Records from the mic (or reads a WAV), runs your student model on-device, and
prints what it heard. Runs on CPU — the model is tiny.

    pip install sounddevice                 # for the mic
    python scripts/listen_model.py --ckpt runs/cmd/model1.pt            # press Enter to start/stop
    python scripts/listen_model.py --ckpt runs/cmd/model1.pt --loop     # keep listening
    python scripts/listen_model.py --ckpt runs/cmd/model1.pt --wav clip.wav   # transcribe a file

NOTE: this model was trained on synthetic TTS voices, so real-voice accuracy
will be lower than on TTS clips. Try `--wav` on a gTTS clip to see it at its
best, then try the mic. Say one of the trained commands clearly, e.g.
"set a timer for ten minutes", "call mom", "what is the weather in london".
"""
from __future__ import annotations

import argparse
import os
import sys
import wave

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from edge_asr.data import load_tokenizer
from edge_asr.data.datasets import load_wav
from edge_asr.decode import greedy_search
from edge_asr.features import LogMelFrontend, OnlineCMVN
from edge_asr.training.utils import build_model1

SR = 16000


def record_until_enter():
    import numpy as np
    import sounddevice as sd

    frames = []
    input("  ▶  Press Enter to START, speak, then press Enter to STOP...")
    stream = sd.InputStream(samplerate=SR, channels=1, dtype="int16",
                            callback=lambda indata, n, t, s: frames.append(indata.copy()))
    stream.start()
    print("  ●  Recording... speak now, Enter to STOP.")
    try:
        input()
    finally:
        stream.stop(); stream.close()
    if not frames:
        return None
    return np.concatenate(frames).reshape(-1).astype("float32") / 32768.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--wav", default=None, help="transcribe a WAV file instead of the mic")
    ap.add_argument("--loop", action="store_true")
    args = ap.parse_args()

    blob = torch.load(args.ckpt, weights_only=False, map_location="cpu")
    tok = load_tokenizer(blob.get("tokenizer", "char"))
    model = build_model1(blob["config"], blob["vocab_size"])
    model.load_state_dict(blob["model"]); model.eval()
    n_mels = blob["config"]["encoder"]["input_dim"]
    fe = LogMelFrontend(n_mels=n_mels); cmvn = OnlineCMVN(n_mels=n_mels)
    print(f"[model] {sum(p.numel() for p in model.parameters())/1e6:.2f} M params, ready.")

    def transcribe(wav: torch.Tensor) -> str:
        feats = fe(wav).squeeze(0)
        feats = cmvn(feats.unsqueeze(0)).squeeze(0)
        return tok.decode(greedy_search(model, feats))

    if args.wav:
        wav = load_wav(args.wav, SR)
        print(f"  \U0001f5e3  heard:  {transcribe(wav)!r}")
        return

    try:
        import sounddevice  # noqa
    except ImportError:
        sys.exit("Install the mic library: pip install sounddevice")

    print("[ready] mic demo. Ctrl+C to quit.\n")
    try:
        while True:
            wav = record_until_enter()
            if wav is None or len(wav) < SR // 4:
                print("  (nothing captured)\n")
                if not args.loop:
                    break
                continue
            print("  ... recognizing ...")
            print(f"\n  \U0001f5e3  Your model heard:  {transcribe(torch.tensor(wav))!r}\n")
            if not args.loop:
                break
    except KeyboardInterrupt:
        print("\n[bye]")


if __name__ == "__main__":
    main()
