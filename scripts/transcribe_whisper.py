"""Phase 0 demo — transcribe a real audio file with Whisper on your Mac.

This is the "feel it work" step: a real pretrained model recognizing real
speech on CPU. The same Whisper here becomes the *teacher* for distillation
in Phase 1+ (scripts/pseudo_label_whisper.py).

Install (pick one, faster-whisper is fastest on Mac CPU):
    pip install faster-whisper          # recommended (CTranslate2, int8 CPU)
    # or: pip install openai-whisper

Run:
    python scripts/transcribe_whisper.py path/to/audio.wav --model base

Models (English-only variants add .en and are smaller/faster):
    tiny(.en) ~39M · base(.en) ~74M · small(.en) ~244M · medium · large-v3
Start with base or small on a Mac; large-v3 is for a GPU box.
"""
from __future__ import annotations

import argparse
import sys


def with_faster_whisper(audio, model_name, device, compute_type):
    from faster_whisper import WhisperModel
    model = WhisperModel(model_name, device=device, compute_type=compute_type)
    segments, info = model.transcribe(audio, beam_size=5)
    text = "".join(seg.text for seg in segments)
    return text.strip(), info.language


def with_openai_whisper(audio, model_name):
    import whisper
    model = whisper.load_model(model_name)
    result = model.transcribe(audio)
    return result["text"].strip(), result.get("language")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("audio", help="path to a .wav/.mp3/.m4a file")
    ap.add_argument("--model", default="base", help="tiny|base|small|medium|large-v3 (+.en)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--compute-type", default="int8", help="faster-whisper: int8|int8_float16|float32")
    args = ap.parse_args()

    try:
        text, lang = with_faster_whisper(args.audio, args.model, args.device, args.compute_type)
        backend = "faster-whisper"
    except ImportError:
        try:
            text, lang = with_openai_whisper(args.audio, args.model)
            backend = "openai-whisper"
        except ImportError:
            sys.exit("No Whisper backend found. Install one:\n"
                     "    pip install faster-whisper   (recommended)\n"
                     "    pip install openai-whisper")

    print(f"[backend] {backend}  [model] {args.model}  [lang] {lang}")
    print(f"[transcript] {text}")


if __name__ == "__main__":
    main()
