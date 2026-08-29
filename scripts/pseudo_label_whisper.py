"""Phase 1 — pseudo-label a folder of audio with Whisper (the teacher) into a
training manifest for the small student.

This is the *real* sequence-level distillation data path: Whisper writes the
transcripts, the student learns from them. Chains straight into training:

    # 1) label your clips
    python scripts/pseudo_label_whisper.py --audio-dir data/clips \
        --out data/pseudo.jsonl --model small

    # 2) train the small student on real audio + Whisper labels (CPU ok)
    python -m edge_asr.training.train_model1 \
        --config configs/model1_general.yaml --tokenizer char \
        --manifest data/pseudo.jsonl --steps 2000 --out runs/student

Downstream training expects **16 kHz mono WAV**. Convert first if needed:
    ffmpeg -i in.m4a -ac 1 -ar 16000 out.wav
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import wave


def load_backend(model_name, device, compute_type):
    try:
        from faster_whisper import WhisperModel
        m = WhisperModel(model_name, device=device, compute_type=compute_type)

        def run(path):
            segs, _ = m.transcribe(path, beam_size=5)
            return "".join(s.text for s in segs).strip()
        return run, "faster-whisper"
    except ImportError:
        try:
            import whisper
            m = whisper.load_model(model_name)

            def run(path):
                return m.transcribe(path)["text"].strip()
            return run, "openai-whisper"
        except ImportError:
            sys.exit("Install a Whisper backend: pip install faster-whisper")


def wav_duration(path):
    try:
        with wave.open(path, "rb") as w:
            return w.getnframes() / float(w.getframerate()), w.getframerate()
    except Exception:
        return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio-dir", required=True)
    ap.add_argument("--out", default="data/pseudo.jsonl")
    ap.add_argument("--model", default="small")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--compute-type", default="int8")
    ap.add_argument("--glob", default="**/*.wav")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.audio_dir, args.glob), recursive=True))
    if not files:
        sys.exit(f"No audio matched {args.audio_dir}/{args.glob}")
    transcribe, backend = load_backend(args.model, args.device, args.compute_type)
    print(f"[teacher] {backend} · {args.model} · {len(files)} files")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    n_ok, n_warn = 0, 0
    with open(args.out, "w") as fo:
        for i, path in enumerate(files):
            text = transcribe(path)
            if not text:
                continue
            dur, sr = wav_duration(path)
            if sr is not None and sr != 16000:
                n_warn += 1  # training needs 16 kHz mono; see header for ffmpeg
            rec = {"audio": path, "text": text}
            if dur:
                rec["duration"] = round(dur, 2)
            fo.write(json.dumps(rec) + "\n")
            n_ok += 1
            print(f"  [{i+1}/{len(files)}] {os.path.basename(path)} -> {text[:60]!r}")

    print(f"[done] wrote {n_ok} labels -> {args.out}")
    if n_warn:
        print(f"[warn] {n_warn} file(s) are not 16 kHz; resample before training "
              f"(ffmpeg -i in.wav -ac 1 -ar 16000 out.wav)")


if __name__ == "__main__":
    main()
