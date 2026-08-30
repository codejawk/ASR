"""Record a few takes of each command in YOUR voice -> labeled training data.

Small real-voice data + fine-tuning (warm-start from the TTS model) is enough
to make the mic demo recognize *you*. Run on your Mac (needs a mic).

    pip install sounddevice
    python scripts/record_commands.py --takes 6 --seconds 3 --out data/myvoice

Then fine-tune the existing command model on your voice:
    python -m edge_asr.training.train_model1 --config configs/asr_command_small.yaml \
        --tokenizer char --manifest data/myvoice/train.jsonl --init runs/cmd/model1.pt \
        --steps 800 --warmup 80 --lr 1e-4 --batch-size 8 --device cpu --out runs/myvoice

Then test it on your voice:
    python scripts/listen_model.py --ckpt runs/myvoice/model1.pt --loop
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import wave

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from edge_asr.data.text_norm import normalize

SR = 16000
DEFAULT_COMMANDS = [
    "call mom",
    "set a timer for ten minutes",
    "play some music",
    "stop the music",
    "what time is it",
    "what is the weather in london",
    "turn on the flashlight",
    "start a workout",
]


def record_fixed(seconds):
    import sounddevice as sd

    audio = sd.rec(int(seconds * SR), samplerate=SR, channels=1, dtype="int16")
    sd.wait()
    return audio.reshape(-1)


def save_wav(int16_audio, path):
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(int16_audio.tobytes())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--commands", default=None, help="comma-separated; default = a small set")
    ap.add_argument("--takes", type=int, default=6, help="recordings per command")
    ap.add_argument("--seconds", type=float, default=3.0, help="seconds per recording")
    ap.add_argument("--out", default="data/myvoice")
    args = ap.parse_args()

    try:
        import sounddevice  # noqa
    except ImportError:
        sys.exit("Install the mic library: pip install sounddevice")

    cmds = [c.strip() for c in args.commands.split(",")] if args.commands else DEFAULT_COMMANDS
    os.makedirs(os.path.join(args.out, "wav"), exist_ok=True)
    print(f"\nRecording {len(cmds)} commands x {args.takes} takes = {len(cmds)*args.takes} clips.")
    print("Speak clearly. First take of each command is held out for eval.\n")

    train, ev, idx = [], [], 0
    for ci, cmd in enumerate(cmds):
        print(f"[{ci+1}/{len(cmds)}] Command:  \"{cmd}\"")
        for t in range(args.takes):
            input(f"    take {t+1}/{args.takes} — press Enter, then say it...")
            print(f"    ● recording {args.seconds:.0f}s...")
            audio = record_fixed(args.seconds)
            p = os.path.abspath(os.path.join(args.out, "wav", f"{idx}.wav"))
            save_wav(audio, p)
            row = {"audio": p, "text": normalize(cmd), "duration": args.seconds}
            (ev if t == 0 else train).append(row)  # hold out take 1 for eval
            idx += 1
        print()

    with open(os.path.join(args.out, "train.jsonl"), "w") as f:
        f.write("\n".join(json.dumps(r) for r in train))
    with open(os.path.join(args.out, "eval.jsonl"), "w") as f:
        f.write("\n".join(json.dumps(r) for r in ev))
    print(f"[done] train {len(train)} | eval {len(ev)} clips -> {args.out}/")
    print("Next: fine-tune with --init runs/cmd/model1.pt (see this file's header).")


if __name__ == "__main__":
    main()
