"""Live mic demo — speak into your Mac and see it transcribed.

    pip install sounddevice faster-whisper
    python scripts/listen.py                 # press Enter to start/stop
    python scripts/listen.py --loop          # keep listening for more
    python scripts/listen.py --seconds 5     # fixed 5-second capture

First run: macOS will ask for **microphone permission** for your terminal
app (Terminal / iTerm / VS Code). Allow it (System Settings → Privacy &
Security → Microphone), then run again.

The same Whisper here is the teacher used for distillation later.
"""
from __future__ import annotations

import argparse
import sys
import tempfile
import wave

SR = 16000


def _need(msg):
    sys.exit(msg)


def record_until_enter():
    import numpy as np
    import sounddevice as sd

    frames = []

    def cb(indata, n, t, status):
        frames.append(indata.copy())

    input("  ▶  Press Enter to START recording...")
    stream = sd.InputStream(samplerate=SR, channels=1, dtype="int16", callback=cb)
    stream.start()
    print("  ●  Recording... speak now, then press Enter to STOP.")
    try:
        input()
    finally:
        stream.stop()
        stream.close()
    if not frames:
        return None
    return np.concatenate(frames).reshape(-1)


def record_fixed(seconds):
    import numpy as np
    import sounddevice as sd

    print(f"  ●  Recording {seconds}s... speak now.")
    audio = sd.rec(int(seconds * SR), samplerate=SR, channels=1, dtype="int16")
    sd.wait()
    return audio.reshape(-1)


def save_wav(audio, path):
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(audio.tobytes())


def make_transcriber(model_name, compute_type):
    try:
        from faster_whisper import WhisperModel
        m = WhisperModel(model_name, device="cpu", compute_type=compute_type)

        def run(path):
            segs, info = m.transcribe(path, beam_size=5)
            return "".join(s.text for s in segs).strip()
        return run
    except ImportError:
        try:
            import whisper
            m = whisper.load_model(model_name)
            return lambda path: m.transcribe(path)["text"].strip()
        except ImportError:
            _need("Install a Whisper backend: pip install faster-whisper")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="base", help="tiny|base|small|medium (+.en)")
    ap.add_argument("--compute-type", default="int8")
    ap.add_argument("--seconds", type=float, default=0, help="fixed duration; 0 = Enter to stop")
    ap.add_argument("--loop", action="store_true", help="keep listening after each turn")
    ap.add_argument("--save", default=None, help="also keep the recording at this path")
    args = ap.parse_args()

    try:
        import sounddevice  # noqa: F401
    except ImportError:
        _need("Install the mic library: pip install sounddevice")

    print(f"[loading] Whisper '{args.model}' (first run downloads it) ...")
    transcribe = make_transcriber(args.model, args.compute_type)
    print("[ready] microphone demo. Ctrl+C to quit.\n")

    try:
        while True:
            audio = record_fixed(args.seconds) if args.seconds > 0 else record_until_enter()
            if audio is None or len(audio) < SR // 4:
                print("  (nothing captured — try again)\n")
                if not args.loop:
                    break
                continue
            path = args.save or tempfile.mktemp(suffix=".wav")
            save_wav(audio, path)
            print("  ... transcribing ...")
            text = transcribe(path)
            print(f"\n  \U0001f5e3  You said:  {text!r}\n")
            if not args.loop:
                break
    except KeyboardInterrupt:
        print("\n[bye]")


if __name__ == "__main__":
    main()
