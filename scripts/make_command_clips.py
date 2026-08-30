"""Generate a real command-speech dataset with TTS (for a WORKING demo).

A small model can learn a limited command vocabulary from a few hundred
clips — unlike general ASR, which needs 100s of hours. This makes real
command phrases with gTTS, saves 16 kHz mono WAVs, and writes normalized
train/eval manifests (ground-truth text — no Whisper needed, though you can
still pseudo-label to exercise that path).

    pip install gTTS librosa soundfile
    python scripts/make_command_clips.py --n 300 --out data/cmd

Then train a small model:
    python -m edge_asr.training.train_model1 --config configs/asr_command_small.yaml \
        --tokenizer char --manifest data/cmd/train.jsonl --steps 4000 --warmup 400 \
        --device auto --out runs/cmd
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from edge_asr.data.text_norm import normalize

NAMES = ["mom", "dad", "alex", "sam", "john", "sara", "boss", "emma", "mike", "lucy"]
NUMS = ["one", "two", "three", "five", "ten", "fifteen", "twenty", "thirty"]
UNITS = ["minutes", "seconds", "hours"]
CITIES = ["london", "paris", "tokyo", "delhi", "berlin", "madrid", "rome"]


def all_phrases():
    p = []
    for n in NAMES:
        p.append(f"call {n}")
        p.append(f"send a message to {n}")
    for num in NUMS:
        for u in UNITS:
            p.append(f"set a timer for {num} {u}")
        p.append(f"set an alarm for {num}")
    for c in CITIES:
        p.append(f"what is the weather in {c}")
    p += ["play some music", "stop the music", "what time is it", "pause the song",
          "turn on the flashlight", "start a workout", "cancel the timer", "next song",
          "turn up the volume", "turn down the volume", "open the maps", "answer the call"]
    return p


# multiple English accents -> more data and acoustic variety (helps it generalize)
TLDS = ["com", "co.uk", "co.in"]


def synth(text, path, tmp, tld, tries=3):
    from gtts import gTTS
    import librosa
    import soundfile as sf

    mp3 = os.path.join(tmp, "t.mp3")
    for k in range(tries):
        try:
            gTTS(text=text, lang="en", tld=tld).save(mp3)
            break
        except Exception:
            import time
            time.sleep(1.5 * (k + 1))
    else:
        raise RuntimeError("gTTS failed after retries (rate limit?)")
    y, _ = librosa.load(mp3, sr=16000, mono=True)
    sf.write(path, y, 16000, subtype="PCM_16")
    return len(y) / 16000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--out", default="data/cmd")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(os.path.join(args.out, "wav"), exist_ok=True)
    tmp = tempfile.mkdtemp()
    # (phrase, accent) pairs -> the dataset
    jobs = [(t, tld) for t in all_phrases() for tld in TLDS]
    rng = random.Random(args.seed)
    rng.shuffle(jobs)
    jobs = jobs[: args.n]
    print(f"[gen] {len(jobs)} clips ({len(all_phrases())} phrases x {len(TLDS)} accents)")

    rows = []
    for i, (text, tld) in enumerate(jobs):
        p = os.path.abspath(os.path.join(args.out, "wav", f"{i}.wav"))
        try:
            dur = synth(text, p, tmp, tld)
        except Exception as e:
            print(f"  [skip {i}] {type(e).__name__}: {e}")
            continue
        rows.append({"audio": p, "text": normalize(text), "duration": round(dur, 2)})
        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{len(jobs)} clips")

    rng.shuffle(rows)
    n_eval = max(20, len(rows) // 6)
    train, ev = rows[:-n_eval], rows[-n_eval:]
    with open(os.path.join(args.out, "train.jsonl"), "w") as f:
        f.write("\n".join(json.dumps(r) for r in train))
    with open(os.path.join(args.out, "eval.jsonl"), "w") as f:
        f.write("\n".join(json.dumps(r) for r in ev))
    print(f"[done] train {len(train)} | eval {len(ev)} clips -> {args.out}/")


if __name__ == "__main__":
    main()
