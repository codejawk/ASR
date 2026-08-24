# Data & licensing — read before writing training code

The highest-value practical finding for a **commercial** ship: some of the
most-used ASR corpora are **not usable commercially**. Verify each against
your own legal review; this is engineering guidance, not legal advice.

## ❌ Do NOT train on

| Corpus | Why |
|--------|-----|
| **GigaSpeech** | Terms restrict to non-commercial research/education; sources may be copyrighted. A lot of small-ASR papers use it — you cannot. |

## ✅ Commercially usable English data

| Corpus | Hours | License | Notes |
|--------|-------|---------|-------|
| **Loquacious** | ~25,000 | commercial-OK | Curated blend (CommonVoice, VoxPopuli, Libriheavy, People's Speech, YODAS), normalized. Built precisely because the alternatives have license/quality problems. **Start here.** |
| MLS (English) | ~44,000 | CC-BY | LibriVox audiobooks; attribution required. |
| Common Voice | ~26,000 | CC0 | 100+ languages; **accent metadata** is valuable for fairness testing. |
| YODAS (en) | 370k+ | CC (auto transcripts) | Treat as **unlabeled** → pseudo-label it. |
| LibriSpeech | 960 | CC BY 4.0 | **Benchmark only** — don't train on it alone. |

## The data that matters most: on-device recordings

Public data will not capture your device. Record on the **SW6100 reference
hardware** through its real NS chain:

- Positions: wrist-distance (20–50 cm), held-to-mouth, arm-down.
- Noise: street, café, gym, in-car, wind.
- Domain: your command grammar + dictation (messages, notes), plus the
  long tail — contact names, app names, numbers, dates.

A few hundred hours of this moves WER more than doubling public data or any
architecture change at this scale.

## Multilingual

**Off the table at 10 MB.** For global ship, use **per-language downloadable
models**: at ~10–30 M params, monolingual training on a balanced mix beats
multilingual (the effect is stronger the smaller the model). This is also
the answer if the 10 MB cap turns out to be **flash** rather than resident
RAM — ship a small resident model and page/download per-language assets.

## Pipeline

1. `scripts/01_prepare_data.sh` — download (post-licensing) → write JSONL
   manifests `{"audio","text","duration"}`; train SentencePiece **BPE-500**.
2. `scripts/02_pseudo_label.sh` — run a large teacher over unlabeled audio;
   confidence-filter; drop dual-teacher disagreements.
3. `training/train_model1.py --manifest <your.jsonl>` — hard-target distill.
