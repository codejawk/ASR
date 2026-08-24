# Roadmap — from this scaffold to a shipped model

Phases are ordered by risk, not by the natural build order: de-risk the
toolchain and the licensing before you spend GPU-weeks.

## Phase 0 — verify the toolchain (this week)
- [x] Whole pipeline runs on synthetic data (`bash scripts/smoke_test.sh`).
- [ ] **Confirm 10 MB = flash or resident RAM** (docs/DEPLOYMENT_SW6100.md).
- [ ] Export a **stock streaming-Zipformer** to QNN, run on real SW6100.
      Regenerate the context binary against the Wear-Elite target.
- [ ] Clear **Loquacious** (and any other corpus) with legal
      (docs/DATA_LICENSING.md). **Do not use GigaSpeech.**

## Phase 1 — recipe pilot
- [ ] `tools/count_params.py` on the target config → confirm <10 MB int8.
- [ ] LibriSpeech-960 pilot (2 days, 4 GPUs) — validates convergence,
      streaming export round-trip, and greedy/streaming WER gap.
- [ ] Swap `StreamingConformerEncoder` → icefall **Zipformer** using the
      same config surface; re-confirm param count.

## Phase 2 — data at scale
- [ ] Loquacious 25k h + Common Voice + MLS manifests.
- [ ] Pseudo-label YODAS-en + unlabeled audio with a large teacher.
- [ ] **Collect on-device wrist recordings** (positions × noise × domain) —
      the highest-WER-impact task in the whole project.
- [ ] MUSAN + RIR augmentation dirs wired into `data/augment.py`.

## Phase 3 — train + compress Model 1
- [ ] Full training: pruned RNN-T + 0.2·CTC, hard-target distillation.
- [ ] **int8 QAT** (W8A8, per-channel; predictor embedding fp16).
- [ ] Calibration/QAT data must include chunked streaming execution
      conditions, not just random clips.

## Phase 4 — Model 2 + cascade
- [ ] Decide fixed vs extensible command set → BC-ResNet or phoneme-CTC+FST.
- [ ] Tune threshold on **tens of hours of real ambient audio** for
      **FA/hour at fixed FRR** (`KeywordSpotter.score_false_accepts`).
- [ ] Wire `CascadePipeline`: eNPU command model gates Hexagon ASR.

## Phase 5 — ship gate
- [ ] Wrist eval set built (not LibriSpeech).
- [ ] Report WER + task-success + latency **p50/p95** + battery/active-min.
- [ ] Contextual biasing (contacts/apps) via n-gram FST — lives outside the
      model, doesn't eat the 10 MB.
- [ ] ITN + punctuation as a separate tiny model — budget separately.

## Open questions to resolve with product
1. English-only, or per-language downloadable models?
2. Streaming, or push-to-talk 3–8 s utterance? (Non-streaming buys ~2%
   absolute WER at the same size.)
3. Target WER and target FA/hour — without these you can't tell when done.
4. Is the command set fixed at ship time or extensible?
