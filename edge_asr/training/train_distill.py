"""Distill a large teacher into a small streaming student (Model 1).

    # 1) train / obtain a teacher checkpoint (larger model)
    # 2) distill a small student from it:
    python -m edge_asr.training.train_distill \
        --teacher runs/teacher/model1.pt --student-config configs/model1_mamba.yaml \
        --manifest data/train.jsonl --steps 2000 --out runs/student

Total loss = RNN-T + 0.2*CTC (student's own) + kd_weight * CTC-KD(teacher||student)
             (+ optional feature-KD).

Two teacher uses:
  * `--pseudo-label` : replace manifest text with the teacher's hypotheses
    (sequence-level KD) — the standard "teacher labels unlabeled audio" path.
  * frame-level CTC-KD : always on when a teacher is given.

Runnable on synthetic data end-to-end (teacher + student both synthetic).
"""
from __future__ import annotations

import argparse
import os

import torch
from torch.utils.data import DataLoader

from ..data import (ManifestDataset, collate_asr, load_tokenizer,
                    make_synthetic_asr_manifest)
from ..decode import greedy_search
from ..distill import FeatureProjector, Teacher, ctc_kd_loss, feature_kd_loss
from ..eval import wer
from ..features import LogMelFrontend, OnlineCMVN
from .utils import build_model1, configure_optimizer, load_config, set_seed


def _load_transducer(ckpt_path, tok):
    blob = torch.load(ckpt_path, weights_only=False)
    m = build_model1(blob["config"], blob["vocab_size"])
    m.load_state_dict(blob["model"])
    return m, blob["config"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", required=True, help="teacher .pt checkpoint")
    ap.add_argument("--student-config", required=True)
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--tokenizer", default="char")
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--kd-weight", type=float, default=1.0)
    ap.add_argument("--tau", type=float, default=2.0)
    ap.add_argument("--feature-kd", action="store_true")
    ap.add_argument("--pseudo-label", action="store_true",
                    help="relabel the manifest with the teacher's hypotheses (sequence-KD)")
    ap.add_argument("--out", default="runs/student")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    set_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    tok = load_tokenizer(args.tokenizer)
    fe = LogMelFrontend(n_mels=80)
    cmvn = OnlineCMVN(n_mels=80)

    if args.manifest is None:
        args.manifest = os.path.join(args.out, "synth_asr.jsonl")
        make_synthetic_asr_manifest(args.manifest, n=64, seed=args.seed)

    teacher_model, _ = _load_transducer(args.teacher, tok)
    teacher = Teacher(teacher_model, tok)
    n_mels = teacher_model.cfg.encoder.input_dim
    fe = LogMelFrontend(n_mels=n_mels); cmvn = OnlineCMVN(n_mels=n_mels)

    scfg = load_config(args.student_config)
    assert scfg["encoder"]["input_dim"] == n_mels, "teacher/student n_mels must match for CTC-KD"
    student = build_model1(scfg, tok.vocab_size)
    print(f"[params] teacher {teacher_model.num_params()['total']/1e6:.2f} M -> "
          f"student {student.num_params()['total']/1e6:.2f} M")

    ds = ManifestDataset(args.manifest, fe, cmvn, tok, task="asr")

    if args.pseudo_label:
        # sequence-level KD: overwrite each transcript with the teacher's
        # hypothesis (the "teacher labels unlabeled audio" path).
        import json
        relabelled = os.path.join(args.out, "pseudo.jsonl")
        with open(relabelled, "w") as fo:
            for i in range(len(ds)):
                feats, _ = ds[i]
                hyp = teacher.transcribe(feats)
                item = dict(ds.items[i]); item["text"] = hyp
                fo.write(json.dumps(item) + "\n")
        ds = ManifestDataset(relabelled, fe, cmvn, tok, task="asr")
        print(f"[pseudo-label] relabelled {len(ds)} utts with teacher hypotheses -> {relabelled}")

    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_asr)

    projector = None
    params = list(student.parameters())
    if args.feature_kd:
        projector = FeatureProjector(student.encoder.out_dim, teacher_model.encoder.out_dim)
        params += list(projector.parameters())
    opt = configure_optimizer(student, lr=args.lr)
    if projector is not None:
        opt.add_param_group({"params": projector.parameters(), "weight_decay": 1e-2})

    student.train()
    it = iter(dl)
    for step in range(args.steps):
        try:
            f, fl, t, tl = next(it)
        except StopIteration:
            it = iter(dl); f, fl, t, tl = next(it)

        soft = teacher.soft_targets(f, fl, t, tl)
        out = student(f, fl, t, tl, return_features=True)
        kd = ctc_kd_loss(out["ctc_logits"], soft["ctc_logits"], out["enc_lens"], tau=args.tau)
        loss = out["loss"] + args.kd_weight * kd
        if projector is not None:
            loss = loss + 0.1 * feature_kd_loss(out["enc"], soft["enc"], out["enc_lens"], projector)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 5.0); opt.step()
        if step % 50 == 0 or step == args.steps - 1:
            print(f"step {step:4d} loss {loss.item():.3f} (rnnt {out['rnnt'].item():.2f} "
                  f"ctc {out['ctc'].item():.2f} kd {kd.item():.3f})")

    ckpt = os.path.join(args.out, "student.pt")
    torch.save({"model": student.state_dict(), "config": scfg,
                "tokenizer": args.tokenizer, "vocab_size": tok.vocab_size}, ckpt)
    print(f"[save] {ckpt}")

    # compare teacher vs student decode
    student.eval()
    refs, ht, hs = [], [], []
    for i in range(min(8, len(ds))):
        fi, ti = ds[i]
        refs.append(tok.decode(ti.tolist()))
        ht.append(teacher.transcribe(fi))
        hs.append(tok.decode(greedy_search(student, fi)))
    print(f"[eval] teacher WER {wer(refs, ht):.3f} | student WER {wer(refs, hs):.3f}")


if __name__ == "__main__":
    main()
