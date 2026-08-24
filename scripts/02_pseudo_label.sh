#!/usr/bin/env bash
# Pseudo-labelling skeleton: run a large teacher over unlabeled audio to
# produce the training transcripts your small student distills from.
# Teacher quality dominates student WER — do not economize here.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "[pseudo-label] skeleton. Requires: pip install nemo_toolkit[asr]"
cat <<'PY'
# import nemo.collections.asr as nemo_asr, glob, json
# teacher = nemo_asr.models.ASRModel.from_pretrained("nvidia/parakeet-tdt-0.6b-v3")
# with open("data/pseudo_train.jsonl","w") as out:
#     for wav in glob.glob("data/unlabeled/**/*.wav", recursive=True):
#         hyp = teacher.transcribe([wav])[0]
#         # confidence-filter; drop where two teachers disagree beyond a threshold
#         out.write(json.dumps({"audio": wav, "text": hyp.text}) + "\n")
PY
echo "Then train Model 1 with --manifest data/pseudo_train.jsonl (hard-target distillation)."
