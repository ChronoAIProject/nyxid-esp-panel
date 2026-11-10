#!/usr/bin/env bash
# Fresh-run reproduction. Install the CPU environment from the explicit lock:
# micromamba create -p .local/wakeword/mfa-cpu-env -f research/wakeword_p4/requirements-mfa-cpu-explicit.lock
# Download pinned models using training/download_alignment_models.py and
# results/4090-mfa-model-sources-v1.json. Existing paired/control inputs required.
set -euo pipefail
python3 research/wakeword_p4/training/prepare_alignment_development.py \
  --paired-manifest .local/wakeword/comparison-validation-v1/manifest.json \
  --official-manifest .local/wakeword/hiesp-official-probe-v1/manifest.json \
  --output .local/wakeword/mfa-development-inputs-v1
MFA_ROOT_DIR="$PWD/.local/wakeword/mfa-runtime-v1" \
  .local/wakeword/mfa-bootstrap-v1/micromamba --no-rc \
  --root-prefix .local/wakeword/mfa-mamba-root run --prefix .local/wakeword/mfa-cpu-env \
  mfa align .local/wakeword/mfa-development-inputs-v1/corpus \
  .local/wakeword/mfa-model-sources-v1/english_us_arpa.dict \
  .local/wakeword/mfa-model-sources-v1/english_us_arpa.zip \
  .local/wakeword/mfa-development-alignment-v1 \
  --single_speaker --no_use_postgres --num_jobs 4 --clean --overwrite
.local/wakeword/mfa-bootstrap-v1/micromamba --no-rc \
  --root-prefix .local/wakeword/mfa-mamba-root run --prefix .local/wakeword/mfa-cpu-env \
  python research/wakeword_p4/training/collect_alignment_development.py \
  --inputs .local/wakeword/mfa-development-inputs-v1 \
  --alignment .local/wakeword/mfa-development-alignment-v1 \
  --models .local/wakeword/mfa-model-sources-v1 \
  --output .local/wakeword/mfa-development-collected-v1.json
