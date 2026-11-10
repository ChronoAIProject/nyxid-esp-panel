#!/usr/bin/env bash
# Reproduce the public-reference diagnostic on fresh local input/output paths.
set -euo pipefail
python3 - <<'PY'
import hashlib
import json
from pathlib import Path
import urllib.request
root=Path('.local/wakeword/mfa-ldc-sample-source-v1')
root.mkdir(parents=True,exist_ok=True)
sources={f'LDC93S1.{suffix}':f'https://catalog.ldc.upenn.edu/desc/addenda/LDC93S1.{suffix}'
         for suffix in ('wav','phn','txt')}
sources['catalog.html']='https://catalog.ldc.upenn.edu/LDC93S1'
records=[]
for name,url in sources.items():
    path=root/name
    if not path.exists():
        with urllib.request.urlopen(url,timeout=60) as response:
            path.write_bytes(response.read())
    records.append(dict(path=name,url=url,bytes=path.stat().st_size,
                        sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
manifest=root/'download.json'
if not manifest.exists():
    manifest.write_text(json.dumps(records,indent=2)+'\n')
PY
python3 research/wakeword_p4/training/prepare_ldc_alignment_reference.py \
  --source .local/wakeword/mfa-ldc-sample-source-v1 \
  --output .local/wakeword/mfa-ldc-sample-inputs-v1
MFA_ROOT_DIR="$PWD/.local/wakeword/mfa-runtime-ldc-sample-v1" \
  .local/wakeword/mfa-bootstrap-v1/micromamba --no-rc \
  --root-prefix .local/wakeword/mfa-mamba-root run \
  --prefix .local/wakeword/mfa-cpu-env mfa align \
  .local/wakeword/mfa-ldc-sample-inputs-v1/corpus \
  .local/wakeword/mfa-model-sources-v1/english_us_arpa.dict \
  .local/wakeword/mfa-model-sources-v1/english_us_arpa.zip \
  .local/wakeword/mfa-ldc-sample-alignment-v1 \
  --single_speaker --no_use_postgres --num_jobs 4 --clean --overwrite
.local/wakeword/mfa-bootstrap-v1/micromamba --no-rc \
  --root-prefix .local/wakeword/mfa-mamba-root run \
  --prefix .local/wakeword/mfa-cpu-env python \
  research/wakeword_p4/training/collect_alignment_development.py \
  --inputs .local/wakeword/mfa-ldc-sample-inputs-v1 \
  --alignment .local/wakeword/mfa-ldc-sample-alignment-v1 \
  --models .local/wakeword/mfa-model-sources-v1 \
  --output research/wakeword_p4/results/4090-mfa-ldc-sample-alignment-v1.json
python3 research/wakeword_p4/training/audit_reference_endpoint.py \
  --inputs .local/wakeword/mfa-ldc-sample-inputs-v1/manifest.json \
  --alignment research/wakeword_p4/results/4090-mfa-ldc-sample-alignment-v1.json \
  --output research/wakeword_p4/results/4090-mfa-ldc-sample-endpoint-audit-v1.json
