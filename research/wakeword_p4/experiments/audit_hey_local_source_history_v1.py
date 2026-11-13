"""Inventory local ONNX bytes and source-name/hash mentions before role assignment.

No audio is opened and no model inference runs. Text matches are evidence to
review, not automatic proof of synthesis or independence.
"""
import hashlib
import json
from pathlib import Path
import subprocess

BASE = Path('.local/wakeword')
OUT = Path('.local/wakeword/hey-paired-source-history-audit-v1')
RECEIPT = Path('research/wakeword_p4/results/4090-hey-local-source-history-audit-v1.json')


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(4*1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    OUT.mkdir(exist_ok=False)
    candidate_root = BASE / 'hey-paired-prospective-models-v1'
    candidates = {p.stem: sha(p) for p in sorted(candidate_root.glob('*/*.onnx'))}
    assert len(candidates) == 4
    needles = {k: k.split('-')[1].lower().encode() for k in candidates}
    needles.update({k+'-sha256': v.encode() for k, v in candidates.items()})
    model_rows = []
    for p in sorted(BASE.rglob('*.onnx')):
        if not p.is_file():
            continue
        digest = sha(p)
        model_rows.append(dict(path=str(p), bytes=p.stat().st_size, sha256=digest,
                               candidate_matches=[k for k,v in candidates.items() if v == digest]))
    metadata = []
    matches = []
    for p in sorted(BASE.rglob('*')):
        if p.suffix not in ('.json', '.jsonl') or not p.is_file() or candidate_root in p.parents:
            continue
        h = hashlib.sha256()
        hits = set()
        tail = b''
        with p.open('rb') as f:
            for chunk in iter(lambda: f.read(4*1024*1024), b''):
                h.update(chunk)
                block = (tail+chunk).lower()
                hits.update(k for k,v in needles.items() if v in block)
                tail = block[-128:]
        metadata.append(dict(path=str(p), sha256=h.hexdigest(), bytes=p.stat().st_size))
        if hits:
            matches.append(dict(path=str(p), matches=sorted(hits)))
    inventory = OUT / 'inventory.json'
    inventory.write_text(json.dumps(dict(models=model_rows, metadata=metadata), indent=2)+'\n')
    duplicates = [r for r in model_rows if r['candidate_matches'] and candidate_root not in Path(r['path']).parents]
    result = dict(schema='hey-local-source-history-audit-v1',
                  source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                  script_sha256=sha(Path(__file__)), candidate_model_sha256=candidates,
                  onnx_files=len(model_rows), metadata_files=len(metadata),
                  inventory_path=str(inventory), inventory_sha256=sha(inventory),
                  candidate_bytes_at_prior_model_paths=duplicates, metadata_mentions=matches,
                  audio_opened=False, inference_run=False, roles_assigned=False,
                  limits='Current accessible local ONNX and JSON/JSONL inventory only, not deleted/external files or PCM identity audit. Mentions are not consumption. Shared parent pretraining remains a limitation. Mutable live metadata snapshot is not an execution pin.')
    with RECEIPT.open('x') as f:
        f.write(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__ == '__main__':
    main()
