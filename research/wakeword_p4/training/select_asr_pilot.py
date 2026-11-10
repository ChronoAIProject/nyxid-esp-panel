"""Freeze a deterministic, training-only raw-waveform pilot for independent ASR."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--source', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--per-label', type=int, default=100)
p.add_argument('--report', type=Path, required=True)
a = p.parse_args()
if a.output.exists() or a.per_label < 1:
    p.error('Use a fresh output directory and positive sample count')
rows = [json.loads(x) for x in (a.source/'manifest.jsonl').read_text().splitlines()]
selected = []
for label in [0, 1]:
    group = [x for x in rows if x['split'] == 'training' and x['label'] == label]
    if len(group) < a.per_label:
        raise ValueError('Insufficient training samples')
    selected.extend(sorted(group, key=lambda x: hashlib.sha256(x['path'].encode()).hexdigest())[:a.per_label])
a.output.mkdir(parents=True)
for row in selected:
    source = (a.source/row['path']).resolve()
    if not source.is_relative_to(a.source.resolve()):
        raise ValueError('Source escaped corpus')
    if hashlib.sha256(source.read_bytes()).hexdigest() != row['sha256']:
        raise ValueError('Source WAV changed')
    target = a.output/row['path']; target.parent.mkdir(parents=True, exist_ok=True)
    target.symlink_to(source)
(a.output/'manifest.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in selected))
report = dict(role='Training-only deterministic ASR pilot; no filtering or relabeling',
              parent_manifest_sha256=hashlib.sha256((a.source/'manifest.jsonl').read_bytes()).hexdigest(),
              manifest_sha256=hashlib.sha256((a.output/'manifest.jsonl').read_bytes()).hexdigest(),
              selected=len(selected), policy=f'First{a.per_label} per label after sorting training paths by SHA256(path)',
              positive_text_counts=dict(Counter(x['text'] for x in selected if x['label'])),
              source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), evaluation_rows_selected=0)
(a.output/'generation.json').write_text(json.dumps(report, indent=2)+'\n')
a.report.write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps(report))
