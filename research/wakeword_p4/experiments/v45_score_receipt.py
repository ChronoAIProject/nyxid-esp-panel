"""Authenticate all saved v45 old-development traces without model inference."""
import hashlib
import json
from pathlib import Path

base = Path('.local/wakeword')
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

model=base/'model-gpu-v45-ami-extra/hey_nyx.tflite'
assert sha(model)=='f5d175a0172d4fde70dce746fa358331c3fcdf590139c2e76113fe14c45e30af'
rows = []
for name, synthetic in [('v45-phone-development-historical-threshold-v1', True),
                        ('v45-vctk-development-historical-threshold-v1', True),
                        ('v45-voxconverse-development-v1', False)]:
    root = base / name
    report = json.loads((root / 'report.json').read_text())
    assert report['model_sha256']==sha(model)
    records = report['records'] if synthetic else report['speakers']
    for i, record in enumerate(records):
        path = root / (f'{i}-scores.npy' if synthetic else record['speaker'] + '-scores.npy')
        expected = record['nyx']['score_sha256'] if synthetic else record['score_sha256']
        actual = sha(path)
        assert actual == expected, str(path)
        rows.append(dict(path=str(path), sha256=actual))
assert len(rows) == 467
result = dict(candidate='v45', authenticated_traces=len(rows), all_match=True,
              script_sha256=sha(Path(__file__)), traces=rows,
              limits='Saved file hashes only; no inference, new data, training or acceptance claim.')
Path('research/wakeword_p4/results/4090-v45-score-receipt-v1.json').write_text(
    json.dumps(result, indent=2) + '\n')
print(json.dumps({k:v for k,v in result.items() if k != 'traces'}))
