"""Authenticate all saved paired-treatment-formal-v1 old-development traces without model inference."""
import hashlib
import json
from pathlib import Path

base = Path('.local/wakeword')
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

model=base/'model-paired-treatment-formal-v1/hey_nyx.tflite'
assert sha(model)=='64f8fe18c5362edb0a9855da866aa0e63891bbd1bc7bee98ac16d488a6d8db2f'
rows = []
for name, synthetic in [('paired-treatment-formal-v1-phone-development-historical-threshold-v1', True),
                        ('paired-treatment-formal-v1-vctk-development-historical-threshold-v1', True),
                        ('paired-treatment-formal-v1-voxconverse-development-v1', False)]:
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
result = dict(candidate='paired-treatment-formal-v1', authenticated_traces=len(rows), all_match=True,
              script_sha256=sha(Path(__file__)), traces=rows,
              limits='Saved file hashes only; no inference, new data, training or acceptance claim.')
Path('research/wakeword_p4/results/4090-paired-treatment-formal-v1-score-receipt-v1.json').write_text(
    json.dumps(result, indent=2) + '\n')
print(json.dumps({k:v for k,v in result.items() if k != 'traces'}))
