"""Refuse changed source or existing formal outputs before starting either arm."""
import hashlib
import json
from pathlib import Path
pins=json.loads(Path('research/wakeword_p4/results/4090-temporal-tail-risk-execution-pins-v1.json').read_text())
for name,expected in pins['file_sha256'].items():
    assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==expected,name
for arm in ['control','treatment']:
    assert not Path(f'.local/wakeword/model-temporal-tail-risk-{arm}-formal-v1').exists()
    assert not Path(f'.local/wakeword/temporal-tail-risk-{arm}-formal-v1.log').exists()
print('Execution pins verified; both formal outputs fresh',flush=True)
