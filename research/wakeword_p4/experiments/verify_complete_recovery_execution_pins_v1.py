"""Require authenticated immutable files and fresh formal outputs."""
import hashlib,json
from pathlib import Path
pins=json.loads(Path('research/wakeword_p4/results/4090-complete-recovery-execution-pins-v1.json').read_text())
for name,digest in pins['file_sha256'].items():
 assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==digest,name
for mode in ['control','recovered']:
 assert not Path(f'.local/wakeword/model-complete-recovery-{mode}-formal-v1').exists()
 assert not Path(f'.local/wakeword/complete-recovery-{mode}-formal-v1.log').exists()
print('Complete recovery pins verified; both outputs fresh',flush=True)
