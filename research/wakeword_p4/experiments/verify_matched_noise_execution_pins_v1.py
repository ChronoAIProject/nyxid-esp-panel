"""Fresh outputs and exact preregistered files required; no automatic retries."""
import hashlib,json
from pathlib import Path
pins=json.loads(Path('research/wakeword_p4/results/4090-matched-noise-execution-pins-v1.json').read_text())
for name,digest in pins['file_sha256'].items():assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==digest,name
for mode in ['independent','common']:
 assert not Path(f'.local/wakeword/model-matched-noise-{mode}-formal-v1').exists()
 assert not Path(f'.local/wakeword/matched-noise-{mode}-formal-v1.log').exists()
print('Matched noise pins verified, formal outputs fresh',flush=True)
