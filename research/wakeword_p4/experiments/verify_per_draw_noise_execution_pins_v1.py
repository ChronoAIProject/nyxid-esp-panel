"""Require authenticated immutable files and fresh formal outputs."""
import hashlib,json
from pathlib import Path
pins=json.loads(Path('research/wakeword_p4/results/4090-per-draw-noise-execution-pins-v1.json').read_text())
for name,digest in pins['file_sha256'].items():
 assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==digest,name
for mode in ['fixed','per_draw']:
 assert not Path(f'.local/wakeword/model-per-draw-noise-{mode}-formal-v1').exists()
 assert not Path(f'.local/wakeword/per-draw-noise-{mode}-formal-v1.log').exists()
print('Per-draw noise pins verified; both outputs fresh',flush=True)
