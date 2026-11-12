"""Replay original candidate/source and negative-window RNG through selected checkpoint."""
import hashlib,json,sys
from collections import Counter
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'training'))
from complete_coverage import CompleteCoverage
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
base=Path('.local/wakeword');pack=base/'mac-evidence-import-v1/old-complete-container-pack-v1/study'
witness_path=Path('research/wakeword_p4/results/4090-v42-negative-window-witness-v2.json')
witness=json.loads(witness_path.read_text())['witnesses'];counts={d['pack_index']:Counter() for d in witness}
coverage=CompleteCoverage(pack);rng=np.random.default_rng(137726)
training=json.loads((base/'model-gpu-v42-ami-extra/training.json').read_text());assert training['selected_step']==15000
for step in range(1,15001):
 ids=coverage.draw_indices()
 for i in ids[16:]:
  end=int(rng.integers(150,coverage.records[i]['raw_shape'][0]+1))
  if i in counts:counts[i][end]+=1
 if step%3000==0:print(json.dumps({'replayed_steps':step}),flush=True)
rows=[]
for d in witness:
 c=counts[d['pack_index']];rows.append(dict(pack_index=d['pack_index'],selected_draws=sum(c.values()),distinct_ends=len(c),half_window_draws=sum(c[e] for e in d['window_half_ends']),historical_window_draws=sum(c[e] for e in d['window_historical_ends']),end_counts={str(e):v for e,v in sorted(c.items())}))
r=dict(scope='Deterministic source/negative-end RNG replay through v42 selected step15000',steps=15000,witness_sha256=sha(witness_path),sampler_sha256=sha(Path('research/wakeword_p4/training/complete_coverage.py')),script_sha256=sha(Path(__file__)),rows=rows,limits='Reconstructs deterministic sampler draws, not per-step model scores or gradients. High-window sets use final selected weights; not evidence windows were high at draw time or causal proof. Positive RNG independent; retained British/ambient RNG not replayed. No optimizer, new/final data or threshold selection.')
Path('research/wakeword_p4/results/4090-v42-sampling-exposure-v1.json').write_text(json.dumps(r,indent=2)+'\n')
print(json.dumps([{k:v for k,v in d.items() if k!='end_counts'} for d in rows]))
