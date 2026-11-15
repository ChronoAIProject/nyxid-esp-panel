"""Analytic full5000step sampling contract, no audio or neural inference."""
import collections,hashlib,json
from pathlib import Path
from matchbox_long_batch_v1 import Sampler as Original
from matchbox_grid_aligned_batch_v1 import Sampler
R=Path('research/wakeword_p4/results')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 a=Original('pooled');b=Sampler('pooled');delta=collections.Counter();phases=collections.Counter();trace=hashlib.sha256();full=0
 for step in range(1,5001):
  old=a.descriptors();new=b.descriptors();assert len(old)==len(new)==128
  for x,y in zip(old,new,strict=True):
   if x['kind']=='full-bare':
    assert y['original_after']==x['after'];assert {k:v for k,v in x.items() if k not in ['start','after']}=={k:v for k,v in y.items() if k not in ['start','after','original_after']}
    row=b.rows[y['index']];end=row['source_lead']+row['complete_source_and_rir_samples'];assert y['start']<=row['source_lead'] and y['start']+44800>=end;assert y['start']%160==0 and (y['start']+44800)%480==0
    delta[y['after']-x['after']]+=1;phases[row['engine']]+=1;full+=1
   else:assert x==y
  trace.update(json.dumps(new,sort_keys=True).encode())
 files=[Path(__file__),Path('research/wakeword_p4/experiments/matchbox_grid_aligned_batch_v1.py'),Path('research/wakeword_p4/experiments/matchbox_long_batch_v1.py'),Path('.local/wakeword/matchbox-long-material-v1/records.json')]
 out=dict(schema='matchbox-grid-aligned-batch-contract-v1',files={str(p):sha(p) for p in files},steps=5000,fullphrase_draws=full,engine_draws=dict(phases),endpoint_delta_samples=dict(sorted(delta.items())),descriptors_sha256=trace.hexdigest(),all_original_source_class_pair_order_background_ambient_descriptors_identical=True,all_full_source_and_rir_inside_windows=True,all_fullphrase_ends_on_deployment480grid=True,training_started=False,policy='Only fullphrase window endpoints change, deterministically nearest admissible480grid end in original256..1024source-end envelope, ties earlier. Same source/offset/background/ambient/permutation RNG calls and source draws, including Qwen multiplicity. Nearest mapping is not uniform over admissible grid ends; counts preserved. No PCM read, inference, training or qualification. Change combines MFCC phase and <=grid-boundary end timing, not isolated phase causality. No new source or source-exposure rebalance.')
 with (R/'4090-matchbox-grid-aligned-batch-contract-v1.json').open('x') as f:json.dump(out,f,indent=2);f.write('\n')
 print(json.dumps(dict(fullphrase_draws=full,min_delta=min(delta),max_delta=max(delta))))
if __name__=='__main__':main()
