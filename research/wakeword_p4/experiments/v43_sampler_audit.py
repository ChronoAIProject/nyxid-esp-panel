"""Independent temporal-window and retained-slot audit before v43 optimization."""
import json,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'training'))
from complete_coverage import CompleteCoverage
root=Path('.local/wakeword/mac-evidence-import-v1/old-complete-container-pack-v1/study')
a=CompleteCoverage(root,negative_timeline=True,positive_timeline=True,negative_container_mixture=True);b=CompleteCoverage(root)
v41=CompleteCoverage(root,negative_timeline=True,positive_timeline=True)
container_rng=np.random.default_rng(157726)
posrng=np.random.default_rng(147726)
rng=np.random.default_rng(137726);ends=[]
base=np.arange(128*150*40,dtype=np.float32).reshape(128,150,40)
for step in range(50):
 ids=b.draw_indices();expected=base.copy()
 samples=np.clip(b.x[ids].astype(np.float32)/25.6,0,26)
 for j,i in enumerate(ids[16:],16):
  raw=np.load(root/b.records[i]['raw'],allow_pickle=False)
  end=int(rng.integers(150,len(raw)+1));ends.append(end)
  samples[j]=np.clip(raw[end-150:end].astype(np.float32)/25.6,0,26)
 for j,i in enumerate(ids[:16]):
  raw=np.load(root/b.records[i]['raw'],allow_pickle=False)
  low=b.records[i]['feature_crop_end'];end=int(posrng.integers(low,min(350,len(raw))+1))
  assert (end-150)*160<=32000 and (end+1)*160>=32000+b.records[i]['signal_samples']
  samples[j]=np.clip(raw[end-150:end].astype(np.float32)/25.6,0,26)
 for j,i in enumerate(ids[16:28],16):
  raw=np.load(root/b.records[i]['raw'],allow_pickle=False)
  low=b.records[i]['feature_crop_end'];end=int(container_rng.integers(low,min(350,len(raw))+1))
  samples[j]=np.clip(raw[end-150:end].astype(np.float32)/25.6,0,26)
 expected[:16]=samples[:16];expected[32:56]=samples[16:]
 actual=base.copy();actual[:80]=a.replace(actual[:80])
 np.testing.assert_array_equal(actual,expected)
 old=v41.replace(base[:80])
 np.testing.assert_array_equal(actual[44:56],old[44:56])
 np.testing.assert_array_equal(actual[:32],old[:32])
 np.testing.assert_array_equal(actual[56:80],old[56:80])
 assert a.positive_rng.bit_generator.state==v41.positive_rng.bit_generator.state
 assert a.container_rng.bit_generator.state==container_rng.bit_generator.state
 assert a.temporal_rng.bit_generator.state==v41.temporal_rng.bit_generator.state
 assert a.rng.bit_generator.state==b.rng.bit_generator.state==v41.rng.bit_generator.state
legal_count=0
for i,raw in a.raw_negatives.items():
 r=a.records[i]
 independent=[e for e in range(150,len(raw)+1) if (e-150)*160<=32000 and (e+1)*160>=32000+r['signal_samples']]
 assert independent==list(range(r['feature_crop_end'],min(350,len(raw))+1))
 legal_count+=len(independent)
assert len(a.raw_negatives)==1995
report=dict(negative_variants=1995,legal_container_ends=legal_count,batches=50,negative_crops=1200,positive_crops=800,v42_retained_negative_and_positive_rng_exact=True,all_crops_and_retained_slots_exact=True,
 source_rng_exact=True,temporal_end_min=min(ends),temporal_end_max=max(ends),
 sampler=a.report,training_run=False,model_scored=False)
Path('research/wakeword_p4/results/4090-v43-sampler-audit-v1.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
