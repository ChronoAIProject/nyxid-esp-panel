"""Independent temporal-window and retained-slot audit before v41 optimization."""
import json,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'training'))
from complete_coverage import CompleteCoverage
root=Path('.local/wakeword/mac-evidence-import-v1/old-complete-container-pack-v1/study')
a=CompleteCoverage(root,negative_timeline=True);b=CompleteCoverage(root)
rng=np.random.default_rng(137726);ends=[]
base=np.arange(128*150*40,dtype=np.float32).reshape(128,150,40)
for step in range(50):
 ids=b.draw_indices();expected=base.copy()
 samples=np.clip(b.x[ids].astype(np.float32)/25.6,0,26)
 for j,i in enumerate(ids[16:],16):
  raw=np.load(root/b.records[i]['raw'],allow_pickle=False)
  end=int(rng.integers(150,len(raw)+1));ends.append(end)
  samples[j]=np.clip(raw[end-150:end].astype(np.float32)/25.6,0,26)
 expected[:16]=samples[:16];expected[32:56]=samples[16:]
 actual=base.copy();actual[:80]=a.replace(actual[:80])
 np.testing.assert_array_equal(actual,expected)
 assert a.rng.bit_generator.state==b.rng.bit_generator.state
report=dict(batches=50,negative_crops=1200,all_crops_and_retained_slots_exact=True,
 source_rng_exact=True,temporal_end_min=min(ends),temporal_end_max=max(ends),
 sampler=a.report,training_run=False,model_scored=False)
Path('research/wakeword_p4/results/4090-v41-sampler-audit-v1.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
