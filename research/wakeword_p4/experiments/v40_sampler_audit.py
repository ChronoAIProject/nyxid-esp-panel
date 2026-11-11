"""Verify fixed v40 draws, labels, canonical groups and retained batch slots."""
import hashlib,json,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'training'))
from complete_coverage import CompleteCoverage
from phone_binary import FullSourceBatch
root=Path('.local/wakeword/mac-evidence-import-v1/old-complete-container-pack-v1/study')
a=CompleteCoverage(root);b=CompleteCoverage(root)
base=FullSourceBatch('.local/wakeword/british-training-features-v1')
reference=FullSourceBatch('.local/wakeword/british-training-features-v1')
rng=np.random.default_rng(127726);counts={}
for step in range(50):
 british=base.first80();expected=reference.first80();np.testing.assert_array_equal(british,expected)
 ids=b.draw_indices();independent=[]
 for j in range(40):
  task='positive' if j<16 else str(rng.choice(['hey nick','hey next','hey mix']))
  engine=str(rng.choice(['kokoro','macos-say','openai-tts','piper','qwen']))
  rows={}
  for i,g in enumerate(b.sources):
   r=g['aliases'][0];t='positive' if r['label']==1 else r['phrase']
   if r['engine']==engine and t==task:
    key=json.dumps(sorted(r['speakers'],key=str),sort_keys=True);rows.setdefault(key,[]).append(i)
  identity=str(rng.choice(sorted(rows)));source=int(rng.choice(rows[identity]));variant=int(rng.integers(3));independent.append(source*3+variant)
  counts[engine+'/'+task]=counts.get(engine+'/'+task,0)+1
 assert ids==independent
 full=np.concatenate([british,np.full((48,150,40),-123,dtype=np.float32)])
 before=full.copy();full[:80]=a.replace(full[:80]);scaled=np.clip(a.x[ids].astype(np.float32)/25.6,0,26)
 np.testing.assert_array_equal(full[:16],scaled[:16]);np.testing.assert_array_equal(full[32:56],scaled[16:])
 np.testing.assert_array_equal(full[16:32],before[16:32]);np.testing.assert_array_equal(full[56:],before[56:])
 np.testing.assert_array_equal(british,before[:80])
report=dict(batches=50,candidate_draws=2000,independent_draws_exact=True,labels_exact=True,retained_british_and_ambient_exact=True,base_rng_unchanged=True,counts=counts,sampler=a.report,training_run=False,model_scored=False)
Path('research/wakeword_p4/results/4090-v40-sampler-audit-v1.json').write_text(json.dumps(report,indent=2)+'\n')
print('PASS50 batches; independent canonical draws, labels, retained slots and base RNG')
