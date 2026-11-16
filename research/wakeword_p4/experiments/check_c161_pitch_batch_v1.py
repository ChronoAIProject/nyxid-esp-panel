"""Actual-batch control: only chosen broad waveform features may change."""
import collections,json
from pathlib import Path
import numpy as np
from causal_source_balanced_batch_v1 import Sampler as Baseline
from causal_pitch_cache_batch_v1 import Sampler

def main():
    baseline=Baseline(726);candidate=Sampler(726);counts=collections.Counter()
    for batch in range(12):
        (x,p,n),ds=baseline.batch();(y,pp,nn),dd=candidate.batch()
        assert np.array_equal(p,pp) and np.array_equal(n,nn)
        for i,(before,after) in enumerate(zip(ds,dd,strict=True)):
            original={k:v for k,v in after.items() if not k.startswith('pitch_')}
            assert before==original,(batch,i)
            if after.get('pitch_applied'):
                assert not np.array_equal(x[i],y[i]);counts['changed/'+str(before['label'])]+=1
            else:assert np.array_equal(x[i],y[i])
            if 'pitch_applied' in after:counts['eligible/'+str(before['label'])]+=1
    assert counts['eligible/1']==counts['eligible/0']==96
    assert 0<counts['changed/1']<96 and 0<counts['changed/0']<96
    out=dict(schema='c161-pitch-batch-check-v1',batches=12,counts=dict(counts),source_schedule_unchanged=True,supervision_unchanged=True,unselected_tensors_unchanged=True,scope='Training-only actual tensors; not recognition or GPU update proof.')
    Path('research/wakeword_p4/results/4090-c161-pitch-batch-check-v1.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
if __name__=='__main__':main()
