"""Check actual final sampler input changes, without model/gradient comparisons."""
import collections
import json
from pathlib import Path
import numpy as np
from causal_positive_coverage_batch_v1 import Sampler as Parent
from causal_source_balanced_batch_v1 import Sampler,digest

def main():
    parent=Parent();new=Sampler();records=[];matches=collections.Counter();engines=collections.Counter()
    for batch in range(12):
        old,od=parent.batch();current,cd=new.batch()
        slots=[i for i,r in enumerate(cd) if r['kind']=='source_balanced_bare_negative']
        assert len(slots)==8
        untouched=[i for i in range(128) if i not in slots]
        assert all(np.array_equal(a[untouched],b[untouched]) for a,b in zip(old,current,strict=True))
        assert all(od[i]==cd[i] for i in untouched)
        assert collections.Counter(r['kind'] for r in cd)==dict(synthetic=40,ambient=48,existing_nearword=12,existing_kokoro_carrier=12,existing_broad_bare_positive=8,source_balanced_bare_negative=8)
        x,p,n=current
        assert np.isfinite(x).all() and np.allclose(n.sum(axis=1),1) and np.allclose(p.sum(axis=1),[r['label'] for r in cd]) and sum(r['label'] for r in cd)==32
        assert not np.any((p>0)&(n>0))
        for i in slots:
            r=cd[i];source=new.positive_by_offset[r['positive_offset']]
            assert r['engine']==source['engine'] and r['conditioning_id']==source['conditioning_id']
            matches[r['match']]+=1;engines[r['engine']]+=1
        records.append(dict(batch=batch+1,x_sha256=digest(x),p_sha256=digest(p),n_sha256=digest(n),changed_slots=slots,descriptors=[cd[i] for i in slots]))
    out=dict(schema='causal-source-balanced-sampling-check-v1',batches=12,changed_negative_presentations=96,unchanged_presentations=1440,all_positive_ambient_command_and_retained_negative_inputs_preserved=True,counts=dict(matches=matches,engines=engines),records=records,limits='Actual final sampler input/mask check only. No model inference, gradient/numerical comparison, fit or recognition claim. Fallback explicitly keeps source conditioning while allowing different speed/draw/variant; original positive is never replaced.')
    path=Path('research/wakeword_p4/results/4090-causal-source-balanced-sampling-check-v1.json')
    with path.open('x') as f:f.write(json.dumps(out,indent=2)+'\n')
    print(json.dumps(dict(passed=True,counts=out['counts'])))
if __name__=='__main__':main()
