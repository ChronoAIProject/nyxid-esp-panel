"""Check paired source/pair/mask invariants and actual input choice, 12 batches."""
import hashlib,json
from pathlib import Path
import numpy as np
from causal_chatterbox_context_batch_v1 import Sampler as Control
from causal_chatterbox_context_mixed_batch_v1 import Sampler

def main():
    mixed,speech,silence=Sampler(),Control(condition='speech'),Control(condition='silence')
    checks=[]
    for batch in range(12):
        (x,p,n),d=mixed.batch();(sx,sp,sn),sd=speech.batch();(zx,zp,zn),zd=silence.batch()
        assert np.array_equal(p,sp) and np.array_equal(p,zp)
        assert np.array_equal(n,sn) and np.array_equal(n,zn)
        contextual=0;counts={}
        for i,row in enumerate(d):
            if row['kind']=='chatterbox_context_training':
                contextual+=1
                assert row['pair']==sd[i]['pair']==zd[i]['pair']
                assert row['replaced']==sd[i]['replaced']==zd[i]['replaced']
                condition=row['condition'];key=f"{row['label']}-{condition}";counts[key]=counts.get(key,0)+1
                assert np.array_equal(x[i],zx[i] if condition=='silence' else sx[i])
            else:
                assert row==sd[i]==zd[i]
                assert np.array_equal(x[i],sx[i]) and np.array_equal(x[i],zx[i])
        assert contextual==8 and all(v==2 for v in counts.values()) and len(counts)==4
        checks.append(dict(batch=batch,unchanged_parent_slots=120,context_slots=8,counts=counts))
    files=[Path(__file__),Path(__file__).with_name('causal_chatterbox_context_mixed_batch_v1.py')]
    result=dict(all_passed=True,batches=checks,files={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in files},recognition_evidence=False)
    with Path('research/wakeword_p4/results/4090-context-mixed-batch-audit-v1.json').open('x') as f:f.write(json.dumps(result,indent=2)+'\n')
    print('all12 batches passed')
if __name__=='__main__':main()
