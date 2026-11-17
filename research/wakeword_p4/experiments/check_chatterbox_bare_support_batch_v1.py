"""Paired actual tensors:126 unchanged slots,2 new bare slots each batch."""
import json,hashlib
from pathlib import Path
from collections import Counter
import numpy as np
from causal_chatterbox_daily_context_batch_v1 import Sampler as Parent
from causal_chatterbox_bare_support_batch_v1 import Sampler,CACHE

def main():
    a=Parent();b=Sampler();counts=Counter();proof=[]
    for step in range(12):
        old,od=a.batch();new,nd=b.batch();changed=[]
        for i,(d,e) in enumerate(zip(od,nd,strict=True)):
            if e['kind']=='chatterbox_bare_support_training':
                assert e['replaced']==d and d['kind']=='existing_broad_bare_positive';changed.append(i);counts[e['voice']+'/'+e['condition']]+=1
                assert np.isclose(new[1][i].sum(),1) and np.isclose(new[2][i].sum(),1) and not np.any((new[1][i]>0)&(new[2][i]>0))
            else:
                assert d==e
                for x,y in zip(old,new,strict=True):assert np.array_equal(x[i],y[i])
        assert len(changed)==2 and {nd[i]['condition'] for i in changed}=={'silence','speech'}
        assert sum(d['kind']=='chatterbox_context_training' and d['label']==1 for d in nd)==4
        proof.append(dict(batch=step+1,changed_slots=changed,unchanged_slots=126))
    assert len(counts)==8 and set(counts.values())=={3}
    files=[Path(__file__),Path('research/wakeword_p4/experiments/causal_chatterbox_bare_support_batch_v1.py'),CACHE/'records.json']
    result=dict(paired_batches=proof,voice_history_counts=dict(counts),files={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},policy='12actual paired128input/mask/descriptor batches;126identical including all4originalChatterboxpositive and4dailynegative slots. Replace2of4remainingbroadbare slots. New23provisionalbare sources, same4trainingvoices, one speech+one silence perbatch, exactvoice/history balance per4batches, separateRNG733. Not lexical-only/phonetic causal isolation, no model training or acceptance.')
    with Path('research/wakeword_p4/results/4090-chatterbox-bare-support-batch-audit-v1.json').open('x') as f:f.write(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))
if __name__=='__main__':main()
