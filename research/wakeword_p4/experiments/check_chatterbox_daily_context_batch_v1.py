"""Full cache hash/mask check and paired12batch source-budget audit."""
import hashlib,json
from pathlib import Path
import numpy as np
from causal_chatterbox_context_mixed_batch_v1 import Sampler as Control
from causal_chatterbox_daily_context_batch_v1 import Sampler,CACHE
from causal_sequence_supervision_v1 import masks

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    result=json.loads((CACHE/'result.json').read_text());assert result['completed'] and result['pairs']==342
    for name,m in result['files'].items():assert (CACHE/name).stat().st_size==m['bytes'] and sha(CACHE/name)==m['sha256']
    s=Sampler();c=Control()
    for r in s.daily_rows:
        _,pw,nw=masks(r['frames'],0,None,None);assert pw.sum()==0 and np.isclose(nw.sum(),1)
        for name,raw in s.daily_values.items():
            a=raw[r['offset']:r['offset']+r['frames']];assert hashlib.sha256(a.astype('<u2').tobytes()).hexdigest()==r['feature_sha256'][name]
    batches=[]
    for batch in range(12):
        (x,p,n),ds=s.batch();(cx,cp,cn),cd=c.batch();changed=0;conditions=[]
        assert np.array_equal(p,cp)
        for i,d in enumerate(ds):
            if d['kind']=='chatterbox_daily_context_training':
                changed+=1;old=cd[i];assert d['replaced']==old and old['label']==0
                assert d['voice']==old['voice'] and d['condition']==old['condition'] and d['gap_samples']==c.by_pair[old['pair']]['gap_samples']
                assert p[i].sum()==0 and np.isclose(n[i].sum(),1);conditions.append(d['condition'])
            else:assert d==cd[i] and np.array_equal(x[i],cx[i]) and np.array_equal(n[i],cn[i])
        assert changed==4 and conditions.count('silence')==conditions.count('speech')==2
        batches.append(dict(batch=batch,unchanged_inputs_masks_descriptors=124,replaced_negatives=4,positive_weights_exact=True,silence_negative_slots=2,speech_negative_slots=2))
    paths=[Path(__file__),Path(__file__).with_name('causal_chatterbox_daily_context_batch_v1.py'),CACHE/'result.json']
    report=dict(all_passed=True,cache_rows_verified=342,batches=batches,files={str(f):sha(f) for f in paths},limits='Mechanics only, not recognition. Four negative lengths/masks/targets change; all124others exact. Same voice/gap/condition but prefix and phase distributions not identical across target datasets. Uniform6category then uniform admitted row within same voice/gap, RNG732. No model scoring/training.')
    with Path('research/wakeword_p4/results/4090-chatterbox-daily-context-batch-audit-v1.json').open('x') as f:f.write(json.dumps(report,indent=2)+'\n')
    print('342rows and12paired batches passed')
if __name__=='__main__':main()
