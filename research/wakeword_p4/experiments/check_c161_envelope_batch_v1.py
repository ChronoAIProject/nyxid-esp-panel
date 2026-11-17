"""Verify all cache rows then12 paired actual batches, no model updates."""
import collections,hashlib,json
from pathlib import Path
import numpy as np
from causal_source_balanced_batch_v1 import Sampler as Baseline
from causal_envelope_cache_batch_v1 import Sampler,CACHE
from build_c161_envelope_cache_v1 import selected,alpha
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
    result=json.loads((CACHE/'result.json').read_text())
    for name,h in result['files'].items():assert sha(CACHE/name)==h
    rows=[json.loads(s) for s in (CACHE/'records.jsonl').read_text().splitlines()];sources=selected()
    data=np.memmap(CACHE/'features.bin',dtype='<u2',mode='r').reshape(-1,40);offset=0;groups={};classes=collections.Counter()
    for row,source in zip(rows,sources,strict=True):
        assert row['offset']==offset and row['original_offset']==source['offset'] and row['frames']==source['frames'] and row['label']==source['label'] and row['alpha']==alpha(source)
        assert row['source_sha256']==source['sha256']
        raw=data[offset:offset+row['frames']];assert hashlib.sha256(raw.tobytes()).hexdigest()==row['features_sha256'];offset+=row['frames']
        key=json.dumps(dict(engine=source['engine'],pair=source['source_pair_key'],variant=source['variant']),sort_keys=True)
        groups.setdefault(key,set()).add(row['alpha']);classes[(row['engine'],row['label'])]+=1
    assert len(rows)==31644 and offset==result['frames']==len(data) and all(len(v)==1 for v in groups.values())
    baseline=Baseline(726);candidate=Sampler(726);counts=collections.Counter();batches=[]
    for batch in range(12):
        (x,p,n),ds=baseline.batch();(y,pp,nn),dd=candidate.batch();assert np.array_equal(p,pp) and np.array_equal(n,nn)
        changed=0
        for i,(before,after) in enumerate(zip(ds,dd,strict=True)):
            assert before=={k:v for k,v in after.items() if not k.startswith('envelope_')}
            if after.get('envelope_applied'):
                assert not np.array_equal(x[i],y[i]);counts['changed/'+str(before['label'])]+=1;changed+=1
            else:assert np.array_equal(x[i],y[i])
            if 'envelope_applied' in after:counts['eligible/'+str(before['label'])]+=1
        batches.append(dict(batch=batch,changed=changed,unchanged=128-changed))
    assert counts['eligible/0']==counts['eligible/1']==96 and all(0<counts['changed/'+str(k)]<96 for k in [0,1])
    out=dict(cache_result_sha256=sha(CACHE/'result.json'),all_row_hashes_verified=31644,source_group_alpha_consistent=True,classes=[dict(engine=e,label=l,rows=n) for (e,l),n in sorted(classes.items())],counts=dict(counts),batches=batches,source_schedule_unchanged=True,supervision_unchanged=True,unselected_tensors_unchanged=True,files={str(p):sha(p) for p in [Path(__file__),Path(__file__).with_name('causal_envelope_cache_batch_v1.py')]})
    dest=Path('research/wakeword_p4/results/4090-c161-envelope-batch-check-v1.json')
    with dest.open('x') as f:json.dump(out,f,indent=2)
    print(json.dumps(out),flush=True)
if __name__=='__main__':main()
