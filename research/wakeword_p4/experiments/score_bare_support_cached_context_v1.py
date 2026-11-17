"""Authenticate all92cache traces, masks and exact native-clock events."""
import json,hashlib,tarfile
from collections import defaultdict
import numpy as np
from replay_bare_support_cached_context_v1 import PLAN,OUT,R,sha
from uint8_event_policy_v1 import event_steps

def main():
    p=json.loads((OUT/'result.json').read_text());assert p['plan_sha256']==sha(PLAN) and len(p['records'])==92
    groups=defaultdict(lambda:dict(clips=0,hits=0,misses=[]))
    for r in p['records']:
        assert sha(r['trace']['path'])==r['trace']['sha256']
        with np.load(r['trace']['path']) as z:
            codes=z['uint8'];assert codes.dtype==np.uint8
            events=[dict(step=s,relative_sample=int(z['relative_block_return'][s-1])) for s in event_steps(codes,.91796875)]
            assert events==r['events'] and [e for e in events if e['relative_sample']>0]==r['target_events']
            assert np.flatnonzero(z['positive_weights']).tolist()==r['positive_calls'] and not np.any((z['positive_weights']>0)&(z['negative_weights']>0))
        g=groups[r['condition']+'/'+str(r['gap_samples'])];g['clips']+=1;g['hits']+=bool(r['target_events'])
        if not r['target_events']:g['misses'].append(dict(index=r['index'],max_target_sum5=r['max_target_sum5'],positive_mask_mean_code=r['positive_mask_mean_code']))
    files=sorted(OUT.iterdir());members={f.name:dict(bytes=f.stat().st_size,sha256=sha(f)) for f in files};archive=R/'4090-bare-support-cached-context-evidence-v1.tar.gz'
    with tarfile.open(archive,'x:gz') as tar:
        for f in files:tar.add(f,arcname=f.name)
    with tarfile.open(archive,'r:gz') as tar:
        assert set(tar.getnames())==set(members)
        for m in tar:
            b=tar.extractfile(m).read();assert len(b)==members[m.name]['bytes'] and hashlib.sha256(b).hexdigest()==members[m.name]['sha256']
    report=dict(actual_replay_exit_code=0,groups=dict(groups),archive=dict(path=str(archive),bytes=archive.stat().st_size,sha256=sha(archive),members=members),policy=p['policy'])
    with (R/'4090-bare-support-cached-context-comparison-v1.json').open('x') as f:f.write(json.dumps(report,indent=2)+'\n')
    print(json.dumps(groups))
if __name__=='__main__':main()
