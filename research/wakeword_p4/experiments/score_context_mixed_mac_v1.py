"""Verify all Mac12 raw events and archive original synthetic host evidence."""
import hashlib,json,tarfile
from pathlib import Path
import numpy as np
from uint8_event_policy_v1 import event_steps
R=Path('research/wakeword_p4/results');ROOT=Path('.local/wakeword/context-mixed-mac-coverage-v1')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    p=json.loads((ROOT/'result.json').read_text());plan=R/'4090-context-mixed-mac-coverage-plan-v1.json'
    assert p['plan_sha256']==sha(plan) and len(p['records'])==12
    groups={}
    for r in p['records']:
        trace=ROOT/f'{r["index"]:03d}.npz';assert sha(trace)==r['trace_sha256']
        with np.load(trace) as z:
            assert z['uint8'].dtype==np.uint8
            events=[dict(step=s,relative_sample=int(z['relative_block_return'][s-1]),finish=bool(z['support'][s-1]>r['source']['samples']+81600)) for s in event_steps(z['uint8'],.91796875)]
        assert events==r['events'] and [e for e in events if e['relative_sample']>0]==r['real_events']
        key=r['source']['voice']+'/'+r['source']['target'];g=groups.setdefault(key,dict(clips=0,hits=0));g['clips']+=1;g['hits']+=bool(r['real_events'])
    files=sorted(ROOT.iterdir());members={f.name:dict(bytes=f.stat().st_size,sha256=sha(f)) for f in files}
    archive=R/'4090-context-mixed-mac-evidence-v1.tar.gz'
    with tarfile.open(archive,'x:gz') as tar:
        for f in files:tar.add(f,arcname=f.name)
    with tarfile.open(archive,'r:gz') as tar:
        assert set(tar.getnames())==set(members)
        for m in tar:
            b=tar.extractfile(m).read();assert len(b)==members[m.name]['bytes'] and hashlib.sha256(b).hexdigest()==members[m.name]['sha256']
    report=dict(groups=groups,original_sha256=sha(ROOT/'result.json'),archive=dict(path=str(archive),bytes=archive.stat().st_size,sha256=sha(archive),members=members),limits='Consumed Mac synthetic PCM host replay, not P4 execution or human voice. No tuning or acceptance access. C161 baseline unchanged.')
    with (R/'4090-context-mixed-mac-comparison-v1.json').open('x') as f:f.write(json.dumps(report,indent=2)+'\n')
    print(json.dumps(groups))
if __name__=='__main__':main()
