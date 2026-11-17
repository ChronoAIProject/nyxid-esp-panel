"""Verify appended-tail raw events and original genuine-support prefix equality."""
import json,hashlib,tarfile
from pathlib import Path
import numpy as np
from replay_bare_support_fixed_tail_v1 import PLAN,OUT,R,sha
from uint8_event_policy_v1 import event_steps

def main():
    p=json.loads((OUT/'result.json').read_text());assert p['plan_sha256']==sha(PLAN)
    original=Path('.local/wakeword/bare-support-training-bare23-replay-v1/result.json');old={r['index']:r for r in json.loads(original.read_text())['records']};rows=[]
    for r in p['records']:
        assert sha(r['trace']['path'])==r['trace']['sha256'];o=old[r['index']];assert sha(o['trace']['path'])==o['trace']['sha256']
        with np.load(r['trace']['path']) as z,np.load(o['trace']['path']) as before:
            events=[dict(step=s,relative_sample=int(z['relative_block_return'][s-1])) for s in event_steps(z['uint8'],.91796875)]
            assert events==r['events']
            n=int(np.count_nonzero(before['support']<=81600+r['original_samples']))
            assert np.array_equal(z['uint8'][:n],before['uint8'][:n]) and np.array_equal(z['support'][:n],before['support'][:n])
        rows.append(dict(index=r['index'],original_hit=bool(o['real_events']),tail_hit=bool(r['real_events']),after_original_end_events=r['after_original_end_events'],real_events=r['real_events'],genuine_support_calls_identical=n))
    files=sorted(OUT.iterdir());members={f.name:dict(bytes=f.stat().st_size,sha256=sha(f)) for f in files};archive=R/'4090-bare-support-fixed-tail-evidence-v1.tar.gz'
    with tarfile.open(archive,'x:gz') as tar:
        for f in files:tar.add(f,arcname=f.name)
    with tarfile.open(archive,'r:gz') as tar:
        assert set(tar.getnames())==set(members)
        for m in tar:
            b=tar.extractfile(m).read();assert len(b)==members[m.name]['bytes'] and hashlib.sha256(b).hexdigest()==members[m.name]['sha256']
    result=dict(actual_replay_exit_code=0,original_result_sha256=sha(original),rows=rows,archive=dict(path=str(archive),bytes=archive.stat().st_size,sha256=sha(archive),members=members),policy=p['policy'])
    with (R/'4090-bare-support-fixed-tail-comparison-v1.json').open('x') as f:f.write(json.dumps(result,indent=2)+'\n')
    print(json.dumps([r for r in rows if not r['original_hit']]))
if __name__=='__main__':main()
