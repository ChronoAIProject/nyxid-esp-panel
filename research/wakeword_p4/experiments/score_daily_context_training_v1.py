"""Verify every training-fit trace and archive member; no inference or tuning."""
import hashlib,json,sys,tarfile
from pathlib import Path
import numpy as np
from uint8_event_policy_v1 import event_steps
from replay_daily_context_training_v1 import lock,output,parent

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main(c):
    root=output(c);plan=lock(c);p=json.loads(plan.read_text());source=root/'result.json';r=json.loads(source.read_text())
    assert r['training_diagnostic'] and r['plan_sha256']==sha(plan)
    assert len(r['records'])==len(p['rows'])==dict(old84=84,new171=171)[c]
    groups={}
    for row,record in zip(p['rows'],r['records'],strict=True):
        assert record['index']==row['index'] and record['request']==row['request'] and record['condition']=='daily-context'
        trace=Path(record['trace']['path']);assert trace.stat().st_size==record['trace']['bytes'] and sha(trace)==record['trace']['sha256']
        with np.load(trace) as z:
            assert z['uint8'].dtype==np.uint8
            import soundfile as sf
            samples=sf.info(row['wav']['path']).frames
            events=[dict(step=s,relative_sample=int(z['relative_block_return'][s-1]),finish=bool(z['support'][s-1]>samples+81600)) for s in event_steps(z['uint8'],.91796875)]
        assert events==record['events'] and [e for e in events if e['relative_sample']>0]==record['real_events']
        key=row['request']['voice']+'/'+row['request'].get('group',str(row['request']['requested_label'])+'/'+row['request']['text']);g=groups.setdefault(key,dict(clips=0,activated=0));g['clips']+=1;g['activated']+=bool(record['real_events'])
    files=sorted(f for f in root.iterdir() if f.is_file());members={f.name:dict(bytes=f.stat().st_size,sha256=sha(f)) for f in files}
    archive=parent.R/f'4090-daily-context-training-{c}-evidence-v1.tar.gz'
    with tarfile.open(archive,'x:gz') as tar:
        for f in files:tar.add(f,arcname=f.name)
    with tarfile.open(archive,'r:gz') as tar:
        assert set(tar.getnames())==set(members)
        for m in tar:
            b=tar.extractfile(m).read();assert len(b)==members[m.name]['bytes'] and hashlib.sha256(b).hexdigest()==members[m.name]['sha256']
    result=dict(cohort=c,plan_sha256=sha(plan),original_sha256=sha(source),groups=groups,archive=dict(path=str(archive),bytes=archive.stat().st_size,sha256=sha(archive),members=members),limits='Complete admitted consumed training-fit only, no generalization or phonetic truth. Fixed profile, no threshold/checkpoint/calibration selection. No promotion.')
    with (parent.R/f'4090-daily-context-training-{c}-comparison-v1.json').open('x') as f:f.write(json.dumps(result,indent=2)+'\n')
    print(json.dumps(groups))
if __name__=='__main__':main(sys.argv[1])
