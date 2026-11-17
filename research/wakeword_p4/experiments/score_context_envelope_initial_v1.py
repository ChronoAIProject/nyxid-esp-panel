"""Rescore every saved raw code, archive originals and verify all members."""
from collections import defaultdict
import hashlib,json
from pathlib import Path
import tarfile
import numpy as np
from uint8_event_policy_v1 import event_steps
R=Path('research/wakeword_p4/results')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    summary={};members={};originals={}
    for cohort,expected in [('builtin',36),('daily',96),('continuous',2)]:
        root=Path(f'.local/wakeword/context-envelope-{cohort}-replay-v1');source=root/'result.json'
        p=json.loads(source.read_text());assert p['completed']
        plan=R/f'4090-context-envelope-{cohort}-replay-plan-v1.json';assert p['plan_sha256']==sha(plan)
        originals[str(source)]=sha(source)
        rows=p['results'] if cohort=='continuous' else p['records'];assert len(rows)==expected
        groups=defaultdict(lambda:dict(clips=0,activated=0));activated=[];continuous=[]
        for row in rows:
            v=row['models']['context-envelope']
            trace=Path(v['trace_path']) if cohort=='continuous' else root/f'{row["index"]:03d}-context-envelope.npz'
            assert sha(trace)==v['trace_sha256']
            with np.load(trace) as z:
                assert z['uint8'].dtype==np.uint8
                events=[dict(step=s,relative_sample=int(z['relative_block_return'][s-1])) for s in event_steps(z['uint8'],.91796875)]
            assert [(e['step'],e['relative_sample']) for e in events]==[(e['step'],e['relative_sample']) for e in v['events']]
            if cohort=='continuous':
                hits=defaultdict(set);other=[]
                for e in events:
                    owner=next(t for t in row['timeline'] if t['start']<e['relative_sample']<=t['end'])
                    saved=next(t for t in v['events'] if t['step']==e['step']);assert saved['owner']==owner
                    if owner['kind']=='clip':hits[owner['group']].add(owner['index'])
                    else:other.append(dict(event=e,owner=owner))
                continuous.append(dict(gap_samples=row['gap_samples'],target_containers=len(hits['nyx']),negative_containers=len(hits['everyday-negative']),willow_containers=len(hits['willow']),outside_clip_events=other,pcm_sha256=row['pcm_sha256']))
            else:
                real=[e for e in events if e['relative_sample']>0];assert real==v['real_events']
                g=groups[row['request']['group']];g['clips']+=1;g['activated']+=bool(real)
                if real:activated.append(dict(index=row['index'],request=row['request'],events=real))
        summary[cohort]=continuous if cohort=='continuous' else dict(groups=dict(groups),activated=activated)
        for f in sorted(root.iterdir()):members[f'{cohort}/{f.name}']=dict(path=str(f),bytes=f.stat().st_size,sha256=sha(f))
    archive=R/'4090-context-envelope-initial-evidence-v1.tar.gz'
    with tarfile.open(archive,'x:gz') as tar:
        for name,m in members.items():tar.add(m['path'],arcname=name)
    with tarfile.open(archive,'r:gz') as tar:
        assert set(tar.getnames())==set(members)
        for m in tar:
            b=tar.extractfile(m).read();assert len(b)==members[m.name]['bytes'] and hashlib.sha256(b).hexdigest()==members[m.name]['sha256']
    report=dict(schema='context-envelope-initial-comparison-v1',actual_replay_exit_codes=dict(builtin=0,daily=0,continuous=0),originals=originals,summary=summary,archive=dict(path=str(archive),bytes=archive.stat().st_size,sha256=sha(archive),members=members),limits='Consumed correlated synthetic development only. Container events not phonetic truth or first-command retention. Fixed profile, no tuning, baseline promotion or acceptance access.')
    with (R/'4090-context-envelope-initial-comparison-v1.json').open('x') as f:f.write(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(summary=summary,archive_members=len(members))))
if __name__=='__main__':main()
