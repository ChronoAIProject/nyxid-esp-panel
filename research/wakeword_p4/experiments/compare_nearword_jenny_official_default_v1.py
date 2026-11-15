"""Shared development corpus comparison with explicit native-prime limitation."""
import hashlib,json
from pathlib import Path
import numpy as np
R=Path('research/wakeword_p4/results');B=Path('.local/wakeword')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def reduce(rows,records,target):
    positives=[];shared=[]
    for row,events in zip(rows,records,strict=True):
        if row['target_id']==target:
            after=[e['sample'] for e in events if e['sample']>=row['lead_samples']]
            positives.append(dict(source_id=row['source_id'],variant=row['variant'],carrier=row['carrier'],hit=bool(after),complete=any(t>=row['generated_prefix_end_sample'] for t in after),first_signed_ms=(after[0]-row['generated_prefix_end_sample'])/16 if after else None))
        elif row['target_id'] not in ['hey_nyx','hey_willow','hi_esp']:
            shared.append(dict(source_id=row['source_id'],variant=row['variant'],events=len(events)))
    assert len(positives)==36 and len(shared)==288
    groups={}
    for name,group in [('all',positives),('command',[x for x in positives if x['carrier']!='bare'])]+[(c,[x for x in positives if x['carrier']==c]) for c in sorted({x['carrier'] for x in positives})]:
        lat=[x['first_signed_ms'] for x in group if x['hit']]
        groups[name]=dict(clips=len(group),hits=sum(x['hit'] for x in group),generated_prefix_complete=sum(x['complete'] for x in group),before_reference_first_events=sum(t<0 for t in lat),first_signed_median_ms=float(np.median(lat)) if lat else None)
    return dict(positive=groups,shared_common=dict(clips=288,activated_clips=sum(x['events']>0 for x in shared),events=sum(x['events'] for x in shared)),target_details=positives)
def main():
    manifest=B/'jenny-development-replay-v1.json';rows=json.loads(manifest.read_text())['records'];assert len(rows)==396
    cases=[]
    for folder,name in [('causal-nearword-jenny-v1','hey_nyx-v30'),('causal-nearword-jenny-v1','hey_nyx-nearword'),('causal-multivoice-jenny-v1','hey_nyx-multivoice')]:
        path=B/folder/'diagnostic'/f'{name}.jsonl';p=json.loads(path.read_text());records=[r for r in p['records'] if r['phase']=='synthetic'];assert [r['index'] for r in records]==list(range(396))
        cases.append(dict(model=name,source=str(path),source_sha256=sha(path),prime_samples=81600,threshold=.91796875,smoothing_steps=5,background=p['summary']['background'],**reduce(rows,[r['events'] for r in records],'hey_nyx')))
    path=R/'4090-nearword-jenny-official-default-behavior-v1.json';official=json.loads(path.read_text());rec=official['synthetic']['records'];assert len(rec)==396
    assert all((a['source_id'],a['variant'])==(b['source_id'],b['variant']) for a,b in zip(rows,rec,strict=True))
    cases.append(dict(model='official-wn9_hiesp-DEFAULT',source=str(path),source_sha256=sha(path),prime_samples=32256,native_identity=official['native_identity'],background=official['background'],**reduce(rows,[[dict(sample=e['sample_end']) for e in r['events']] for r in rec],'hi_esp')))
    output=dict(schema='nearword-jenny-official-default-comparison-v1',manifest_sha256=sha(manifest),models=cases,strict_matched_protocol=False,threshold_selected=False,acceptance_accessed=False,candidate_promoted=False,limits='Same original396 development corpus and full12background9.614h, shared288common negatives explicitly aligned by source/variant. Different target phrases and detector-native512/480chunk clocks; official63x512=32256zero prime versus81600candidateprime. Native default threshold is not custom threshold equivalence. Descriptive same-source reference, not strict identical deployment initialization/P4 equivalence or independent acceptance. Generated boundaries and signed pre-boundary hits are synthesis references, not human phonetic truth; early events are not an automatic veto. Correlated renders are not independent sources. No new inference or tuning in this reduction.')
    dest=R/'4090-nearword-jenny-official-default-comparison-v1.json'
    with dest.open('x') as f:f.write(json.dumps(output,indent=2)+'\n')
    for c in cases:print(c['model'],c['positive'],c['shared_common'])
if __name__=='__main__':main()
