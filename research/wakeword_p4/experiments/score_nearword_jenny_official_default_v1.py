"""Summarize native official DEFAULT returns without threshold selection."""
import json,hashlib,sys
from pathlib import Path
from calibrate_complete_official_v1 import metrics
from hey_development_metrics_v1 import far_upper
R=Path('research/wakeword_p4/results');PLAN=R/'4090-nearword-jenny-official-default-plan-v1.json'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    p=json.loads(PLAN.read_text());base=Path(p['output']);complete=json.loads((base/'complete.json').read_text());assert complete['streams']==408 and complete['plan_sha256']==sha(PLAN)
    clips=[];background=[];identities=set();receipts=[]
    for i,row in enumerate(p['records']):
        path=base/f'{i:03}-report.json';r=json.loads(path.read_text());assert r['record']==row and r['execution_sha256']==sha(PLAN) and r['input_samples']==row['samples'] and r['model']=='wn9_hiesp'
        assert sha(base/f'{i:03}-raw-returns.bin')==r['raw_returns_sha256'];assert sha(base/f'{i:03}-serial.log')==r['serial_sha256']
        identities.add((r['mode'],r['actual_float32_bits'],r['actual_threshold']))
        if row['phase']=='synthetic':clips.append((row,r['events']))
        else:background.append(dict(index=i,samples=row['samples'],events=len(r['events'])))
        receipts.append(dict(index=i,report_sha256=sha(path)))
    assert len(identities)==1 and len(clips)==396 and len(background)==12
    point=metrics(clips,'hiesp',[r for r in p['records'] if r['phase']=='synthetic'],['bare']);hours=sum(r['samples'] for r in background)/16000/3600;events=sum(r['events'] for r in background)
    result=dict(schema='nearword-jenny-official-default-behavior-v1',plan_sha256=sha(PLAN),native_identity=list(identities)[0],synthetic=point,background=dict(events=events,hours=hours,far_upper95=far_upper(events,hours),records=background),reports=receipts,threshold_selected=False,acceptance_accessed=False,candidate_promoted=False,limits=p['policy'])
    dest=R/'4090-nearword-jenny-official-default-behavior-v1.json'
    with dest.open('x') as f:f.write(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(positive=point['complete_phrase_activated'],any_after_source=point['any_after_source_recall'],negative=point['negatives'],background=result['background'])))
if __name__=='__main__':main()
