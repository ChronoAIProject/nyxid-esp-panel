"""Freeze cell/slot choices from original authenticated traces, no new audio."""
import gzip,hashlib,json
from pathlib import Path
from common_carrier_slot_policy_v1 import Decisions

R=Path('research/wakeword_p4/results');BASE=Path('.local/wakeword')
OUT=BASE/'hey-common-carrier-replacement-decisions-v1'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    rp=R/'4090-hey-broad-surviving-window-census-v2.json';ap=R/'4090-hey-broad-surviving-window-independent-audit-v1.json'
    result=json.loads(rp.read_text());audit=json.loads(ap.read_text());assert audit['all_passed'] and audit['original_result_sha256']==sha(rp)
    nativeplan=R/'4090-hey-common-carrier-kokoro-generation-execution-v1.json';p=json.loads(nativeplan.read_text())
    policy=Path(__file__).with_name('common_carrier_slot_policy_v1.py');assert sha(policy)==p['files'][str(policy.resolve())]
    cells={(f'kokoro-negative-{r["request"]["anchor_order"]:05}',r['request']['carrier']):r['request'] for r in p['requests']};assert len(cells)==1620
    OUT.mkdir(exist_ok=False);reports=[];shared=None
    for target in result['targets']:
        word=target['target'];root=BASE/'hey-broad-training-features-v1'/word
        rows=json.loads((root/'features.json').read_text());trace=Path(target['trace_path']);assert sha(trace)==target['trace_sha256']
        schedule=BASE/'hey-broad-schedules-v1'/(word+'.jsonl');dest=OUT/(word+'.jsonl.gz');decisions=Decisions();eligible=selected=0;identity=hashlib.sha256()
        with gzip.open(trace,'rt') as source,schedule.open() as s,gzip.open(dest,'xt') as output:
            for index,line in enumerate(source,1):
                batch=json.loads(line);wanted=json.loads(next(s));assert batch['step']==wanted['index']==index and batch['batch_sha256']==wanted['tensor_sha256']
                choices=[]
                for slot,rec in enumerate(batch['survivors']):
                    if rec['kind']!='ordinary' or rec['label']!=0:continue
                    row=rows[rec['row']]
                    if row['engine']!='kokoro' or row['target_id']!='common_negative':continue
                    eligible+=1;replace,carrier=decisions.next();req=cells[row['source_id'],carrier]
                    assert row['source_pair_key']['conditioning']==req['conditioning_id'] and row['source_pair_key']['scale']==req['speed'] and row['text']==req['anchor_text']
                    lo=max(225,row['lead_frame']-25);n=row['frames']-lo+1;assert lo<=rec['end']<=row['frames']
                    decision=dict(slot=slot,replace=replace,carrier=carrier,anchor_id=row['source_id'],variant=row['variant'],native_order=req['order'],
                                  old_end=rec['end'],old_frames=row['frames'],old_lead=row['lead_frame'],crop_midpoint_numerator=2*(rec['end']-lo)+1,crop_midpoint_denominator=2*n)
                    choices.append(decision);selected+=replace
                identity.update((json.dumps(dict(step=index,choices=choices),sort_keys=True,separators=(',',':'))+'\n').encode())
                output.write(json.dumps(dict(step=index,original_batch_sha256=batch['batch_sha256'],choices=choices),separators=(',',':'))+'\n')
        assert index==10000 and eligible==74219
        if shared is None:shared=identity.hexdigest()
        else:assert identity.hexdigest()==shared
        reports.append(dict(target=word,batches=index,eligible_slots=eligible,selected_slots=selected,decision_identity_sha256=identity.hexdigest(),path=str(dest),sha256=sha(dest),trace_sha256=sha(trace)))
    out=dict(source_sha256=sha(__file__),policy_sha256=sha(policy),native_execution_sha256=sha(nativeplan),original_census_sha256=sha(rp),original_independent_audit_sha256=sha(ap),targets=reports,
             cell_decisions_identical_between_words=True,model_invocations=0,audio_decodes=0,
             limits='Prospective cell/slot choices only, not actual replacement tensor schedule. New frame lengths/endpoints/features/native/rendered quality still pending; preserve every original nonselected row/label/weight. No model training, freshdev/acceptance or claimed recognition improvement.')
    with (R/'4090-hey-common-carrier-replacement-decisions-v1.json').open('x') as f:f.write(json.dumps(out,indent=2)+'\n')
    print(json.dumps(out))
if __name__=='__main__':main()
