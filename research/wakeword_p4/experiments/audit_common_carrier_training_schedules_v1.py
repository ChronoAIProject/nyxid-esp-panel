"""Independent full coordinate reconstruction of four training schedules."""
import gzip,hashlib,json
from pathlib import Path
import numpy as np

R=Path('research/wakeword_p4/results');BASE=Path('.local/wakeword')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def digest(x,y,w):
    h=hashlib.sha256()
    for a in (x,y,w):h.update(a.tobytes())
    return h.hexdigest()
def main():
    reportpath=R/'4090-hey-common-carrier-training-schedules-v1.json'
    if not reportpath.exists():print(json.dumps(dict(status='not_ready',missing=str(reportpath))));return 2
    report=json.loads(reportpath.read_text());execution=R/'4090-hey-common-carrier-training-schedule-execution-v1.json';assert sha(execution)==report['execution_sha256']
    for f,h in json.loads(execution.read_text())['files'].items():assert sha(f)==h
    census=json.loads((R/'4090-hey-broad-surviving-window-census-v2.json').read_text());choices=json.loads((R/'4090-hey-common-carrier-replacement-decisions-v1.json').read_text())
    root=BASE/'hey-common-carrier-features-v1';newrows=json.loads((root/'features.json').read_text());new={(r['anchor_id'],r['carrier'],r['variant']):r for r in newrows};assert len(new)==4860
    newraw=np.memmap(root/'features.bin',dtype='<u2',mode='r').reshape(-1,40);results=[]
    for word in ('hey_willow','hey_nyx'):
        oldroot=BASE/'hey-broad-training-features-v1'/word;oldrows=json.loads((oldroot/'features.json').read_text())
        provenance=json.loads((oldroot/'target-provenance.json').read_text())
        for n,h in provenance['files'].items():assert sha(oldroot/n)==h
        raw=np.memmap(oldroot/'features.bin',dtype='<u2',mode='r').reshape(-1,40)
        metadata=sum([json.loads((oldroot/n).read_text()) for n in ('human.json','ambient.json')],[]);ambient=[np.load(oldroot/r['path'],mmap_mode='r') for r in metadata]
        original=next(t for t in census['targets'] if t['target']==word);decision=next(t for t in choices['targets'] if t['target']==word)
        assert sha(original['trace_path'])==original['trace_sha256'] and sha(decision['path'])==decision['sha256']
        cases={c['condition']:c for c in report['conditions'] if c['target']==word};assert set(cases)=={'control','command_context'}
        for c in cases.values():assert sha(c['path'])==c['sha256']
        selected=0;count=0
        with gzip.open(original['trace_path'],'rt') as traces,gzip.open(decision['path'],'rt') as fixed,open(cases['control']['path']) as control,open(cases['command_context']['path']) as treatment:
            for number,line in enumerate(traces,1):
                t=json.loads(line);d=json.loads(next(fixed));c=json.loads(next(control));v=json.loads(next(treatment))
                assert t['step']==d['step']==c['step']==v['step']==number
                xs=[];labels=[];weights=[];eligible=[]
                for slot,rec in enumerate(t['survivors']):
                    start,end=rec['start'],rec['end'];assert end-start==225
                    if rec['kind']=='ambient':data=ambient[rec['ambient_index']][start:end]
                    else:
                        row=oldrows[rec['row']];assert row['offset']==rec['offset'];at=row['offset'];data=raw[at+start:at+end]
                        if rec['kind']=='ordinary' and row['engine']=='kokoro' and row['target_id']=='common_negative':eligible.append(slot)
                    xs.append(np.clip(data.astype(np.float32)/25.6,0,26));labels.append([rec['label']]);weights.append(1 if rec['label'] else 20)
                x=np.asarray(xs,np.float32);y=np.asarray(labels,np.float32);w=np.asarray(weights,np.float32)
                original_sha=digest(x,y,w);assert original_sha==t['batch_sha256']==d['original_batch_sha256']==c['original_batch_sha256']==v['original_batch_sha256']==c['tensor_sha256']
                assert [a['slot'] for a in d['choices']]==eligible
                changed=x.copy();expected=[]
                for choice in d['choices']:
                    slot=choice['slot'];old=oldrows[t['survivors'][slot]['row']]
                    assert choice['anchor_id']==old['source_id'] and choice['variant']==old['variant'] and choice['old_end']==t['survivors'][slot]['end']
                    lo=max(225,old['lead_frame']-25);n=old['frames']-lo+1
                    assert choice['crop_midpoint_numerator']==2*(choice['old_end']-lo)+1 and choice['crop_midpoint_denominator']==2*n
                    if not choice['replace']:continue
                    row=new[choice['anchor_id'],choice['carrier'],choice['variant']]
                    assert row['native_order']==choice['native_order'] and row['conditioning_id']==old['conditioning_id'] and row['speed']==old['source_pair_key']['scale'] and y[slot,0]==0 and w[slot]==20
                    newlo=max(225,row['lead_frame']-25);newn=row['frames']-newlo+1;end=newlo+(choice['crop_midpoint_numerator']*newn)//choice['crop_midpoint_denominator'];assert newlo<=end<=row['frames']
                    at=row['offset'];changed[slot]=np.clip(newraw[at+end-225:at+end].astype(np.float32)/25.6,0,26)
                    expected.append(dict(slot=slot,new_order=row['order'],new_end=end));selected+=1
                assert expected==c['selected']==v['selected'] and digest(changed,y,w)==v['tensor_sha256']
                assert c['nonselected_rows_exact'] and v['nonselected_rows_exact'] and c['labels_weights_exact'] and v['labels_weights_exact']
                count=number
                if number%1000==0:print(json.dumps(dict(target=word,batches=number)),flush=True)
            assert fixed.readline()==control.readline()==treatment.readline()==''
        assert count==10000 and selected==37114
        results.append(dict(target=word,batches=count,selected_windows=selected,all_original_and_both_condition_hashes_exact=True,unchanged_labels_weights_and_unselected_rows=True,all_quantile_endpoints_exact=True))
    result=dict(all_passed=True,original_result_sha256=sha(reportpath),source_sha256=sha(__file__),targets=results,model_invocations=0,sampler_invocations=0,limits='Complete saved-coordinate reconstruction; shared NumPy math, no model efficacy/recognition/fresh data or independent phonetic truth.')
    with (R/'4090-hey-common-carrier-training-schedule-independent-audit-v1.json').open('x') as f:f.write(json.dumps(result,indent=2)+'\n')
    return 0
if __name__=='__main__':raise SystemExit(main())
