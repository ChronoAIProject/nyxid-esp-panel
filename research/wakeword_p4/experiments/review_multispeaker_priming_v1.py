import json,hashlib
from pathlib import Path
import numpy as np
root=Path('.local/wakeword/multispeaker-priming-v1');checked=0

def independent(codes,threshold):
    cumulative=np.concatenate(([0],np.cumsum(codes,dtype=np.int64)))
    sums=cumulative[5:]-cumulative[:-5]
    high=np.flatnonzero(sums>=threshold*1280)+5
    low=np.flatnonzero(sums<threshold*.5*1280)+5
    events=[];start=5
    while True:
        at=np.searchsorted(high,start)
        if at==len(high):return events
        step=int(high[at]);events.append(step)
        at=np.searchsorted(low,step+67)
        if at==len(low):return events
        start=int(low[at])
for name in ['hey_nyx-narrow','hey_nyx-broad']:
    records=json.loads((root/name/'report.json').read_text())['records'];raws=[]
    for rec in records:
        with np.load(rec['trace_path']) as a:raws.append(a['uint8'].copy())
    lines=(root/'diagnostic'/f'{name}.jsonl').read_text().splitlines()
    assert len(lines)==138
    for line in lines:
        r=json.loads(line);summary=r['summary'];threshold=summary['threshold'];primecount=0;bg=0;hours=0.;common=0;cross=0;hits=0;barehits=0
        assert len(r['records'])==372
        for rec,codes,item in zip(records,raws,r['records'],strict=True):
            row=rec['record'];total=row['samples']+81600;steps=independent(codes,threshold)
            pairs=[(s,min(((s*3+2)*160+479)//480*480,total)-81600) for s in steps]
            prime=[(s,t) for s,t in pairs if t<=0];real=[(s,t) for s,t in pairs if t>0]
            assert [(d['step'],d['relative_sample']) for d in item['priming_events']]==prime
            assert [(d['step'],d['sample']) for d in item['events']]==real
            for d in item['events']:
                assert d['support_relative_sample']==(d['step']*3+2)*160-81600
                assert d['finish_call']==(((d['step']*3+2)*160+479)//480*480>total)
            primecount+=len(prime)
            if rec['phase']=='background':bg+=len(real);hours+=row['samples']/16000/3600
            elif row['target_id']=='hey_nyx':
                after=[t for s,t in real if t>=row['lead_samples']]
                complete=any(t>=row['generated_prefix_end_sample'] for t in after)
                assert item['positive']['complete']==complete
                assert item['positive']['any_after_source']==bool(after)
                assert item['positive']['latency_ms']==((after[0]-row['generated_prefix_end_sample'])/16 if after else None)
                hits+=complete;barehits+=bool(complete and row['carrier']=='')
            elif row['target_id']=='hey_willow':cross+=bool(real)
            else:common+=bool(real)
        assert summary['priming_events_ignored_by_app']==primecount and summary['background']['events']==bg and summary['background']['hours']==hours
        assert summary['positive']['all']['complete']==hits and summary['positive']['standalone']['complete']==barehits
        assert summary['negative']['common']['activated_clips']==common and summary['negative']['cross_target']['activated_clips']==cross
        checked+=1
    print(name,'138 points independent event/prime/app/whole exposure/positive/common/cross/latency pass',flush=True)
path=root/'diagnostic/summary.json'
report=dict(schema='multispeaker-priming-scoring-review-v1',points=checked,all276_independent_integer_cumulative_sum_event_and_app_clock_checks_passed=True,summary_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),reviewer_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
Path('research/wakeword_p4/results/4090-multispeaker-priming-scoring-review-v1.json').write_text(json.dumps(report,indent=2)+'\n')
