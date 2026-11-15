"""Behavior at equal totalzero PCM warmup, original native detector outputs."""
import hashlib,json
from pathlib import Path
from compare_nearword_jenny_official_default_v1 import reduce
from hey_development_metrics_v1 import far_upper
R=Path('research/wakeword_p4/results');PLAN=R/'4090-jenny-official-matched-prime-plan-v1.json'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    p=json.loads(PLAN.read_text());base=Path(p['output']);c=json.loads((base/'complete.json').read_text());assert c['streams']==408 and c['plan_sha256']==sha(PLAN)
    rows=[];events=[];background=[];receipts=[];identities=set();prime_events=0;cross_clips=cross_hits=cross_events=0
    for i,row in enumerate(p['records']):
        path=base/f'{i:03}-report.json';r=json.loads(path.read_text());assert r['record']==row and r['execution_sha256']==sha(PLAN)
        assert r['source_input_samples']==row['samples'] and r['source_pcm_sha256']==row['pcm_sha256'] and r['input_samples']==row['samples']+49344
        assert r['total_prime_samples']==81600 and r['additional_prime_samples']==49344 and r['continuous_model'] and not r['transition_reset']
        assert sha(base/f'{i:03}-raw-returns.bin')==r['raw_returns_sha256'] and sha(base/f'{i:03}-serial.log')==r['serial_sha256']
        assert r['input_calls']==(r['input_samples']+511)//512
        identities.add((r['mode'],r['actual_float32_bits'],r['actual_threshold']))
        original_events=[dict(sample=e['sample_end']-49344) for e in r['events'] if e['sample_end']>49344]
        assert [x['sample'] for x in original_events]==[e['relative_sample'] for e in r['real_events']]
        prime_events+=len(r['additional_priming_events'])+sum(result==1 for call,result in r['prime_returns'])
        if row['phase']=='synthetic':
            rows.append(row);events.append(original_events)
            if row['target_id'] in ['hey_nyx','hey_willow']:
                cross_clips+=1;cross_hits+=bool(original_events);cross_events+=len(original_events)
        else:background.append(dict(index=i,samples=row['samples'],events=len(original_events)))
        receipts.append(dict(index=i,report_sha256=sha(path)))
    assert len(identities)==1 and len(rows)==396 and len(background)==12 and cross_clips==72
    hours=sum(x['samples'] for x in background)/16000/3600;count=sum(x['events'] for x in background)
    out=dict(schema='jenny-official-matched-prime-behavior-v1',plan_sha256=sha(PLAN),native_identity=list(identities)[0],total_prime_samples=81600,priming_events_ignored_by_application=prime_events,**reduce(rows,events,'hi_esp'),cross_target=dict(clips=cross_clips,activated_clips=cross_hits,events=cross_events),background=dict(events=count,hours=hours,far_upper95=far_upper(count,hours),records=background),reports=receipts,threshold_selected=False,acceptance_accessed=False,candidate_promoted=False,limits=p['policy'])
    path=R/'4090-jenny-official-matched-prime-behavior-v1.json'
    with path.open('x') as f:f.write(json.dumps(out,indent=2)+'\n')
    print(json.dumps(dict(positive=out['positive'],shared_common=out['shared_common'],background=out['background'])))
if __name__=='__main__':main()
