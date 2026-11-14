"""Independent saved-UINT8 event and complete-exposure scientific report review."""
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.stats import chi2
ROOT=Path('.local/wakeword/causal-sequence-formal-development-v1')
R=Path('research/wakeword_p4/results')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def events(raw,threshold,frozen):
    if frozen:
        means=np.convolve(raw.astype(np.float32)/256,np.ones(5)/5,mode='valid')
        high=np.flatnonzero(means>=threshold)+5;low=np.flatnonzero(means<threshold*.5)+5
    else:
        cumulative=np.concatenate(([0],np.cumsum(raw,dtype=np.int64)));sums=cumulative[5:]-cumulative[:-5]
        high=np.flatnonzero(sums>=threshold*1280)+5;low=np.flatnonzero(sums<threshold*.5*1280)+5
    found=[];start=5
    while True:
        at=np.searchsorted(high,start)
        if at==len(high):return found
        step=int(high[at]);found.append(step);at=np.searchsorted(low,step+67)
        if at==len(low):return found
        start=int(low[at])
def main():
    result=json.loads((ROOT/'result.json').read_text());assert result['all_passed'];points=[];traces_checked=0
    for model in result['models']:
        name=model['model'];report_path=ROOT/name/'report.json';assert sha(report_path)==model['report_sha256'];report=json.loads(report_path.read_text());records=report['records'];assert len(records)==372;raws=[]
        for rec in records:
            assert sha(rec['trace_path'])==rec['trace_sha256'] and rec['repeat_exact']
            with np.load(rec['trace_path']) as a:
                raw=a['uint8'].copy();assert raw.dtype==np.uint8 and np.array_equal(raw,a['uint8_repeat'])
                assert np.array_equal(a['probability'],raw.astype(np.float32)/256)
                samples=rec['record']['samples'];assert len(raw)==(((samples+159)//160-2)+2)//3
                support=(np.arange(1,len(raw)+1)*3+2)*160
                assert np.array_equal(a['support_unclamped'],support) and np.array_equal(a['sample_end'],np.minimum(support,samples))
            raws.append(raw);traces_checked+=1
        lines=(ROOT/'diagnostic'/f'{name}.jsonl').read_text().splitlines();assert len(lines)==(139 if name=='hey_nyx-v30-frozen' else 138)
        for line in lines:
            row=json.loads(line);s=row['summary'];threshold=s['threshold'];frozen=s['frozen_parameters_reference'];assert not frozen or (name=='hey_nyx-v30-frozen' and threshold==.91796875)
            bg=0;hours=0.;common=0;cross=0;common_events=0;cross_events=0;complete=0;bare=0;pre=0;positive_lat=[];all_missing=0
            for rec,raw,item in zip(records,raws,row['records'],strict=True):
                src=rec['record'];steps=events(raw,threshold,frozen);times=[min((x*3+2)*160,src['samples']) for x in steps]
                assert [(e['step'],e['sample']) for e in item['events']]==list(zip(steps,times,strict=True))
                if rec['phase']=='synthetic':
                    for e in item['events']:
                        for boundary in ['generated_prefix_end_sample','full_rir_prefix_end_sample','generated_container_end_sample','full_rir_container_end_sample']:assert e[boundary+'_signed_ms']==(e['sample']-src[boundary])/16
                if rec['phase']=='background':bg+=len(times);hours+=src['samples']/16000/3600
                elif src['target_id']=='hey_nyx':
                    after=[t for t in times if t>=src['lead_samples']];hit=any(t>=src['generated_prefix_end_sample'] for t in after);lat=(after[0]-src['generated_prefix_end_sample'])/16 if after else None
                    assert item['positive']['complete']==hit and item['positive']['latency_ms']==lat and item['positive']['any_after_source']==bool(after)
                    complete+=hit;bare+=bool(hit and not src['carrier']);pre+=sum(t<src['lead_samples'] for t in times);all_missing+=not after
                    if lat is not None:positive_lat.append(lat)
                elif src['target_id']=='hey_willow':cross+=bool(times);cross_events+=len(times)
                else:common+=bool(times);common_events+=len(times)
            assert s['background']['events']==bg and s['background']['hours']==hours
            assert s['background']['far_upper95']==float(.5*chi2.ppf(.95,2*(bg+1))/hours)
            assert s['negative']['common']['activated_clips']==common and s['negative']['common']['events']==common_events
            assert s['negative']['cross_target']['activated_clips']==cross and s['negative']['cross_target']['events']==cross_events
            assert s['positive']['all']['complete']==complete and s['positive']['standalone']['complete']==bare and s['pre_source_events']==pre and s['positive']['all']['missing']==all_missing
            latency=s['positive']['all']['signed_latency_ms'];assert latency['observed']==len(positive_lat)
            assert latency['median']==(float(np.median(positive_lat)) if positive_lat else None)
            assert latency['p95']==(float(np.percentile(positive_lat,95)) if positive_lat else None)
            points.append(dict(model=name,threshold=threshold,frozen=frozen))
        print(json.dumps(dict(model=name,traces=372,points=len(lines),all_behavior_checks_passed=True)),flush=True)
    assert traces_checked==744 and len(points)==277
    audit=dict(schema='causal-sequence-complete-behavior-review-v1',trace_count=744,point_count=277,all_trace_hashes_repeat_codes_clocks_events_exposure_hits_confusion_latency_and_far_passed=True,summary_sha256=sha(ROOT/'diagnostic/summary.json'),source_sha256=sha(__file__),scope='Independent integer cumulative-sum/searchsorted event reconstruction and original v30 convolution point on saved actual traces; no float-model comparison or qualification gate.')
    with (R/'4090-causal-sequence-formal-behavior-review-v1.json').open('x') as f:f.write(json.dumps(audit,indent=2)+'\n')
if __name__=='__main__':main()
