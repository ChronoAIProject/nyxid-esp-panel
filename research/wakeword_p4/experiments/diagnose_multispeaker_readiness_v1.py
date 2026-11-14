"""Consumed-only explicit readiness state machine on immutable saved UINT8 streams."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import numpy as np
from scipy.stats import chi2
from uint8_event_policy_v1 import event_steps
from wake_readiness_policy_v1 import ready_event_steps

R=Path('research/wakeword_p4/results')
SOURCE=Path('.local/wakeword/multispeaker-formal-development-v1')
OUT=Path('.local/wakeword/multispeaker-readiness-diagnostic-v1')
LOCK=R/'4090-multispeaker-readiness-diagnostic-plan-v1.json'
PROTOCOL=Path(__file__).with_name('multispeaker_readiness_diagnostic_v1.md')
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def upper(events,hours):
    return float(.5*chi2.ppf(.95,2*(events+1))/hours) if hours>0 else None

def plan():
    collection=R/'4090-multispeaker-formal-development-execution-v1.json'
    scoring=R/'4090-multispeaker-formal-scoring-independent-audit-v1.json'
    parent=json.loads(collection.read_text());audit=json.loads(scoring.read_text())
    assert audit['all_passed'] and audit['points']==276
    paths=[Path(__file__),PROTOCOL,collection,scoring,SOURCE/'diagnostic/summary.json',
           R/'4090-multispeaker-formal-export-plan-v1.json',
           Path(__file__).with_name('uint8_event_policy_v1.py'),Path(__file__).with_name('wake_readiness_policy_v1.py'),
           R/'4090-multispeaker-formal-collection-independent-audit-v1.json']
    assert sha(SOURCE/'diagnostic/summary.json')==audit['summary_sha256']
    for model in parent['models']:
        name=model['target']+'-'+model['condition'];report=SOURCE/name/'report.json'
        paths.append(report)
        for rec in json.loads(report.read_text())['records']:
            assert sha(rec['trace_path'])==rec['trace_sha256'];paths.append(Path(rec['trace_path']))
    export=json.loads((R/'4090-multispeaker-formal-export-plan-v1.json').read_text());assert export['context_frames']==309
    p=dict(schema='multispeaker-readiness-diagnostic-plan-v1',files={str(p):sha(p) for p in paths},
           versions={n:importlib.metadata.version(n) for n in ['numpy','scipy']},
           models=parent['models'],grid=parent['grid'],context_frames=309,stride_frames=3,
           first_valid_call=103,discard_score_count=102,smoothing_steps=5,first_ready_call=107,
           first_ready_support_sample=(107*3+2)*160,hop_samples=160,
           policy='Explicit new system readiness diagnostic only; re-run mean/arming/cooldown state from rawscore index102. Earliest detectioncall107/sample51680. Same complete streams, once at actual start only. Original curves/outputbytes kept; no filtered event lists, internal resets, chosen duration/threshold or freshdata.')
    assert p['first_valid_call']==(p['context_frames']+2)//3
    assert p['first_ready_call']==p['first_valid_call']+p['smoothing_steps']-1
    with LOCK.open('x') as f:f.write(json.dumps(p,indent=2)+'\n')
    print(json.dumps(dict(plan_sha256=sha(LOCK),first_ready_sample=p['first_ready_support_sample'],first_ready_seconds=p['first_ready_support_sample']/16000)))

def run():
    assert subprocess.check_output(['git','show','origin/research-hye-nyx-voice-gpu:'+str(LOCK)])==LOCK.read_bytes()
    p=json.loads(LOCK.read_text())
    for path,h in p['files'].items():assert sha(path)==h,path
    for package,v in p['versions'].items():assert importlib.metadata.version(package)==v
    subprocess.run(['/usr/bin/python3','research/wakeword_p4/tools/check_disk_budget.py','--expected-gib','0.25'],check=True)
    OUT.mkdir(exist_ok=False);points=[];trace_receipts=[]
    old=json.loads((SOURCE/'diagnostic/summary.json').read_text())['points']
    skip=p['discard_score_count'];ready_sample=p['first_ready_support_sample']
    for model in p['models']:
        name=model['target']+'-'+model['condition']
        report=json.loads((SOURCE/name/'report.json').read_text());traces=[]
        for rec in report['records']:
            with np.load(rec['trace_path'],allow_pickle=False) as a:raw=a['uint8'].copy()
            assert raw.dtype==np.uint8 and raw.ndim==1
            traces.append((rec,raw))
            trace_receipts.append(dict(model=name,phase=rec['phase'],index=rec['index'],trace_path=rec['trace_path'],trace_sha256=sha(rec['trace_path']),
                pre_ready_uint8=raw[:min(len(raw),p['first_ready_call']-1)].tolist(),valid_buffer_uint8=raw[skip:min(len(raw),p['first_ready_call']-1)].tolist(),
                ready_sample_unclamped=ready_sample,original_samples=rec['record']['samples'],not_ready_original_samples=min(ready_sample,rec['record']['samples'])))
        with (OUT/(name+'.jsonl')).open('x') as stream:
            for threshold in p['grid']:
                records=[];positives=[];neg={'common':dict(clips=0,activated_clips=0,events=0),'cross_target':dict(clips=0,activated_clips=0,events=0)}
                bg=0;hours=0.;ready_hours=0.;pre=0;old_before_ready=0
                for rec,raw in traces:
                    row=rec['record'];original_steps=event_steps(raw,threshold)
                    steps=ready_event_steps(raw,threshold,p['context_frames'],p['stride_frames'])
                    assert all(s>=p['first_ready_call'] for s in steps)
                    times=[min((3*s+2)*160,row['samples']) for s in steps]
                    old_early=[s for s in original_steps if s<p['first_ready_call']];old_before_ready+=len(old_early)
                    item=dict(phase=rec['phase'],index=rec['index'],source_id=row['source_id'],original_pre_ready_events=old_early,
                              events=[dict(step=s,sample=t) for s,t in zip(steps,times,strict=True)])
                    if rec['phase']=='background':
                        bg+=len(times);hours+=row['samples']/57600000
                        ready_hours+=max(0,row['samples']-ready_sample)/57600000
                    else:
                        item.update({k:row[k] for k in ['target_id','variant','carrier','length_scale']})
                        for event in item['events']:
                            for boundary in ['generated_prefix_end_sample','full_rir_prefix_end_sample','generated_container_end_sample','full_rir_container_end_sample']:
                                event[boundary+'_signed_ms']=(event['sample']-row[boundary])/16
                        if row['target_id']==model['target']:
                            after=[t for t in times if t>=row['lead_samples']]
                            hit=dict(any_after_source=bool(after),complete=any(t>=row['generated_prefix_end_sample'] for t in after),
                                     missing=not after,pre_source_events=sum(t<row['lead_samples'] for t in times),
                                     carrier=row['carrier'],variant=row['variant'],ready_after_container_end=ready_sample>row['generated_container_end_sample'],
                                     signed_latency_ms=(after[0]-row['generated_prefix_end_sample'])/16 if after else None)
                            item['positive']=hit;positives.append(hit);pre+=hit['pre_source_events']
                        else:
                            kind='cross_target' if row['target_id']=='hey_willow' else 'common'
                            neg[kind]['clips']+=1;neg[kind]['activated_clips']+=bool(times);neg[kind]['events']+=len(times)
                    records.append(item)
                assert len(positives)==36 and len(records)==372
                groups={}
                for key,group in [('all',positives),('standalone',[x for x in positives if not x['carrier']]),('command',[x for x in positives if x['carrier']])]:
                    lat=[x['signed_latency_ms'] for x in group if x['signed_latency_ms'] is not None]
                    groups[key]=dict(clips=len(group),complete=sum(x['complete'] for x in group),any_after_source=sum(x['any_after_source'] for x in group),missing=sum(x['missing'] for x in group),ready_after_container_end=sum(x['ready_after_container_end'] for x in group),signed_latency_ms=dict(observed=len(lat),median=float(np.median(lat)) if lat else None,p95=float(np.percentile(lat,95)) if lat else None))
                baseline=next(x for x in old if x['model']==name and x['threshold']==threshold)
                summary=dict(model=name,threshold=threshold,positive=groups,negative=neg,pre_source_events=pre,
                             original_pre_ready_events=old_before_ready,background=dict(events=bg,original_hours=hours,ready_hours=ready_hours,excluded_hours=hours-ready_hours,far_upper95_ready=upper(bg,ready_hours),far_upper95_original=upper(bg,hours)),
                             original_no_wait_summary=baseline,threshold_selected=False)
                stream.write(json.dumps(dict(summary=summary,records=records))+'\n');points.append(summary)
                print(json.dumps(dict(model=name,threshold=threshold,complete=groups['all']['complete'],negative_events=sum(x['events'] for x in neg.values()),far_upper95_ready=upper(bg,ready_hours))),flush=True)
    (OUT/'trace-readiness.json').write_text(json.dumps(trace_receipts,indent=2)+'\n')
    result=dict(schema='multispeaker-readiness-diagnostic-original-v1',execution_sha256=sha(LOCK),points=points,
                trace_readiness_sha256=sha(OUT/'trace-readiness.json'),threshold_selected=False,acceptance_scored=False,candidate_promoted=False,
                limits='Consumed-only explicit new startup-ready system behavior, not retrospective model success. All original curves and pre-ready codes retained, earliest detection3.23s; all36positiveclips remain, missed/latefirstphrases count. No inference/training/newdata or thresholdselection.')
    with (OUT/'summary.json').open('x') as f:f.write(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('action',choices=['plan','run']);args=a.parse_args();plan() if args.action=='plan' else run()
