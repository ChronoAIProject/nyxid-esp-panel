"""Independent existing event oracle on saved readiness behavior; no model calls."""
import json
import math
from pathlib import Path
import numpy as np
from scipy.stats import chi2
from diagnose_multispeaker_readiness_v1 import R,SOURCE,OUT,LOCK,sha
from audit_multispeaker_formal_scoring_v1 import means,events

def main():
    plan=json.loads(LOCK.read_text());summary=json.loads((OUT/'summary.json').read_text())
    assert summary['execution_sha256']==sha(LOCK) and len(summary['points'])==276
    assert plan['discard_score_count']==102 and plan['first_ready_call']==107 and plan['first_ready_support_sample']==51680
    prefix=json.loads((OUT/'trace-readiness.json').read_text());assert sha(OUT/'trace-readiness.json')==summary['trace_readiness_sha256']
    summaries=[];groups=[]
    original=json.loads((SOURCE/'diagnostic/summary.json').read_text())['points']
    for model in plan['models']:
        name=model['target']+'-'+model['condition'];records=json.loads((SOURCE/name/'report.json').read_text())['records'];averages=[]
        assert len(records)==372
        for rec in records:
            assert sha(rec['trace_path'])==rec['trace_sha256']
            with np.load(rec['trace_path'],allow_pickle=False) as trace:raw=trace['uint8'].copy()
            pre=next(x for x in prefix if x['model']==name and x['phase']==rec['phase'] and x['index']==rec['index'])
            assert pre['pre_ready_uint8']==raw[:106].tolist() and pre['valid_buffer_uint8']==raw[102:106].tolist()
            assert pre['not_ready_original_samples']==min(51680,rec['record']['samples'])
            averages.append((means(raw),means(raw[102:])))
        points=[json.loads(x) for x in (OUT/(name+'.jsonl')).read_text().splitlines()]
        assert len(points)==138 and [x['summary']['threshold'] for x in points]==plan['grid']
        for point in points:
            s=point['summary'];assert s['model']==name
            assert s['original_no_wait_summary']==next(x for x in original if x['model']==name and x['threshold']==s['threshold'])
            assert len(point['records'])==372
            counts={'common':[0,0,0],'cross_target':[0,0,0]};pos=[];bg=before=pre_source=0;total_samples=active_samples=0
            for rec,avg,seen in zip(records,averages,point['records'],strict=True):
                row=rec['record'];old,new=avg
                assert (seen['phase'],seen['index'],seen['source_id'])==(rec['phase'],rec['index'],row['source_id'])
                steps=[v+102 for v in events(new,s['threshold'])];times=[min((3*v+2)*160,row['samples']) for v in steps]
                old_early=[v for v in events(old,s['threshold']) if v<107]
                assert seen['original_pre_ready_events']==old_early;before+=len(old_early)
                assert [x['step'] for x in seen['events']]==steps and [x['sample'] for x in seen['events']]==times
                if rec['phase']=='background':
                    bg+=len(times);total_samples+=row['samples'];active_samples+=max(0,row['samples']-51680)
                else:
                    for event in seen['events']:
                        for boundary in ['generated_prefix_end_sample','full_rir_prefix_end_sample','generated_container_end_sample','full_rir_container_end_sample']:
                            assert event[boundary+'_signed_ms']==(event['sample']-row[boundary])/16
                    if row['target_id']==model['target']:
                        after=[t for t in times if t>=row['lead_samples']]
                        expected=dict(any_after_source=bool(after),complete=any(t>=row['generated_prefix_end_sample'] for t in after),missing=not after,pre_source_events=sum(t<row['lead_samples'] for t in times),carrier=row['carrier'],variant=row['variant'],ready_after_container_end=51680>row['generated_container_end_sample'],signed_latency_ms=(after[0]-row['generated_prefix_end_sample'])/16 if after else None)
                        assert seen['positive']==expected;pos.append(expected);pre_source+=expected['pre_source_events']
                    else:
                        key='cross_target' if row['target_id']=='hey_willow' else 'common';counts[key][0]+=1;counts[key][1]+=bool(times);counts[key][2]+=len(times)
            assert len(pos)==36 and bg==s['background']['events'] and before==s['original_pre_ready_events'] and pre_source==s['pre_source_events']
            assert s['negative']=={k:dict(clips=v[0],activated_clips=v[1],events=v[2]) for k,v in counts.items()}
            hours=total_samples/57600000;active=active_samples/57600000
            assert math.isclose(hours,s['background']['original_hours'],rel_tol=1e-14) and math.isclose(active,s['background']['ready_hours'],rel_tol=1e-14)
            for suffix,exposure in [('original',s['background']['original_hours']),('ready',s['background']['ready_hours'])]:
                assert s['background']['far_upper95_'+suffix]==float(.5*chi2.ppf(.95,2*(bg+1))/exposure)
            for key,got in s['positive'].items():
                group=pos if key=='all' else [x for x in pos if (not x['carrier'] if key=='standalone' else bool(x['carrier']))]
                assert got['clips']==len(group)
                for k in ['complete','any_after_source','missing','ready_after_container_end']:assert got[k]==sum(x[k] for x in group)
                lat=[x['signed_latency_ms'] for x in group if x['signed_latency_ms'] is not None]
                assert got['signed_latency_ms']==dict(observed=len(lat),median=float(np.median(lat)) if lat else None,p95=float(np.percentile(lat,95)) if lat else None)
            summaries.append(s)
        ps=[x['summary'] for x in points];far=[x for x in ps if x['background']['far_upper95_ready']<=.5]
        joint=[x for x in far if not x['pre_source_events'] and not any(v['events'] for v in x['negative'].values())]
        def span(xs):return [min(xs),max(xs)] if xs else [None,None]
        groups.append(dict(model=name,far_points=len(far),negative_only_joint_points=len(joint),complete_at_far=span([x['positive']['all']['complete'] for x in far]),standalone_at_far=span([x['positive']['standalone']['complete'] for x in far]),complete_at_negative_joint=span([x['positive']['all']['complete'] for x in joint]),minimum_far_upper95_ready=min(x['background']['far_upper95_ready'] for x in ps),background_original_hours=ps[0]['background']['original_hours'],background_ready_hours=ps[0]['background']['ready_hours'],background_excluded_hours=ps[0]['background']['excluded_hours']))
    assert summaries==summary['points'] and len(prefix)==744
    result=dict(schema='multispeaker-readiness-saved-behavior-review-v1',reviewer_sha256=sha(__file__),original_summary_sha256=sha(OUT/'summary.json'),points=276,summary=groups,all_events_metrics_and_exposure_checks_passed=True,model_calls=0,threshold_selected=False,limits='Independent existing vector event oracle and metric reduction on saved outputs, shared chi2 library. Descriptive consumed diagnosis only; no selected threshold or freshdata/system qualification.')
    with (R/'4090-multispeaker-readiness-saved-behavior-review-v1.json').open('x') as f:f.write(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))
if __name__=='__main__':main()
