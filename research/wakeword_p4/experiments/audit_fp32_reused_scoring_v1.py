"""Independent event-index oracle and reduction of all original diagnostic points."""
import hashlib,json,math
from pathlib import Path
import numpy as np
from scipy.stats import chi2
from collect_fp32_reused_development_v1 import R,LOCK,OUT,sha
def means(raw):
 values=list(map(float,raw));return np.asarray([math.fsum(values[i:i+5])/5 for i in range(len(values)-4)],np.float64)
def events(avg,threshold):
 high=np.flatnonzero(avg>=threshold);low=np.flatnonzero(avg<threshold*.5);result=[];start=0
 while True:
  h=int(np.searchsorted(high,start))
  if h==len(high):break
  index=int(high[h]);result.append(index+5)
  l=int(np.searchsorted(low,index+67))
  if l==len(low):break
  start=int(low[l])
 return result
def main():
 p=json.loads(LOCK.read_text());dest=OUT/'diagnostic';summary=json.loads((dest/'summary.json').read_text());assert summary['execution_sha256']==sha(LOCK) and len(summary['points'])==552
 from fp32_event_policy_v1 import event_steps
 rng=np.random.default_rng(20261114)
 for _ in range(10):
  raw=rng.integers(0,256,512).astype(np.float32)/256
  for threshold in [.5,.7,.9,.99]:assert events(means(raw),threshold)==event_steps(raw,threshold)
 checked=[];all_summaries=[]
 for model in p['models']:
  name=model['target']+'-'+model['condition'];report=json.loads((OUT/name/'report.json').read_text());records=report['records'];assert len(records)==372
  averages=[]
  for rec in records:
   assert sha(rec['trace_path'])==rec['trace_sha256']
   with np.load(rec['trace_path'],allow_pickle=False) as a:averages.append(means(a['probability']))
  points=[json.loads(x) for x in (dest/(name+'.jsonl')).read_text().splitlines()];assert len(points)==138 and [x['summary']['threshold'] for x in points]==p['grid']
  for point in points:
   s=point['summary'];assert s['model']==name and s['threshold_selected']is False and len(point['records'])==372
   positives=[];neg={'common':dict(clips=0,activated_clips=0,events=0),'cross_target':dict(clips=0,activated_clips=0,events=0)};bg=0;hours=0.;pre=0
   for rec,avg,observed in zip(records,averages,point['records'],strict=True):
    row=rec['record'];assert (observed['phase'],observed['index'],observed['source_id'])==(rec['phase'],rec['index'],row['source_id'])
    steps=events(avg,s['threshold']);times=[min((3*t+2)*160,row['samples']) for t in steps]
    assert [e['step'] for e in observed['events']]==steps and [e['sample'] for e in observed['events']]==times
    if rec['phase']=='background':bg+=len(times);hours+=row['samples']/57600000
    else:
     for event in observed['events']:
      for b in ['generated_prefix_end_sample','full_rir_prefix_end_sample','generated_container_end_sample','full_rir_container_end_sample']:assert event[b+'_signed_ms']==(event['sample']-row[b])/16
     if row['target_id']==model['target']:
      after=[t for t in times if t>=row['lead_samples']];complete=any(t>=row['generated_prefix_end_sample'] for t in after);pre+=sum(t<row['lead_samples'] for t in times)
      expected=dict(any_after_source=bool(after),complete=complete,before_only=bool(after) and not complete,missing=not after,latency_ms=(after[0]-row['generated_prefix_end_sample'])/16 if after else None,carrier=row['carrier'],variant=row['variant'])
      assert observed['positive']==expected;positives.append(expected)
     else:
      other='hey_nyx' if model['target']=='hey_willow' else 'hey_willow';kind='cross_target' if row['target_id']==other else 'common';neg[kind]['clips']+=1;neg[kind]['activated_clips']+=int(bool(times));neg[kind]['events']+=len(times)
   assert neg==s['negative'] and pre==s['pre_source_events'] and bg==s['background']['events']
   assert math.isclose(hours,s['background']['hours'],rel_tol=1e-14)
   far=float(.5*chi2.ppf(.95,2*(bg+1))/s['background']['hours']);assert far==s['background']['far_upper95']
   for key,got in s['positive'].items():
    group=positives if key=='all' else [x for x in positives if (x['carrier']=='' if key=='standalone' else x['carrier']!='' if key=='command' else x['variant']==int(key.removeprefix('variant')))]
    assert got['clips']==len(group)
    for k in ['any_after_source','complete','before_only','missing']:assert got[k]==sum(x[k] for x in group)
    lat=[x['latency_ms'] for x in group if x['latency_ms']is not None];assert got['signed_latency_ms']==dict(median=float(np.median(lat)) if lat else None,p95=float(np.percentile(lat,95)) if lat else None,observed=len(lat))
   all_summaries.append(s)
  checked.append(dict(model=name,points=138,complete_inputs_per_point=372,records_sha256=sha(dest/(name+'.jsonl'))))
 assert all_summaries==summary['points'] and all(summary[k]is False for k in ['threshold_selected','acceptance_scored','candidate_promoted'])
 with (R/'4090-fp32-reused-scoring-independent-audit-v1.json').open('x') as f:f.write(json.dumps(dict(summary_sha256=sha(dest/'summary.json'),models=checked,points=552,all_passed=True,limits='Independent vector event-index oracle uses binary64 fsum means, authenticated original saved outputs and all emitted events/metrics. No model invoke, workpoint selection or freshdata. FAR chi2 library shared, count/exposure reduction independent.'),indent=2)+'\n')
 print('All552 diagnostic points/events independently verified')
if __name__=='__main__':main()
