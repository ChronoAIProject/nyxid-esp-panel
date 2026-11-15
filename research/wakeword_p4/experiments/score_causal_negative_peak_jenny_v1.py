"""Existing priming fixed-grid app-event diagnosis; retain priming event state."""
import json,subprocess,sys
from pathlib import Path
import numpy as np
from collect_causal_negative_peak_jenny_v1 import R,LOCK,OUT
from collect_multispeaker_priming_v1 import sha,PRIME
from uint8_event_policy_v1 import event_steps,EVENT_ARITHMETIC
from hey_development_metrics_v1 import far_upper
sys.path.insert(0,str(Path('research/wakeword_p4/training').resolve()))
from evaluate import smooth as frozen_smooth,event_indices as frozen_event_indices
BOUNDARIES=('generated_prefix_end_sample','full_rir_prefix_end_sample','generated_container_end_sample','full_rir_container_end_sample')
def main():
 assert subprocess.check_output(['git','show','origin/research-hye-nyx-voice-gpu:'+str(Path(__file__).relative_to(Path.cwd()))])==Path(__file__).read_bytes()
 assert subprocess.check_output(['git','show','origin/research-hye-nyx-voice-gpu:'+str(LOCK)])==LOCK.read_bytes();p=json.loads(LOCK.read_text())
 for f,h in p['files'].items():assert sha(f)==h,f
 original=json.loads((OUT/'result.json').read_text());assert original['execution_sha256']==sha(LOCK) and original['all_passed']
 dest=OUT/'diagnostic';dest.mkdir(exist_ok=False);summaries=[]
 for model in p['models']:
  name=model['target']+'-'+model['condition'];base=OUT/name;report=json.loads((base/'report.json').read_text());assert len(report['records'])==408 and report['execution_sha256']==sha(LOCK)
  traces=[]
  for rec in report['records']:
   assert sha(rec['trace_path'])==rec['trace_sha256'] and rec['prime_samples']==PRIME
   with np.load(rec['trace_path'],allow_pickle=False) as a:
    raw=a['uint8'].copy();assert raw.dtype==np.uint8;assert rec['quantization']['output']==[1/256,0];assert len(raw)==(max(0,(rec['record']['samples']+PRIME+159)//160-2)+2)//3
    support=(3*np.arange(1,len(raw)+1,dtype=np.int64)+2)*160;assert np.array_equal(a['support_unclamped'],support) and np.array_equal(a['sample_end'],np.minimum(support,rec['record']['samples']+PRIME))
   traces.append((rec,raw))
  with (dest/(name+'.jsonl')).open('x') as log:
   for threshold,frozen_policy in [(.91796875,model['condition']=='v30')]:
    details=[];positive=[];neg={'common':dict(clips=0,activated_clips=0,events=0),'cross_target':dict(clips=0,activated_clips=0,events=0)};bg=0;hours=0.;pre_source=0
    for rec,raw in traces:
     row=rec['record'];all_steps=([int(i)+5 for i in frozen_event_indices(frozen_smooth(raw.astype(np.float32)/256),threshold)] if frozen_policy else event_steps(raw,threshold))
     total=row['samples']+PRIME
     all_times=[min(((3*s+2)*160+479)//480*480,total)-PRIME for s in all_steps]
     prime_events=[dict(step=s,relative_sample=t) for s,t in zip(all_steps,all_times,strict=True) if t<=0]
     steps=[s for s,t in zip(all_steps,all_times,strict=True) if t>0];times=[t for t in all_times if t>0]
     item=dict(phase=rec['phase'],index=rec['index'],source_id=row['source_id'],priming_events=prime_events,events=[dict(step=s,sample=t,support_relative_sample=(3*s+2)*160-PRIME,finish_call=((3*s+2)*160+479)//480*480>total) for s,t in zip(steps,times,strict=True)])
     if rec['phase']=='background':bg+=len(times);hours+=row['samples']/16000/3600
     else:
      item.update({k:row[k] for k in ['target_id','variant','carrier','length_scale']})
      for event in item['events']:event.update({b+'_signed_ms':(event['sample']-row[b])/16 for b in BOUNDARIES})
      if row['target_id']==model['target']:
       after=[t for t in times if t>=row['lead_samples']];complete=any(t>=row['generated_prefix_end_sample'] for t in after);pre_source+=sum(t<row['lead_samples'] for t in times)
       hit=dict(any_after_source=bool(after),complete=complete,before_only=bool(after) and not complete,missing=not after,latency_ms=(after[0]-row['generated_prefix_end_sample'])/16 if after else None,carrier=row['carrier'],variant=row['variant']);positive.append(hit);item['positive']=hit
      else:
       other='hey_nyx' if model['target']=='hey_willow' else 'hey_willow';kind='cross_target' if row['target_id']==other else 'common';neg[kind]['clips']+=1;neg[kind]['activated_clips']+=int(bool(times));neg[kind]['events']+=len(times)
     details.append(item)
    assert len(positive)==36 and neg['common']['clips']==324 and neg['cross_target']['clips']==36
    groups={}
    for key,group in [('all',positive),('standalone',[x for x in positive if x['carrier']=='bare']),('command',[x for x in positive if x['carrier']!='bare'])]+[(f'variant{v}',[x for x in positive if x['variant']==v]) for v in sorted({x['variant'] for x in positive})]:
     lat=[x['latency_ms'] for x in group if x['latency_ms']is not None]
     groups[key]=dict(clips=len(group),**{k:sum(x[k] for x in group) for k in ['any_after_source','complete','before_only','missing']},signed_latency_ms=dict(median=float(np.median(lat)) if lat else None,p95=float(np.percentile(lat,95)) if lat else None,observed=len(lat)))
    summary=dict(model=name,threshold=threshold,priming_events_ignored_by_app=sum(len(d['priming_events']) for d in details),event_arithmetic=('original-float64-convolution-five-v30' if frozen_policy else EVENT_ARITHMETIC),frozen_parameters_reference=frozen_policy,positive=groups,negative=neg,pre_source_events=pre_source,background=dict(events=bg,hours=hours,far_upper95=far_upper(bg,hours)),threshold_selected=False)
    log.write(json.dumps(dict(summary=summary,records=details))+'\n');log.flush();summaries.append(summary);print(json.dumps(summary),flush=True)
 (dest/'summary.json').write_text(json.dumps(dict(execution_sha256=sha(LOCK),points=summaries,threshold_selected=False,acceptance_scored=False,candidate_promoted=False,limits='Existing Jenny396 plus complete earnings9.614h development replay, fixed threshold and smoothing5. Not final acceptance, threshold search, or multi-voice proof. Generated boundaries are synthesis references, not human phonetic truth.'),indent=2)+'\n')
if __name__=='__main__':main()
