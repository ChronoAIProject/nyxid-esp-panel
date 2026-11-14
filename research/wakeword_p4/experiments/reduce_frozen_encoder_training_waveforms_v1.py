"""Post-result descriptive all-grid reduction of original135 saved training waveforms."""
import hashlib,json,math
from collections import defaultdict
from pathlib import Path
import numpy as np
from fp32_event_policy_v1 import event_steps
R=Path('research/wakeword_p4/results')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def oracle(raw,t):
 values=list(map(float,raw));avg=np.array([math.fsum(values[i:i+5])/5 for i in range(len(values)-4)],np.float64)
 high=np.flatnonzero(avg>=t);low=np.flatnonzero(avg<t*.5);out=[];start=0
 while True:
  pos=int(np.searchsorted(high,start))
  if pos==len(high):return out
  k=int(high[pos]);out.append(k+5);j=int(np.searchsorted(low,k+67))
  if j==len(low):return out
  start=int(low[j])
def main():
 plan=R/'4090-hey-broad-trained-quantization-plan-v1.json';numeric=R/'4090-frozen-encoder-fp32-waveform-result-v1.json';audit=R/'4090-frozen-encoder-fp32-waveform-independent-audit-v1.json';dev=R/'4090-frozen-encoder-diagnostic-summary-v1.json';devlock=R/'4090-frozen-encoder-development-execution-v1.json'
 files={str(p):sha(p) for p in [plan,numeric,audit,dev,devlock,Path(__file__),Path(__file__).with_name('fp32_event_policy_v1.py')]}
 requests=json.loads(plan.read_text())['requests'];assert len(requests)==135;lookup={r['order']:r for r in requests};num=json.loads(numeric.read_text());assert num['all_passed'];assert json.loads(audit.read_text())['all_passed'];points=json.loads(dev.read_text())['points'];grid=json.loads(devlock.read_text())['grid'];assert len(grid)==138
 models=[]
 for target in ['hey_nyx','hey_willow']:
  for condition in ['full','frozen']:
   name=target+'-'+condition;cases=[c for c in num['cases'] if c['target']==target and c['condition']==condition];assert len(cases)==135;traces=[]
   for c in cases:
    assert sha(c['arrays_path'])==c['arrays_sha256'];files[c['arrays_path']]=c['arrays_sha256'];r=lookup[c['request_order']]
    with np.load(c['arrays_path'],allow_pickle=False) as a:
     raw=a['fp32'].copy();assert raw.dtype==np.float32 and np.array_equal(raw,a['fp32_repeat']) and np.isfinite(raw).all() and np.all((raw>=0)&(raw<=1))
     clocks=(3*np.arange(1,len(raw)+1)+2)*160;assert np.array_equal(a['support_unclamped'],clocks) and np.array_equal(a['sample_end'],np.minimum(clocks,r['samples']))
    traces.append((r,raw))
   curves=[]
   for threshold in grid:
    groups=defaultdict(list);details=[]
    for r,raw in traces:
     steps=event_steps(raw,threshold);assert steps==oracle(raw,threshold);times=[min((3*s+2)*160,r['samples']) for s in steps];src=r['source'];role='own' if src['target']==target else 'common' if src['target']=='common_negative' else 'reciprocal';after=[t for t in times if t>=r['lead_samples']];complete=bool(role=='own' and any(t>=r['generated_prefix_end_sample'] for t in after));hit=dict(order=r['order'],role=role,any_event=bool(times),pre_source_events=sum(t<r['lead_samples'] for t in times),complete=complete,events=len(times));details.append(dict(**hit,event_steps=steps))
     groups[(role,src['engine'],src['pair_key']['carrier'])].append(hit)
    aggregated=[dict(role=k[0],engine=k[1],carrier=k[2],clips=len(v),complete=sum(x['complete'] for x in v),activated_clips=sum(x['any_event'] for x in v),events=sum(x['events'] for x in v),pre_source_events=sum(x['pre_source_events'] for x in v)) for k,v in sorted(groups.items())]
    ref=next(p for p in points if p['model']==name and p['threshold']==threshold)
    curves.append(dict(threshold=threshold,training_groups=aggregated,training_records=details,training_complete=sum(x['complete'] for x in details),training_positive_clips=27,training_negative_activated=sum(x['any_event'] for x in details if x['role']!='own'),training_negative_clips=108,consumed_cross_source_complete=ref['positive']['all']['complete'],consumed_cross_source_positive_clips=36,consumed_cross_source_common_events=ref['negative']['common']['events'],consumed_cross_source_far_upper95=ref['background']['far_upper95']))
   models.append(dict(model=name,points=curves));print(name,'threshold0.5', {k:v for k,v in curves[0].items() if k not in ['training_groups','training_records']},flush=True)
   print('positive groups .5',[x for x in curves[0]['training_groups'] if x['role']=='own'],flush=True)
 out=dict(files=files,models=models,all_74520_event_lists_independently_matched=True,model_invocations=0,selected_threshold=None,candidate_promoted=False,limits='Post-result descriptive reduction of original135 numerics witnesses/model, chosen before this experiment, not all training sources or independent heldout data. Same saved ownFP32 waveform event function/original138grid and generated-prefix completeness policy; no fresh inference or threshold selection. All rows/startup/tails retained. Training27 positives vs consumed36 differ in voice/source/engine/carrier/lead/duration; observed gaps do not isolate a cause. No INT8/FAR/acceptance qualification or modified v30.')
 with (R/'4090-frozen-encoder-training-waveform-comparison-v1.json').open('x') as f:json.dump(out,f,indent=2)
if __name__=='__main__':main()
