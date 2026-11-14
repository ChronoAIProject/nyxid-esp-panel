"""Saved-output/training-schedule alignment only; no model call or mining for training."""
import hashlib,json,math
from pathlib import Path
import numpy as np
R=Path('research/wakeword_p4/results');BASE=Path('.local/wakeword/common-carrier-command-training-streams-v1')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 schedules=R/'4090-hey-common-carrier-training-schedules-v1.json'
 formal=R/'4090-hey-common-carrier-raw-formal-result-v1.json'
 spec=next(s for s in json.loads(schedules.read_text())['conditions'] if s['target']=='hey_nyx' and s['condition']=='command_context')
 sp=Path(spec['path']);assert sha(sp)==spec['sha256']
 training=next(c for c in json.loads(formal.read_text())['cases'] if c['target']=='hey_nyx' and c['condition']=='command_context')['repeats'][0]
 actual=Path(training['batch_trace_path']);assert sha(actual)==training['batch_trace_sha256']
 observed=[json.loads(s) for s in actual.read_text().splitlines()];planned=[json.loads(s) for s in sp.read_text().splitlines()]
 assert len(observed)==len(planned)==10000
 ends={};selected=0
 for i,(a,b) in enumerate(zip(observed,planned),1):
  assert a['step']==b['step']==i and a['tensor_sha256']==b['tensor_sha256']
  for c in b['selected']:ends.setdefault(c['new_order'],[]).append(c['new_end']);selected+=1
 assert selected==37114
 records=BASE/'records.jsonl';output=[];groups={}
 for line in records.read_text().splitlines():
  rec=json.loads(line)
  if rec['condition']!='command_context':continue
  r=rec['row'];assert r['order']==rec['original_index'];path=Path(rec['trace']);assert sha(path)==rec['trace_sha256']
  with np.load(path,allow_pickle=False) as a:
   v=a['probability'];means=np.array([math.fsum(float(z) for z in v[j-4:j+1])/5 for j in range(4,len(v))]);peak=int(np.argmax(means));last_step=peak+5;endpoint=last_step*3
   # All five component outputs have end-exclusive feature coordinates separated by3.
   components=list(range(endpoint-12,endpoint+1,3));sampled=ends.get(r['order'],[])
   assert all(225<=e<=r['frames'] for e in sampled)
   row=dict(order=r['order'],anchor=r['anchor_text'],carrier=r['carrier'],conditioning_id=r['conditioning_id'],variant=r['variant'],trace_sha256=rec['trace_sha256'],maximum_five=float(means[peak]),first_maximum_five_last_step=last_step,component_feature_end_exclusive=components,actual_training_draws=len(sampled),unique_training_ends=len(set(sampled)),sampled_at_peak_components=sorted(set(sampled)&set(components)),nearest_sampled_end_distance=min(abs(e-c) for e in sampled for c in components) if sampled else None,peak_before_valid225=endpoint<225,peak_uses_padded_tail=endpoint>r['frames'])
  output.append(row);groups.setdefault((row['anchor'],row['carrier']),[]).append(row)
 assert len(output)==4860
 summary=[]
 for (anchor,carrier),items in sorted(groups.items()):
  assert len(items)==180
  summary.append(dict(anchor=anchor,carrier=carrier,clips=180,never_sampled=sum(not r['actual_training_draws'] for r in items),any_peak_component_sampled=sum(bool(r['sampled_at_peak_components']) for r in items),maxfive_ge_half=sum(r['maximum_five']>=.5 for r in items),maxfive_ge_half_and_peak_component_sampled=sum(r['maximum_five']>=.5 and bool(r['sampled_at_peak_components']) for r in items),maxfive_ge_half_without_any_training_draw=sum(r['maximum_five']>=.5 and not r['actual_training_draws'] for r in items)))
 result=dict(files={str(p):sha(p) for p in (schedules,sp,formal,actual,records,Path(__file__))},training_steps=10000,matched_actual_tensor_hashes=10000,selected_draws=selected,rows=output,groups=summary,model_invocations=0,threshold_selected=False,limits='First global max-five window only, ties resolved first. Feature end-exclusive coordinates, NOT PCM samples or phoneme boundaries. Schedule selected endpoints authenticated via actual tensor hashes; same endpoint does not mean same state/phase or current/final weights at training time. .5 is fixed descriptive score support, not a selected workpoint/event/FAR. Nearest endpoint distance is not causal; no new fitting inputs or freshdev/acceptance.')
 with (R/'4090-command-training-endpoint-alignment-v1.json').open('x') as f:json.dump(result,f,indent=2)
 for g in summary:
  if g['anchor'] in ('Hey Nick.','Hey Next.'):print(json.dumps(g))
if __name__=='__main__':main()
