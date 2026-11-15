"""All candidate grid counts plus frozenv30 reference; no acceptance or promotion."""
import argparse
import gzip
import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
from collect_melo_development_v2 import LOCK,OUT,sha,PRIME
from melo_grid_event_counts_v2 import grid_event_counts
from uint8_event_policy_v1 import event_steps
from select_melo_joint_development_v2 import STRATA
from select_unconsumed_threshold_v1 import select
sys.path.insert(0,str(Path('research/wakeword_p4/training').resolve()))
from evaluate import smooth,event_indices

def main():
 assert subprocess.check_output(['git','show','origin/research-hye-nyx-voice-gpu:'+str(LOCK)])==LOCK.read_bytes();plan=json.loads(LOCK.read_text())
 for f,h in plan['files'].items():assert sha(f)==h,f
 for n,v in plan['grid_runtime_packages'].items():assert metadata.version(n)==v
 original=json.loads((OUT/'result.json').read_text());assert original['all_passed'] and original['execution_sha256']==sha(LOCK)
 dest=OUT/'metrics';dest.mkdir(exist_ok=False)
 hours=sum(r['samples'] for r in plan['records'] if r['phase']=='background')/16000/3600
 population=hashlib.sha256(json.dumps(plan['records'],sort_keys=True).encode()).hexdigest();bg=plan['background_manifest_sha256']
 try:
  for model in plan['models']:
   source=OUT/model['id']/'records.jsonl';assert sha(source)==original['record_files'][model['id']]
   records=[json.loads(x) for x in source.read_text().splitlines()];assert len(records)==474
   if model['role']=='selectable-candidate':
    points=[dict(threshold=k/1280,positive={s:dict(clips=0,hits=0) for s in STRATA},confusion_clips=0,confusion_activated_clips=0,background_events=0,cross_target_clips=0,cross_target_activated_clips=0,priming_events=0) for k in range(1,1281)]
   else:points=[dict(threshold=.91796875,positive={s:dict(clips=0,hits=0) for s in STRATA},confusion_clips=0,confusion_activated_clips=0,background_events=0,cross_target_clips=0,cross_target_activated_clips=0,priming_events=0)]
   with gzip.open(dest/(model['id']+'-per-record-grid.jsonl.gz'),'xt') as log:
    for rec in records:
     assert sha(rec['trace_path'])==rec['trace_sha256'];row=rec['record']
     with np.load(rec['trace_path'],allow_pickle=False) as a:codes=a['uint8'].copy()
     if model['role']=='selectable-candidate':counts=grid_event_counts(codes,int(row['samples']),int(row['lead_samples']) if row['phase']=='synthetic' else 0)
     else:
      steps=[int(i)+5 for i in event_indices(smooth(codes.astype(np.float32)/256),.91796875)]
      clocks=[min((((3*s+2)*160+479)//480)*480,row['samples']+PRIME)-PRIME for s in steps];real=[x for x in clocks if x>0];lead=row['lead_samples'] if row['phase']=='synthetic' else 0
      counts=np.array([[len(real),int(any(t>=lead for t in real)),len(clocks)-len(real)]],np.int64)
     log.write(json.dumps(dict(index=rec['index'],source_id=row.get('source_id',row['id']),phase=row['phase'],real_events=counts[:,0].tolist(),post_source_hit=counts[:,1].tolist(),prime_events=counts[:,2].tolist()))+'\n')
     for p,c in zip(points,counts,strict=True):
      events,hit,prime=map(int,c);p['priming_events']+=prime
      if row['phase']=='background':p['background_events']+=events
      elif row['phrase_id']=='nyx':
       stratum=row['voice_id'].split(':')[-1]+('/bare' if row['carrier']=='bare' else '/command');p['positive'][stratum]['clips']+=1;p['positive'][stratum]['hits']+=hit
      elif row['phrase_id']=='official':p['cross_target_clips']+=1;p['cross_target_activated_clips']+=int(events>0)
      else:p['confusion_clips']+=1;p['confusion_activated_clips']+=int(events>0)
     print(json.dumps(dict(model=model['id'],completed=rec['index']+1,total=474,phase=row['phase'])),flush=True)
   table=dict(role='unconsumed-threshold-development',candidate_id=model['id'],model_sha256=model['sha256'],population_sha256=population,background_manifest_sha256=bg,background_hours=hours,expected_positive_strata=list(STRATA),points=points,collection_execution_sha256=sha(LOCK),promotion_qualified=False)
   with (dest/(model['id']+'-table.json')).open('x') as f:f.write(json.dumps(table,indent=2)+'\n')
   if model['role']=='selectable-candidate':
    result=select(table)
    with (dest/(model['id']+'-point.json')).open('x') as f:f.write(json.dumps(result,indent=2)+'\n')
    threshold=result['selected_threshold']
   else:threshold=.91796875
   if threshold is not None:
    with gzip.open(dest/(model['id']+'-event-details.jsonl.gz'),'xt') as log:
     for rec in records:
      row=rec['record'];assert sha(rec['trace_path'])==rec['trace_sha256']
      with np.load(rec['trace_path'],allow_pickle=False) as a:codes=a['uint8'].copy()
      steps=event_steps(codes,threshold) if model['role']=='selectable-candidate' else [int(i)+5 for i in event_indices(smooth(codes.astype(np.float32)/256),threshold)]
      events=[]
      for step in steps:
       clock=min((((3*step+2)*160+479)//480)*480,row['samples']+PRIME)-PRIME
       event=dict(step=int(step),sample=int(clock),priming=bool(clock<=0))
       if row['phase']=='synthetic':
        for boundary in ['generated_prefix_end_sample','full_rir_prefix_end_sample','generated_container_end_sample','full_rir_container_end_sample']:event[boundary+'_signed_ms']=(clock-row[boundary])/16
       events.append(event)
      log.write(json.dumps(dict(index=rec['index'],record=row,threshold=threshold,events=events,synthetic_clocks_are_phonetic_truth=False))+'\n')
  (dest/'completed.json').write_text(json.dumps(dict(models=3,all_complete=True,threshold_joint_selected=False,official_reference_pending=True,promotion_qualified=False))+'\n')
 except BaseException as e:
  (dest/'failure.json').write_text(json.dumps(dict(type=type(e).__name__,error=str(e)))+'\n');raise
if __name__=='__main__':main()
