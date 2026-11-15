"""Fixed official operating point, same complete source population, raw receipts."""
import gzip
import json
from pathlib import Path
from collect_melo_development_v2 import sha
from select_melo_joint_development_v2 import STRATA
R=Path('research/wakeword_p4/results');PLAN=R/'4090-melo-official-execution-v2.json'
def main():
 p=json.loads(PLAN.read_text());out=Path(p['output']);complete=json.loads((out/'complete.json').read_text());assert complete['streams']==474 and complete['plan_sha256']==sha(PLAN)
 result=dict(threshold=.6299999952316284,positive={s:dict(clips=0,hits=0) for s in STRATA},confusion_clips=0,confusion_activated_clips=0,cross_target_clips=0,cross_target_activated_clips=0,background_events=0,priming_events=0);details=[];hours=0.
 for i,row in enumerate(p['records']):
  path=out/f'{i:04}-report.json';r=json.loads(path.read_text());assert r['record']==row and r['execution_sha256']==sha(PLAN) and r['source_input_samples']==row['samples'] and r['source_pcm_sha256']==row['pcm_sha256'] and r['input_samples']==row['samples']+49344
  assert r['actual_threshold']==.6299999952316284 and r['total_prime_samples']==81600 and r['continuous_model'] and not r['transition_reset']
  assert sha(out/f'{i:04}-raw-returns.bin')==r['raw_returns_sha256'] and sha(out/f'{i:04}-serial.log')==r['serial_sha256']
  times=[e['sample_end']-49344 for e in r['events'] if e['sample_end']>49344];prime=len(r['additional_priming_events'])+sum(v==1 for _,v in r['prime_returns']);result['priming_events']+=prime
  d=dict(index=i,phase=row['phase'],source_id=row.get('source_id',row['id']),report_sha256=sha(path),events=[dict(sample=t) for t in times],priming_events=prime)
  if row['phase']=='background':result['background_events']+=len(times);hours+=row['samples']/16000/3600
  else:
   d.update(voice_id=row['voice_id'],phrase_id=row['phrase_id'],carrier=row['carrier'],variant=row['variant'])
   for event in d['events']:
    for boundary in ['generated_prefix_end_sample','full_rir_prefix_end_sample','generated_container_end_sample','full_rir_container_end_sample']:event[boundary+'_signed_ms']=(event['sample']-row[boundary])/16
   if row['phrase_id']=='official':
    s=row['voice_id'].split(':')[-1]+('/bare' if row['carrier']=='bare' else '/command');result['positive'][s]['clips']+=1;result['positive'][s]['hits']+=int(any(t>=row['lead_samples'] for t in times))
   elif row['phrase_id']=='nyx':result['cross_target_clips']+=1;result['cross_target_activated_clips']+=int(bool(times))
   else:result['confusion_clips']+=1;result['confusion_activated_clips']+=int(bool(times))
  details.append(d)
 assert {s:v['clips'] for s,v in result['positive'].items()}==STRATA and result['confusion_clips']==288 and result['cross_target_clips']==72
 report=dict(schema='melo-official-point-v2',point=result,background_hours=hours,execution_sha256=sha(PLAN),acceptance_accessed=False,promotion_qualified=False,limits=p['policy'])
 with (out/'point.json').open('x') as f:f.write(json.dumps(report,indent=2)+'\n')
 with gzip.open(out/'event-details.jsonl.gz','xt') as f:
  for d in details:f.write(json.dumps(d)+'\n')
 print(json.dumps(report),flush=True)
if __name__=='__main__':main()
