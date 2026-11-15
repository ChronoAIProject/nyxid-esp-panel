"""All 432 synthetic reference receipts, explicitly not background completion."""
import json
from pathlib import Path
import numpy as np
from collect_melo_development_v2 import sha
from select_melo_joint_development_v2 import STRATA
ROOT=Path('research/wakeword_p4');PLAN=ROOT/'results/4090-melo-official-execution-v2.json';BASE=Path('.local/wakeword/melo-official-v2')
def main():
 plan=json.loads(PLAN.read_text());positive={s:dict(clips=0,hits=0) for s in STRATA};groups={};receipts=[];latencies=[];prime_events=0
 for index,row in enumerate(plan['records']):
  if row['phase']!='synthetic':continue
  path=BASE/f'{index:04}-report.json';r=json.loads(path.read_text());assert r['record']==row and r['record_index']==index and r['execution_sha256']==sha(PLAN)
  assert r['actual_threshold']==.6299999952316284 and r['total_prime_samples']==81600 and r['source_pcm_sha256']==row['pcm_sha256'] and r['source_input_samples']==row['samples']
  assert sha(BASE/f'{index:04}-serial.log')==r['serial_sha256'] and sha(BASE/f'{index:04}-raw-returns.bin')==r['raw_returns_sha256']
  times=[e['sample_end']-49344 for e in r['events'] if e['sample_end']>49344];prime_events+=len(r['additional_priming_events'])+sum(v==1 for _,v in r['prime_returns'])
  group=groups.setdefault(row['phrase_id'],dict(clips=0,activated=0));group['clips']+=1;group['activated']+=int(bool(times))
  if row['phrase_id']=='official':
   s=row['voice_id'].split(':')[-1]+('/bare' if row['carrier']=='bare' else '/command');positive[s]['clips']+=1;positive[s]['hits']+=int(any(t>=row['lead_samples'] for t in times))
   if times:latencies.append((times[0]-row['generated_prefix_end_sample'])/16)
  receipts.append(dict(index=index,report_sha256=sha(path)))
 assert len(receipts)==432 and {s:v['clips'] for s,v in positive.items()}==STRATA and all(g['clips']==72 for g in groups.values())
 summary=dict(schema='melo-official-synthetic-interim-v2',official_threshold=.6299999952316284,synthetic_complete=True,synthetic_records=432,positive=positive,phrase_groups=groups,priming_events=prime_events,first_detected_target_generated_prefix_signed_ms=dict(zip(['min','p25','median','p75','max'],map(float,np.percentile(latencies,[0,25,50,75,100])))) if latencies else None,plan_sha256=sha(PLAN),receipts=receipts,background_complete=False,far_qualified=False,acceptance_accessed=False,promotion_qualified=False,limits='All synthetic development records only; the original full background run continues. No final FAR/matched-level claim. Own-target phrases differ between detectors. Internal synthesis boundary latency is descriptive, not acoustic phonetic truth. Development was consumed; acceptance remains closed after candidate confusion failure.')
 out=ROOT/'results/4090-melo-official-synthetic-interim-v2.json'
 with out.open('x') as f:f.write(json.dumps(summary,indent=2)+'\n')
 print(json.dumps({k:v for k,v in summary.items() if k!='receipts'}),flush=True)
if __name__=='__main__':main()
