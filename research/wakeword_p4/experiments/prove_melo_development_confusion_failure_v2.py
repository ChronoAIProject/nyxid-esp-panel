"""Conservative impossibility bound, retaining the full ongoing official run."""
from fractions import Fraction
import hashlib
import json
from pathlib import Path
BASE=Path('.local/wakeword/melo-official-v2');R=Path('research/wakeword_p4/results')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 plan=R/'4090-melo-official-execution-v2.json';p=json.loads(plan.read_text());joint=Path('.local/wakeword/melo-development-int8-v2/metrics/joint-selection.json');selected=json.loads(joint.read_text());point=selected['selected_point'];assert point['confusion_clips']==288
 receipts=[];activated=0;identities=set()
 for path in sorted(BASE.glob('*-report.json')):
  report=json.loads(path.read_text());row=report['record']
  if row['phase']!='synthetic' or row['phrase_id'] not in ['nick','next','mix','muse']:continue
  index=report['record_index'];assert row==p['records'][index] and index not in identities;identities.add(index)
  assert report['execution_sha256']==sha(plan) and report['actual_threshold']==.6299999952316284 and report['total_prime_samples']==81600
  assert sha(BASE/f'{index:04}-serial.log')==report['serial_sha256'] and sha(BASE/f'{index:04}-raw-returns.bin')==report['raw_returns_sha256']
  hit=any(e['sample_end']>49344 for e in report['events']);activated+=int(hit)
  receipts.append(dict(index=index,path=str(path),sha256=sha(path),activated=hit))
 processed=len(receipts);maximum=activated+288-processed
 limit=Fraction(maximum,288)+Fraction(1,40);actual=Fraction(point['confusion_activated_clips'],288)
 assert actual>limit,'Failure not yet proven by conservative bound'
 out=R/'4090-melo-development-confusion-failure-bound-v2.json'
 report=dict(schema='melo-development-confusion-failure-bound-v2',selected_candidate=selected['selected_candidate'],selected_threshold=selected['selected_threshold'],selection_sha256=sha(joint),official_plan_sha256=sha(plan),candidate_common_activated=point['confusion_activated_clips'],common_population=288,official_completed_common=processed,official_activated_so_far=activated,official_remaining_common=288-processed,official_maximum_final_activated=maximum,even_all_remaining_official_activate_candidate_exceeds_allowance=True,engineering_allowance_fraction='1/40',candidate_fraction=str(actual),maximum_allowed_fraction=str(limit),development_confusion_gate_proven_failed=True,full_official_run_still_required=True,acceptance_accessed=False,promotion_qualified=False,v30_retained=True,receipts=receipts,limits='Conservative arithmetic bound, not a final official score: all remaining common clips are assumed to activate official HiESP. Continue original full run and publish complete metrics/latencies/background. Do not open acceptance, replace threshold or fall back to runner-up.')
 with out.open('x') as f:f.write(json.dumps(report,indent=2)+'\n')
 print(json.dumps({k:v for k,v in report.items() if k!='receipts'}),flush=True)
if __name__=='__main__':main()
