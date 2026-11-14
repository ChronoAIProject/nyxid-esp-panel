"""Keep all phase experiment curves, matched deltas and immutable references."""
import hashlib,json
from pathlib import Path
from collect_phase_aligned_development_v1 import OUT,LOCK
R=Path('research/wakeword_p4/results')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 ap=R/'4090-phase-aligned-scoring-independent-audit-v1.json';sp=OUT/'diagnostic/summary.json';ref=R/'4090-hey-common-carrier-comparison-v1.json'
 for p in (ap,sp,ref):
  if not p.exists():print(json.dumps(dict(status='not_ready',missing=str(p))));return 2
 audit=json.loads(ap.read_text());summary=json.loads(sp.read_text());old=json.loads(ref.read_text());lock=json.loads(LOCK.read_text())
 assert audit['all_passed'] and audit['points']==276 and audit['summary_sha256']==sha(sp)
 assert summary['execution_sha256']==sha(LOCK) and all(summary[k]is False for k in ('threshold_selected','acceptance_scored','candidate_promoted'))
 points=summary['points'];assert len(points)==276
 arms={c:[p for p in points if p['model']=='hey_nyx-'+c] for c in ('nominal','phase')}
 for ps in arms.values():
  assert len(ps)==138 and [p['threshold'] for p in ps]==lock['grid']
  assert all(p['background']['hours']==old['background_hours'] for p in ps)
 rows=[]
 def span(xs):return [min(xs),max(xs)] if xs else [None,None]
 for c,ps in arms.items():
  qualified=[p for p in ps if p['background']['far_upper95']<=.5]
  joint=[p for p in qualified if not p['pre_source_events'] and not p['negative']['common']['events'] and not p['negative']['cross_target']['events']]
  rows.append(dict(condition=c,total_points=138,far_qualified_points=len(qualified),far_and_zero_negative_points=len(joint),complete_at_all_far_points=span([p['positive']['all']['complete'] for p in qualified]),bare_at_all_far_points=span([p['positive']['standalone']['complete'] for p in qualified]),command_at_all_far_points=span([p['positive']['command']['complete'] for p in qualified]),complete_at_joint_points=span([p['positive']['all']['complete'] for p in joint]),minimum_background_far_upper95=min(p['background']['far_upper95'] for p in ps)))
 deltas=[]
 for n,p in zip(arms['nominal'],arms['phase'],strict=True):
  assert n['threshold']==p['threshold']
  deltas.append(dict(threshold=n['threshold'],phase_minus_nominal_complete=p['positive']['all']['complete']-n['positive']['all']['complete'],phase_minus_nominal_common_events=p['negative']['common']['events']-n['negative']['common']['events'],phase_minus_nominal_cross_events=p['negative']['cross_target']['events']-n['negative']['cross_target']['events'],phase_minus_nominal_background_events=p['background']['events']-n['background']['events'],phase_minus_nominal_pre_source=p['pre_source_events']-n['pre_source_events']))
 result=dict(files={str(p):sha(p) for p in (LOCK,ap,sp,ref,Path(__file__))},background_hours=old['background_hours'],summary=rows,all_new_points=points,all_matched_deltas=deltas,all_original_reference_curves=old['all_curves'],threshold_selected=False,candidate_promoted=False,limits='Every original138point/arm retained, ranges are descriptive rather than chosen workpoints. Matched nominal/phase shares all source/end/label/weight schedules and fixed budgets; single correlated consumed Semaine source and one fit/arm do not establish independent generalization. Generated timing is not phonetic truth. Official native512/raw480 policies differ. FP32 recognition is not realwaveINT8 qualification or prospective freshdev/acceptance. Frozenv30 unchanged.')
 with (R/'4090-phase-aligned-comparison-v1.json').open('x') as f:json.dump(result,f,indent=2)
 print(json.dumps(rows,indent=2));return 0
if __name__=='__main__':raise SystemExit(main())
