"""All paired engine-adversarial points, deltas and historical references."""
import json
from pathlib import Path
from collect_engine_adversarial_completed_nyx_v1 import R,LOCK,OUT,sha
def span(xs):return [min(xs),max(xs)] if xs else [None,None]
def main():
 ap=R/'4090-engine-adversarial-completed-nyx-scoring-independent-audit-v1.json';sp=OUT/'diagnostic/summary.json';ref=R/'4090-hey-common-carrier-comparison-v1.json'
 for p in [ap,sp,ref]:
  if not p.exists():print(json.dumps({'status':'not_ready','missing':str(p)}));return 2
 audit=json.loads(ap.read_text());summary=json.loads(sp.read_text());old=json.loads(ref.read_text());lock=json.loads(LOCK.read_text())
 assert audit['all_passed'] and audit['points']==276 and audit['summary_sha256']==sha(sp) and summary['execution_sha256']==sha(LOCK)
 assert all(summary[k]is False for k in ['threshold_selected','acceptance_scored','candidate_promoted'])
 points=summary['points'];assert len(points)==276;rows=[];deltas=[];rejected=[]
 for target in ['hey_nyx']:
  arms={c:[p for p in points if p['model']==target+'-'+c] for c in ['control','adversarial']}
  for condition,ps in arms.items():
   assert len(ps)==138 and [p['threshold'] for p in ps]==lock['grid'] and all(p['background']['hours']==old['background_hours'] for p in ps)
   far=[p for p in ps if p['background']['far_upper95']<=.5]
   joint=[p for p in far if not p['pre_source_events'] and not p['negative']['common']['events'] and not p['negative']['cross_target']['events']]
   rows.append({'target':target,'condition':condition,'far_qualified_points':len(far),'negative_only_joint_points':len(joint),'complete_at_all_far_points':span([p['positive']['all']['complete'] for p in far]),'bare_at_all_far_points':span([p['positive']['standalone']['complete'] for p in far]),'command_at_all_far_points':span([p['positive']['command']['complete'] for p in far]),'complete_at_all_negative_joint_points':span([p['positive']['all']['complete'] for p in joint]),'minimum_far_upper95':min(p['background']['far_upper95'] for p in ps)})
   if not joint:rejected.append(target+'-'+condition)
  for full,frozen in zip(arms['control'],arms['adversarial'],strict=True):
   assert full['threshold']==frozen['threshold']
   a={'target':target,'threshold':full['threshold'],'adversarial_minus_control_complete':frozen['positive']['all']['complete']-full['positive']['all']['complete'],'adversarial_minus_control_background_events':frozen['background']['events']-full['background']['events'],'adversarial_minus_control_pre_source':frozen['pre_source_events']-full['pre_source_events']}
   for kind in ['common','cross_target']:a['adversarial_minus_control_'+kind+'_events']=frozen['negative'][kind]['events']-full['negative'][kind]['events']
   deltas.append(a)
 result={'files':{str(p):sha(p) for p in [LOCK,ap,sp,ref,Path(__file__)]},'summary':rows,'all_new_points':points,'all_matched_deltas':deltas,'all_original_reference_curves':old['all_curves'],'no_negative_only_joint_point':rejected,'selected_threshold':None,'candidate_promoted':False,'limits':'Single fit/arm, correlated consumed Semaine diagnostics only. All point ranges descriptive; silent zero-negative detectors are not success. Actual INT8 consumed diagnosis does not establish source quality or freshdev/independent acceptance. Negative-only joint points are not full qualification. Native512/raw480 event policies differ, generated boundaries not human phonetic truth. v30 unchanged.'}
 with (R/'4090-engine-adversarial-completed-nyx-comparison-v1.json').open('x') as f:json.dump(result,f,indent=2)
 print(json.dumps(rows));return 0
if __name__=='__main__':raise SystemExit(main())
