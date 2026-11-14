"""Compare complete saved bare/command training score supports without point selection."""
import hashlib,json
from pathlib import Path
R=Path('research/wakeword_p4/results')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 paths=[R/'4090-common-carrier-training-streams-result-v1.json',R/'4090-common-carrier-command-training-streams-result-v1.json',R/'4090-hey-common-carrier-failure-localization-v1.json']
 bare,command,diagnostic=[json.loads(p.read_text()) for p in paths]
 assert bare['audited_traces']==6120 and command['audited_traces']==9720
 assert bare['all_repeat_bytes_exact'] and command['all_repeat_bytes_exact']
 rows=[]
 for g in command['groups']:
  original=next(b for b in bare['groups'] if b['condition']==g['condition'] and b['text']==g['text'])
  assert len(g['all_grid_score_support'])==len(original['all_grid_score_support'])==138
  assert [p['threshold'] for p in g['all_grid_score_support']]==[p['threshold'] for p in original['all_grid_score_support']]
  vals=[p['clips_with_any_five_mean_at_or_above'] for p in g['all_grid_score_support']]
  old=[p['clips_with_any_five_mean_at_or_above'] for p in original['all_grid_score_support']]
  rows.append(dict(condition=g['condition'],anchor=g['text'],carrier=g['carrier'],clips=g['clips'],bare_clips=original['clips'],bare_support_range=[min(old),max(old)],command_support_range=[min(vals),max(vals)],bare_median_maxfive=original['maximum_five_quantiles']['0.5'],command_median_maxfive=g['maximum_five_quantiles']['0.5'],command_max_maxfive=g['maximum_five_quantiles']['1'],points_with_more_command_support=sum(n>o for n,o in zip(vals,old))))
 out=dict(files={str(p):sha(p) for p in paths},rows=rows,diagnostic=diagnostic['models'],threshold_selected=False,model_invocations=0,limits='Training-origin cohorts, correlated rendering variants. Control never trained on replacement command rows; context used prescribed crops only. Score-support crossings are not complete-phrase/latched-event counts or FAR qualification. Same-anchor bare and command PCM change generated realization/duration/endpoint support jointly; not a pure causal carrier or domain experiment. All138 points retained, no freshdev or acceptance.')
 with (R/'4090-common-carrier-training-support-comparison-v1.json').open('x') as f:json.dump(out,f,indent=2)
 for r in rows:
  if r['anchor'] in ('Hey Next.','Hey Nick.'):print(json.dumps(r))
if __name__=='__main__':main()
