"""Exhaustive consumed-grid bounds, never a replacement operating point."""
import hashlib
import json
from pathlib import Path
from scipy.stats import chi2
BASE=Path('.local/wakeword/melo-development-int8-v2/metrics')
OUT=Path('research/wakeword_p4/results/4090-melo-grid-tradeoff-diagnostic-v2.json')
def main():
 report=dict(schema='melo-grid-tradeoff-diagnostic-v2',caps_explanation={'0':'No common confusion clips','7':'Floor(288*0.025), illustrative allowance only if full official common confusion is zero','55':'Consumed c161 selected-point confusion count; descriptive comparison only'},models={},selection_changed=False,acceptance_accessed=False,promotion_qualified=False,limits='Post-selection exhaustive bounds on already consumed development grid. No new candidate, threshold replacement, qualification or final official result. Maxima may occur at different points and do not establish any combined achievable operating point.')
 for name in ['c161','speecht5-final20000']:
  path=BASE/(name+'-table.json');table=json.loads(path.read_text());hours=table['background_hours'];points=table['points'];assert len(points)==1280
  feasible=[p for p in points if float(chi2.ppf(.95,2*(p['background_events']+1))/(2*hours))<=.5]
  bounds={}
  for cap in [0,7,55]:
   rows=[p for p in feasible if p['confusion_activated_clips']<=cap]
   bounds[str(cap)]=dict(points=len(rows),maximum_total_hits=max((sum(v['hits'] for v in p['positive'].values()) for p in rows),default=None),maximum_worst_stratum_recall=max((min(v['hits']/v['clips'] for v in p['positive'].values()) for p in rows),default=None),any_point_overall80_worst75=any(sum(v['hits'] for v in p['positive'].values())>=58 and all(v['hits']/v['clips']>=.75 for v in p['positive'].values()) for p in rows))
  report['models'][name]=dict(table_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),background_feasible_points=len(feasible),bounds=bounds)
 with OUT.open('x') as f:f.write(json.dumps(report,indent=2)+'\n')
 print(json.dumps(report),flush=True)
if __name__=='__main__':main()
