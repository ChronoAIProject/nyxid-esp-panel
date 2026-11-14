"""All-point descriptive native/custom detection versus fixed generated boundaries."""
from collections import defaultdict
import hashlib,json
from pathlib import Path
R=Path('research/wakeword_p4/results')
def main():
 paths=[R/'4090-old-fixed-grid-regression-boundary-supplement-v1.json',R/'4090-old-fixed-grid-regression-metrics-v2.json',R/'4090-official-boundary-event-descriptive-v1.json'];b,m,o=[json.loads(p.read_text()) for p in paths];assert len(b['points'])==len(m['points'])==1242;rows=[]
 for boundary,metric in zip(b['points'],m['points'],strict=True):
  assert all(boundary[k]==metric[k] for k in ['model','kind','threshold']) and boundary['events_sha256']==metric['events_sha256'];ps=boundary['positive_rows'];assert len(ps)==36 and metric['synthetic']['pre_source_positive_events']==0
  anyhit=sum(x['event_count']>0 for x in ps);complete=sum(x['boundaries']['generated_prefix_end_sample']['first_at_or_after_ms'] is not None for x in ps);assert complete==metric['synthetic']['complete_phrase_activated']
  rows.append(dict(model=metric['model'],path='bounded-float' if metric['kind']=='float' else metric['kind'],request=metric['threshold'],after_source_any=anyhit,complete_phrase=complete,before_only=anyhit-complete,no_event=36-anyhit,far_upper95=metric['background']['upper95_per_hour']))
 for point in o['points']:
  c=point['counts'];rows.append(dict(model='official-'+point['model'],path='native',request=point['request'],after_source_any=c['after_source_any'],complete_phrase=c['complete_phrase'],before_only=c['after_source_before_prefix_only'],no_event=c['no_event']))
 groups=defaultdict(list)
 for r in rows:groups[r['model']+'/'+r['path']].append(r)
 summaries=[]
 for name,rs in sorted(groups.items()):
  summaries.append(dict(model_path=name,points=len(rs),any_hit_range=[min(x['after_source_any'] for x in rs),max(x['after_source_any'] for x in rs)],complete_hit_range=[min(x['complete_phrase'] for x in rs),max(x['complete_phrase'] for x in rs)],before_only_range=[min(x['before_only'] for x in rs),max(x['before_only'] for x in rs)]))
 out=dict(schema='all-point-detection-boundary-comparison-v1',files={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},points=rows,summaries=summaries,selection_authorized=False,limits='All1268 existing consumed points retained; ranges are descriptive, not workpoint selection or matched FAR result. Complete versus any difference is boundary-associated, not proven refractory causality. Custom float is bounded-float; no raw FP32 recognition. Native and custom event policies differ; synthetic boundaries are not acoustic phoneme truth. No model/media reads.')
 (R/'4090-all-point-detection-boundary-comparison-v1.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(summaries,indent=2))
if __name__=='__main__':main()
