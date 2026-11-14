"""Describe every original official positive event relative to fixed generated boundaries."""
import hashlib,json
from pathlib import Path
R=Path('research/wakeword_p4/results')
def main():
 cp=R/'4090-original-official-development-calibration-v1.json';ep=R/'4090-hey-official-development-execution-v1.json';c=json.loads(cp.read_text());e=json.loads(ep.read_text());lookup={(r['source_id'],r['variant']):r for r in e['records'] if r['phase']=='synthetic'};points=[]
 for model,curve in c['curves'].items():
  target='hey_willow' if model=='willow' else 'hi_esp'
  for p in curve:
   cases=[];counts=dict(no_event=0,pre_source_only=0,after_source_before_prefix_only=0,complete_phrase=0,after_source_any=0,pre_prefix_and_complete=0)
   for record in p['synthetic']['records']:
    if record['target_id']!=target:continue
    row=lookup[record['source_id'],record['variant']];events=record['events'];after=[v for v in events if v['sample_end']>=row['lead_samples']];complete=[v for v in after if v['sample_end']>=row['generated_prefix_end_sample']];before=[v for v in after if v['sample_end']<row['generated_prefix_end_sample']]
    category='complete_phrase' if complete else 'after_source_before_prefix_only' if after else 'pre_source_only' if events else 'no_event';counts[category]+=1;counts['after_source_any']+=bool(after);counts['pre_prefix_and_complete']+=bool(before and complete)
    cases.append(dict(source_id=record['source_id'],variant=record['variant'],carrier=record['carrier'],lead_samples=row['lead_samples'],generated_prefix_end_sample=row['generated_prefix_end_sample'],full_rir_prefix_end_sample=row['full_rir_prefix_end_sample'],category=category,events=events))
   assert len(cases)==36 and counts['complete_phrase']==p['synthetic']['complete_phrase_activated'];assert abs(counts['after_source_any']/36-p['synthetic']['any_after_source_recall'])<1e-12
   points.append(dict(model=model,request=p['request'],actual_float32_bits=p['actual_float32_bits'],counts=counts,cases=cases))
 out=dict(schema='official-boundary-event-descriptive-v1',files={str(x):hashlib.sha256(x.read_bytes()).hexdigest() for x in [cp,ep,Path(__file__)]},points=points,selection_authorized=False,limits='All original 26 points/936 positive clip evaluations. Categories use fixed original lead/generated-prefix boundaries. Before-prefix-only is a detected event, but fails original complete-phrase criterion. Boundary is synthesis allocation, not measured last phoneme. Native events cannot prove counterfactual detection absent refractory; no refractory interval inferred or event rescoring. Full RIR reference retained separately. No inference, new data or threshold selection.')
 (R/'4090-official-boundary-event-descriptive-v1.json').write_text(json.dumps(out,indent=2)+'\n')
 for p in points:print(p['model'],p['request'],p['counts'])
if __name__=='__main__':main()
