"""Signed first-detection latency from original native events; misses retained."""
import hashlib,json
from pathlib import Path
import numpy as np
R=Path('research/wakeword_p4/results')
def main():
 source=R/'4090-official-boundary-event-descriptive-v1.json';d=json.loads(source.read_text());points=[]
 for p in d['points']:
  groups={'all':p['cases'],'standalone':[c for c in p['cases'] if not c['carrier']],'command':[c for c in p['cases'] if c['carrier']]}
  groups.update({'variant'+str(v):[c for c in p['cases'] if c['variant']==v] for v in range(3)});summary={}
  for name,cases in groups.items():
   values=[]
   for c in cases:
    after=[e['sample_end'] for e in c['events'] if e['sample_end']>=c['lead_samples']]
    if after:values.append((min(after)-c['generated_prefix_end_sample'])/16)
   x=np.asarray(values,float);stats=None if not len(x) else dict(minimum=float(x.min()),median=float(np.median(x)),p95=float(np.percentile(x,95)),maximum=float(x.max()),before_generated_boundary=int(np.sum(x<0)))
   summary[name]=dict(total=len(cases),detected=len(values),missing=len(cases)-len(values),first_after_source_signed_ms=stats)
  points.append(dict(model=p['model'],request=p['request'],actual_float32_bits=p['actual_float32_bits'],groups=summary))
 out=dict(schema='official-first-detection-signed-generated-latency-v1',files={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [source,Path(__file__)]},points=points,selection_authorized=False,limits='First native event at/after source onset relative to unchanged generated-prefix reference; negative signed latency means before generated boundary, not human phonetic anticipation. Missing clips retained in denominators; statistics conditional on detections, not overall latency guarantees. All26points, no workpoint selection, acoustic inference or scoring changes.')
 (R/'4090-official-signed-generated-latency-v1.json').write_text(json.dumps(out,indent=2)+'\n')
 for p in points:
  if p['request']=='DEFAULT':print(p['model'],json.dumps(p['groups']['all']))
if __name__=='__main__':main()
