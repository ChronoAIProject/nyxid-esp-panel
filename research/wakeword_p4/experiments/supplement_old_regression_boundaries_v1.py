"""Additive generated-boundary offsets from audited old saved ledgers only."""
import json,hashlib
from pathlib import Path
import numpy as np
from run_hey_broad_trained_quantization_v1 import published
R=Path('research/wakeword_p4/results')
BOUNDARIES=('generated_prefix_end_sample','full_rir_prefix_end_sample','generated_container_end_sample','full_rir_container_end_sample')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def stats(values):
 return dict(count=len(values),minimum=min(values),median=float(np.median(values)),p95=float(np.quantile(values,.95)),maximum=max(values)) if values else dict(count=0,minimum=None,median=None,p95=None,maximum=None)
def main():
 source=Path(__file__).resolve().relative_to(Path.cwd());plan=R/'4090-old-fixed-grid-regression-plan-v2.json';metrics=R/'4090-old-fixed-grid-regression-metrics-v2.json'
 # Independent audit must be completed and published before this additive reduction.
 audit=R/'4090-old-fixed-grid-regression-independent-audit-v2.json'
 for p in (source,plan,metrics,audit):published(p)
 lock=json.loads(plan.read_text());report=json.loads(metrics.read_text());assert report['plan_sha256']==sha(plan)
 checked=json.loads(audit.read_text());assert checked['metrics_sha256']==sha(metrics) and len(checked['summary'])==9
 assert all(s['points']==138 and s['complete_event_ledgers_proved'] and s['all_metric_reductions_exact'] for s in checked['summary'])
 manifest=Path(lock['synthetic_manifest']);assert sha(manifest)==lock['files'][str(manifest)]
 records=json.loads(manifest.read_text())['records'];assert len(records)==360 and len(report['points'])==1242
 out=[];seen=set()
 for point in report['points']:
  key=(point['model'],point['kind'],point['threshold']);assert key not in seen;seen.add(key)
  path=Path(point['events_path']);assert sha(path)==point['events_sha256']
  with np.load(path,allow_pickle=False) as arrays:
   offsets=arrays['synthetic_offsets'];steps=arrays['synthetic_steps'];assert offsets.shape==(361,) and offsets[0]==0 and offsets[-1]==len(steps) and np.all(np.diff(offsets)>=0)
   rows=[]
   for index,record in enumerate(records):
    if record['target_id']!=point['target']:continue
    events=steps[offsets[index]:offsets[index+1]];assert np.all(np.diff(events)>0)
    times=[min((3*int(s)+2)*160,record['samples']) for s in events]
    boundaries={}
    for boundary in BOUNDARIES:
     end=record[boundary];assert 0<=end<=record['samples'];after=[t for t in times if t>=end]
     boundaries[boundary]=dict(first_event_signed_ms=(times[0]-end)/16 if times else None,first_at_or_after_ms=(after[0]-end)/16 if after else None)
    rows.append(dict(index=index,source_id=record['source_id'],event_count=len(times),boundaries=boundaries))
   assert len(rows)==36
   summary={b:{field:stats([r['boundaries'][b][field] for r in rows if r['boundaries'][b][field] is not None]) for field in ('first_event_signed_ms','first_at_or_after_ms')} for b in BOUNDARIES}
   out.append(dict(model=point['model'],kind=point['kind'],threshold=point['threshold'],events_sha256=point['events_sha256'],positive_rows=rows,summary=summary))
 result=dict(files={str(p):sha(p) for p in (source,plan,metrics,audit,manifest)},points=out,selection_authorized=False,model_invocations=0,limits='Additive old-consumed descriptive report. All fixed1242 points, all36 target positives per point, four generated boundaries. First event may precede boundary and is signed; first-at-or-after excludes earlier events explicitly. Missing events remain null and counts are reported. Generated allocation/convolution support is not human phonetic truth. Does not alter original reports/gates or authorize fresh data/v30 replacement.')
 with (R/'4090-old-fixed-grid-regression-boundary-supplement-v1.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
if __name__=='__main__':main()
