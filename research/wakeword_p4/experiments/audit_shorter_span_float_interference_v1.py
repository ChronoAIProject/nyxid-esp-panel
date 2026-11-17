"""Complete daily/background precision census, saved results only."""
import hashlib,json
from pathlib import Path
R=Path('research/wakeword_p4/results')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
out={'schema':'shorter-span-float-interference-census-v1','arms':{},'limits':'Consumed96daily clips and full reused backgrounds; no pooled/additional independent exposure, human or P4 claim. Equal event counts do not imply equal event identity.'}
for arm in ['control','shortspan']:
 p=Path(f'.local/wakeword/shorter-span-float-v1/{arm}/result.json');rows=json.loads(p.read_text())['rows'];groups={}
 for c in ['daily','primed','jenny','melo']:
  rr=[r for r in rows if r['cohort']==c and (c=='daily' or r['phase']=='background')];g={'streams':len(rr),'events':{},'event_sources':[]}
  for n in ['float','int8']:g['events'][n]=sum(sum(e['relative_sample']>0 for e in r['events'][n]) for r in rr)
  for r in rr:
   ev={n:[e for e in r['events'][n] if e['relative_sample']>0] for n in ['float','int8']}
   if any(ev.values()):g['event_sources'].append({'index':r['index'],'source':r['source'],'events':ev,'trace':r['trace']})
  groups[c]=g
 out['arms'][arm]={'source_report_sha256':sha(p),'groups':groups}
with (R/'4090-shorter-span-float-interference-census-v1.json').open('x') as f:json.dump(out,f,indent=2);f.write('\n')
