"""Compare all original paired-arm grid points without rescoring or selecting thresholds."""
import hashlib
import json
from pathlib import Path
base=Path('research/wakeword_p4/results')
inputs={a:base/f'4090-paired-{a}-formal-v1-reused-development-exploratory-calibration-v1.json' for a in ['control','treatment']}
data={a:json.loads(p.read_text()) for a,p in inputs.items()}
rows=[]
for c,t in zip(data['control']['grid'],data['treatment']['grid'],strict=True):
    assert c['threshold']==t['threshold']
    row={'threshold':c['threshold']}
    for arm,r in [('control',c),('treatment',t)]:
        row[arm]={'macro':r['synthetic_macro_clip_trigger'],'negative_false_clips':sum(g['false_activation_clips'] for g in r['groups'].values()),'background_events':r['background_events'],'eligible':r['eligible']}
    rows.append(row)
assert len(rows)==138
out={'input_sha256':{a:hashlib.sha256(p.read_bytes()).hexdigest() for a,p in inputs.items()},'rows':rows,'treatment_lower_false_clips_points':sum(r['treatment']['negative_false_clips']<r['control']['negative_false_clips'] for r in rows),'treatment_lower_macro_points':sum(r['treatment']['macro']<r['control']['macro'] for r in rows),'limits':'All original grid points; no selected threshold, new inference/data, tuning or acceptance. Single seed; score suppression is not proven as the unique mechanism.'}
(base/'4090-paired-same-threshold-audit-v1.json').write_text(json.dumps(out,indent=2)+'\n')
print({k:v for k,v in out.items() if k!='rows'})
