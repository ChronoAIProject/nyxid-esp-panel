"""Descriptive all-point reduction of existing consumed development reports."""
import hashlib
import json
from collections import defaultdict
from pathlib import Path
R=Path('research/wakeword_p4/results')
def main():
    paths=[R/'4090-original-official-development-calibration-v1.json',R/'4090-old-fixed-grid-regression-metrics-v2.json',R/'4090-official-custom-input-identity-v1.json',R/'4090-old-float-head-semantics-correction-v1.json']
    official,old,identity,correction=[json.loads(p.read_text()) for p in paths]
    assert identity['synthetic_ordered_rows_exact']==360 and identity['background_ordered_pcm_identities_exact']==12
    assert correction['original_plan_locks_exact']
    groups=defaultdict(list)
    for model,points in official['curves'].items():
        for p in points:groups['official/'+model].append((p['synthetic'],p['background_upper95'],False))
    for p in old['points']:
        name=p['model']+'/'+('bounded-float' if p['kind']=='float' else 'uint8')
        groups[name].append((p['synthetic'],p['background']['upper95_per_hour'],p['frozen_v30_reference_point']))
    summaries=[]
    for name,rows in sorted(groups.items()):
        far=[r for r in rows if r[1]<=.5]
        def full(r):
            s=r[0]
            return r[1]<=.5 and s['macro']>=.8 and s['standalone']>=.75 and s['command']>=.75 and s['pre_source_positive_events']==0 and all(n['events']==0 for n in s['negatives'].values())
        summaries.append(dict(path=name,registered_points=len(rows),far_upper95_le_half_points=len(far),full_diagnostic_points=sum(full(r) for r in rows),far_only_complete_hit_range=[min(r[0]['complete_phrase_activated'] for r in far),max(r[0]['complete_phrase_activated'] for r in far)] if far else None,frozen_v30=[dict(complete_hits=r[0]['complete_phrase_activated'],far_upper95=r[1],full_diagnostic=full(r)) for r in rows if r[2]]))
    out=dict(schema='shared-far-descriptive-diagnostics-v1',files={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},summaries=summaries,threshold_selected=False,acceptance_authorized=False,limits='Consumed diagnostic data only. Counts/ranges describe ALL registered points, not independent points, chosen endpoints or candidates. Full diagnostic applies the official descriptive criteria to saved custom metrics; not numerical qualification/admission. Different target positives/native versus five-step event policies remain. Float means bounded-float here; no raw sigmoid FP32 result. No inference or fresh data.')
    (R/'4090-shared-far-descriptive-diagnostics-v1.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(summaries,indent=2))
if __name__=='__main__':main()
