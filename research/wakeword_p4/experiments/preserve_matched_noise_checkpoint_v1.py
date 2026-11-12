"""Preserve a completed original float checkpoint record without selecting a model."""
import argparse,hashlib,json
from pathlib import Path
from scipy.stats import chi2
p=argparse.ArgumentParser();p.add_argument('--condition',choices=['independent','common'],required=True);p.add_argument('--step',type=int,required=True);a=p.parse_args()
root=Path(f'.local/wakeword/model-matched-noise-{a.condition}-formal-v1')
with (root/'progress.jsonl').open('rb') as f:records=[line for line in f if line.endswith(b'\n')]
raw=next(line for line in records if json.loads(line)['step']==a.step)
r=json.loads(raw);s=r['selected'];eligible=[x for x in r['grid'] if x['ambient_faph']<=.5 and x['max_source_false_rate']<=.01]
tag=f'4090-matched-noise-{a.condition}-formal-checkpoint{a.step}'
out=Path('research/wakeword_p4/results')
assert not (out/f'{tag}-v1.json').exists() and not (out/f'{tag}-audit-v1.json').exists()
(out/f'{tag}-v1.json').write_bytes(raw)
report={'condition':a.condition,'step':a.step,'original_record_sha256':hashlib.sha256(raw).hexdigest(),'original_record_bytes':len(raw),'float_selection_eligible_points':len(eligible),'selected_is_fallback':not bool(eligible),'selected':s,'one_sided_95_background_far_upper':float(chi2.ppf(.95,2*(s['ambient_events']+1))/(2*s['ambient_hours'])),'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'limits':'Original reused float training-validation record, provisional selection only; not complete-run, INT8 or independent acceptance. No re-selection/configuration change/newdev/final access.'}
(out/f'{tag}-audit-v1.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
