"""Rebuild historical float selection ranks; no checkpoint rescoring or reselection."""
import hashlib
import json
from pathlib import Path

base=Path('.local/wakeword');reports={}
for tag in ['phrase-ranking-control','phrase-ranking-treatment','temporal-tail-risk-control','temporal-tail-risk-treatment']:
 p=base/f'model-{tag}-formal-v1/training.json';d=json.loads(p.read_text());trajectory=[];best=None;chosen=None
 for entry in d['history']:
  grid=entry['grid'];eligible=[r for r in grid if r['ambient_faph']<=.5 and r['max_source_false_rate']<=.01]
  if eligible:
   selected=max(eligible,key=lambda r:(r['macro_recall'],-r['ambient_faph'],-r['max_source_false_rate']))
   rank=(1,selected['macro_recall'],-selected['ambient_faph'],-selected['max_source_false_rate'])
  else:
   selected=min(grid,key=lambda r:(r['ambient_faph'],r['max_source_false_rate'],-r['macro_recall']))
   rank=(0,-selected['ambient_faph'],-selected['max_source_false_rate'],selected['macro_recall'])
  assert selected==entry['selected']
  if best is None or rank>best:best=rank;chosen=entry['step']
  trajectory.append({'step':entry['step'],'eligible_points':len(eligible),'selected':selected,'rank':rank,
                     'zero_confusion_and_zero_background_points':sum(r['max_source_false_rate']==0 and r['ambient_events']==0 for r in grid)})
 assert len(trajectory)==24 and chosen==d['selected_step']
 reports[tag]={'training_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'selected_step':chosen,'reconstructed_selection_exact':True,
 'eligible_checkpoints':sum(t['eligible_points']>0 for t in trajectory),'trajectory':trajectory,
 'saved_weight_files':[x.name for x in sorted(p.parent.glob('*.h5'))]}
out={'scope':'Original24 checkpoint float grids per four completed arms, reused historical validation only.','arms':reports,
 'findings':['Trainer already uses five-step smoothing, latched positive events and phase-aligned float windows; missing smoothing is not the issue.',
 'Selection admits up to1percent per-source confusion and uses background point rate, while final joint gate requires zero confusion and Poisson upper bound; protocols/datasets differ.',
 'Selected step exactly reconstructs original rank in every arm. This is not evidence another checkpoint would pass INT8 joint gate.',
 'Only saved weight artifacts are inventoried; unsaved checkpoint metrics cannot reconstruct weights or justify post-hoc replacement.'],
 'limits':'No new inference, threshold tuning, checkpoint replacement, optimizer, development/final waveform access or promotion. Historical trajectory does not establish a unique training failure cause.'}
Path('research/wakeword_p4/results/4090-checkpoint-selection-trajectory-v1.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({k:{j:v for j,v in r.items() if j!='trajectory'} for k,r in reports.items()},indent=2))
