"""Verify all 120k ordered input draws and actual initialization across matched arms."""
import hashlib
import json
from pathlib import Path
base=Path('.local/wakeword')
roots=[base/f'model-temporal-tail-risk-{arm}-formal-v1' for arm in ('control','treatment')]
sha=lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
for name in ['actual-initialization-inventory.json','paired-sampling-trace.jsonl']:
    assert sha(roots[0]/name)==sha(roots[1]/name), name
for coefficient,root in enumerate(roots):
    report=json.loads((root/'training.json').read_text())
    assert report['steps']==120000
    paired=report['paired_rendering']
    assert paired['objective']=='temporal-tail-risk' and paired['phrase_groups']==10
    assert paired['ranking_coefficient']==1 and paired['temporal_windows']==5 and paired['seed']==237726
    assert paired['temporal_coefficient']==coefficient
    assert paired['coefficient']==coefficient and paired['optimizer_updates']==120000
    assert sum(1 for _ in (root/'paired-sampling-trace.jsonl').open())==120000
    assert paired['sampling_trace_sha256']==sha(root/'paired-sampling-trace.jsonl')
    assert paired['loss_trace_sha256']==sha(root/'paired-losses.jsonl')
    assert sum(1 for _ in (root/'paired-losses.jsonl').open())==120000
    assert paired['initialization_snapshot_sha256']==sha(root/'actual-initialization.weights.h5')
    assert paired['initialization_inventory_sha256']==sha(root/'actual-initialization-inventory.json')
first=[json.loads((root/'paired-losses.jsonl').open().readline()) for root in roots]
assert {k:v for k,v in first[0].items() if k!='total'}=={k:v for k,v in first[1].items() if k!='total'}
out=base/'temporal-tail-risk-formal-input-audit-v1.json'
assert not out.exists()
out.write_text(json.dumps(dict(steps_per_arm=120000,initialization_inventory_exact=True,
    all_ordered_input_draws_exact=True,trace_sha256=sha(roots[0]/'paired-sampling-trace.jsonl'),
    first_step_observations_exact=True,acceptance_claim=False),indent=2)+'\n')
