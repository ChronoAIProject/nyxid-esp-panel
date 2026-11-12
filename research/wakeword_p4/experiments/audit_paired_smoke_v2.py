"""Audit fresh matched smoke execution against the explicit full-FP32 reference."""
import hashlib
import json
from pathlib import Path
import numpy as np

base = Path('.local/wakeword')
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
roots = [base/f'model-paired-{arm}-smoke-v2' for arm in ('control', 'treatment')]
reports = [json.loads((r/'training.json').read_text()) for r in roots]
losses = [[json.loads(line) for line in (r/'paired-losses.jsonl').read_text().splitlines()] for r in roots]
for coefficient, r, t, rows in zip((0, 1), roots, reports, losses):
    assert t['steps'] == 2 and len(rows) == 2
    paired = t['paired_rendering']
    assert paired['coefficient'] == coefficient and paired['optimizer_updates'] == 2
    assert paired['numerical_policy']['op_determinism'] and not paired['numerical_policy']['tf32']
    assert t['batchnorm_initial'] == t['batchnorm_selected']
    assert paired['initialization_snapshot_sha256'] == sha(r/'actual-initialization.weights.h5')
    assert paired['initialization_inventory_sha256'] == sha(r/'actual-initialization-inventory.json')
    assert t['model_sha256'] == sha(r/'hey_nyx.tflite')
for name in ('actual-initialization-inventory.json', 'paired-sampling-trace.jsonl'):
    assert (roots[0]/name).read_bytes() == (roots[1]/name).read_bytes(), name
# Loss-log field spelling is taken from the actual trainer, not inferred.
keys = list(k for k in losses[0][0] if k not in ('step', 'total'))
np.testing.assert_array_equal([losses[0][0][k] for k in keys], [losses[1][0][k] for k in keys])
reference = np.load(base/'paired-numerical-replay-cuda-v1/arrays.npz')['metrics']
for rows in losses:
    np.testing.assert_allclose([rows[0][k] for k in ('main_bce', 'pair_bce', 'consistency')], reference, rtol=1e-5, atol=1e-5)
for arm in ('control', 'treatment'):
    audit = json.loads((base/f'paired-{arm}-smoke-v2-export-audit.json').read_text())
    assert audit['steps'] == 2
result = dict(smoke_checks_passed=True, formal_training_started=False,
    initialization_inventory_exact=True, ordered_sampling_exact=True,
    first_step_observations_exact_between_arms=True, reference_rtol=1e-5, reference_atol=1e-5,
    first_steps=[r[0] for r in losses], original_v1_failure_preserved=True,
    limits='Fresh deterministic full-FP32 smoke only; no retroactive original probe passage, coefficient change, acceptance or new-data access.')
out = Path('research/wakeword_p4/results/4090-paired-smoke-comparison-v2.json')
assert not out.exists()
out.write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps(result))
