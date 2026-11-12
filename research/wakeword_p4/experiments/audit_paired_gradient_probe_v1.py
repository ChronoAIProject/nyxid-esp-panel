"""Independent saved tensors/gradient arithmetic; no TensorFlow/model inference."""
import ast
import hashlib
import json
import subprocess
from pathlib import Path
import numpy as np

root = Path('.local/wakeword/paired-consistency-init-gradient-probe-v1')
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
r = json.loads((root/'gradient-probe.json').read_text())
assert sha(root/'probe-gradients.npz') == r['gradients_sha256']
assert sha(root/'probe-batches.npz') == r['batches_sha256']
assert sha(root/'initialization.json') == r['initialization_sha256']
init = json.loads((root/'initialization.json').read_text())
assert init['classifier_initial_sha256'] == 'a41cc84a5e15a42a13ee9fe02e38242fca4dd9afe67a19ba736587970b46fc54'
assert init['initial_weights_sha256'] == '8e7de8447bb95660844d687c5e44cd7d5c93f3106d08681e2ceda7cd6a28643b'
g = np.load(root/'probe-gradients.npz', allow_pickle=False)
b = np.load(root/'probe-batches.npz', allow_pickle=False)
mask = np.concatenate([np.full(v['size'], v['encoder'], dtype=bool) for v in r['variable_layout']])
assert all(a.shape == (50, len(mask)) and np.isfinite(a).all() for a in g.values())
np.testing.assert_allclose(g['main']+g['pair'], g['supervised'], rtol=1e-5, atol=1e-5)
assert np.count_nonzero(g['consistency'][:, ~mask]) == 0
ns, nd, cosine = [], [], []
for i, row in enumerate(r['traces']):
    for prefix in ['main', 'pair']:
        data = b[prefix+'_x'][i].tobytes()+b[prefix+'_y'][i].tobytes()+b[prefix+'_weight'][i].tobytes()
        assert hashlib.sha256(data).hexdigest() == row[prefix+'_tensor_sha256']
    assert b['main_y'][i].sum() == 32 and b['pair_y'][i].sum() == 10
    for prefix in ['main', 'pair']:
        np.testing.assert_array_equal(b[prefix+'_weight'][i], np.where(b[prefix+'_y'][i, :, 0] > 0, 1., 20.))
    a, d = g['supervised'][i, mask].astype(np.float64), g['consistency'][i, mask].astype(np.float64)
    na, nb = np.linalg.norm(a), np.linalg.norm(d); c = a@d/(na*nb)
    np.testing.assert_allclose([na, nb, c], [row['encoder'][k] for k in ['supervised_norm', 'consistency_norm', 'cosine']], rtol=1e-12)
    ns.append(na); nd.append(nb); cosine.append(c)
    at = 0
    for layout, reported in zip(r['variable_layout'], row['per_variable']):
        end = at+layout['size']
        for key, source in [('supervised_norm', 'supervised'), ('consistency_norm', 'consistency')]:
            np.testing.assert_allclose(np.linalg.norm(g[source][i, at:end].astype(np.float64)), reported[key], rtol=1e-12)
        at = end
ratio = float(np.median(ns)/np.median(nd)); coefficient = min(1., 10.**np.floor(np.log10(.1*ratio)))
assert ratio == r['R'] and coefficient == r['proposed_fixed_coefficient']
np.testing.assert_array_equal(coefficient*np.array(nd)/np.array(ns), r['scaled_gradient_ratios'])
# Source-level identity only; old historical main tensor draws were not saved.
def batch_ast(source):
    tree = ast.parse(source)
    return ast.dump(next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == 'batch'), include_attributes=False)
prior = subprocess.check_output(['git', 'show', '37d940d:research/wakeword_p4/training/train_stream.py'], text=True)
current = Path('research/wakeword_p4/training/train_stream.py').read_text()
assert batch_ast(prior) == batch_ast(current)
result = dict(original_report_sha256=sha(root/'gradient-probe.json'), gradients_sha256=r['gradients_sha256'],
    batches_sha256=r['batches_sha256'], script_sha256=sha(Path(__file__)), verified_batches=50,
    original_head_and_encoder_initialization_match=True, consistency_head_gradient_exact_zero=True,
    main_batch_function_ast_identical_to_v47=True, R=ratio, coefficient=coefficient,
    scaled_gradient_ratio_range=[min(r['scaled_gradient_ratios']), max(r['scaled_gradient_ratios'])],
    gradient_cosine_range=[min(cosine), max(cosine)], negative_cosine_batches=int(np.sum(np.array(cosine) < 0)),
    limits='Saved arrays and source arithmetic only; not independent autodiff or historical original tensor reproduction, training efficacy or acceptance.')
Path('research/wakeword_p4/results/4090-paired-init-gradient-audit-v1.json').write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps(result))
