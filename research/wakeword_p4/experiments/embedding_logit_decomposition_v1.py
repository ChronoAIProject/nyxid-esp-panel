"""Exact symmetric norm/direction logit decomposition; saved training arrays only.

For e=r*u and head w, a=dot(u,w), z=r*a+b. Relative to quiet,
delta_z = .5*(r-rq)*(a+aq) + .5*(r+rq)*(a-aq).
The symmetric identity avoids choosing norm-first vs direction-first order.
This is an algebraic decomposition, not causal mediation or loss selection.
"""
import hashlib
import json
from pathlib import Path
import numpy as np

root = Path('.local/wakeword/factorized-embedding-diagnostic-v1')
out = Path('.local/wakeword/embedding-logit-decomposition-v1')
assert not out.exists()
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(root/'arrays.npz') == 'e82b16f41d191a0b53555cba59bae7cddf33232906416c22ed8a8139e45cb821'
report = json.loads((root/'report.json').read_text())
z = np.load(root/'arrays.npz', allow_pickle=False)
arrays, summaries, checks = {}, {}, {}
def stats(a):
    return dict(count=int(a.size), mean=float(a.mean()),
                quantiles=np.quantile(a, [0, .05, .5, .95, 1]).tolist(),
                negative_count=int(np.sum(a < 0)), positive_count=int(np.sum(a > 0)))

for version in ['v46', 'v47']:
    e = z[version+'_embedding'].astype(np.float64)
    w = np.asarray(report['models'][version]['head_kernel'], dtype=np.float64)[:, 0]
    bias = report['models'][version]['head_bias'][0]
    norm = np.sqrt(np.sum(e*e, axis=-1)); assert np.all(norm > 0)
    projection = np.einsum('sead,d->sea', e, w)
    angle = projection/norm
    norm_term = .5*(norm-norm[:, :, :1])*(angle+angle[:, :, :1])
    direction_term = .5*(norm+norm[:, :, :1])*(angle-angle[:, :, :1])
    delta = projection-projection[:, :, :1]
    np.testing.assert_allclose(norm_term+direction_term, delta, rtol=1e-11, atol=1e-11)
    np.testing.assert_allclose(projection+bias, z[version+'_logit'], rtol=1e-5, atol=1e-5)
    checks[version] = dict(max_identity_error=float(np.abs(norm_term+direction_term-delta).max()),
        max_float32_head_difference=float(np.abs(projection+bias-z[version+'_logit']).max()))
    for name, value in [('delta', delta), ('norm_term', norm_term), ('direction_term', direction_term)]:
        assert np.isfinite(value).all(); arrays[version+'_'+name] = value
    for phrase in ['hey_nyx', 'hey_nick', 'hey_next', 'hey_mix']:
        ids = [i for i, row in enumerate(report['sources']) if row['target'] == phrase]
        assert len(ids) == 2016
        for endpoint in range(2):
            for arm, name in enumerate(report['arms']):
                summaries[f'{version}/{phrase}/{endpoint}/{name}'] = {
                    key: stats(value[ids, endpoint, arm]) for key, value in
                    [('logit_delta', delta), ('norm_term', norm_term), ('direction_term', direction_term)]}
out.mkdir(); np.savez_compressed(out/'arrays.npz', **arrays)
result = dict(source_report_sha256=sha(root/'report.json'), source_arrays_sha256=sha(root/'arrays.npz'),
    script_sha256=sha(Path(__file__)), arrays_sha256=sha(out/'arrays.npz'), checks=checks,
    summaries=summaries, formula='delta_logit = .5*(r-rq)*(a+aq) + .5*(r+rq)*(a-aq); a=dot(e/r,w)',
    limits='Algebra on already-scored training windows; not causal attribution, counterfactual, streaming FAR, new-data evaluation or loss selection.')
(out/'report.json').write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps(checks))
for version in ['v46', 'v47']:
    print(version, json.dumps(summaries[f'{version}/hey_nyx/1/noise15']))
