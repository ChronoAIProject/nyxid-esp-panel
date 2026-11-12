"""Recompute frozen embedding diagnostic summaries from original saved arrays."""
import hashlib
import json
from pathlib import Path
import numpy as np

root = Path('.local/wakeword/factorized-embedding-diagnostic-v1')
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
r = json.loads((root/'report.json').read_text())
assert sha(root/'arrays.npz') == r['arrays_sha256']
assert len(r['sources']) == 8064 and r['raw_files_authenticated'] == 40320
z = np.load(root/'arrays.npz', allow_pickle=False)
assert np.all(z['endpoints'][:, 0] <= z['endpoints'][:, 1])
lookup = {(a['voice'], a['length_multiplier'], a['target']): i for i, a in enumerate(r['sources'])}
assert len(lookup) == 8064
checked = 0
def check_stats(a, expected):
    global checked
    assert np.size(a) == expected['count']
    np.testing.assert_allclose(np.quantile(a, [0, .05, .5, .95, 1]), expected['quantiles'], rtol=1e-12, atol=1e-12)
    checked += 1

for version in ['v46', 'v47']:
    e = z[version+'_embedding']; p = z[version+'_probability']
    assert e.shape == (8064, 2, 5, 300) and p.shape == (8064, 2, 5)
    assert np.isfinite(e).all() and np.isfinite(p).all()
    model = r['models'][version]
    logits = (e@np.asarray(model['head_kernel'], dtype=np.float32)+np.asarray(model['head_bias'], dtype=np.float32))[..., 0]
    np.testing.assert_array_equal(logits, z[version+'_logit'])
    sigmoid = 1/(1+np.exp(-logits.astype(np.float64)))
    np.testing.assert_allclose(sigmoid, p, atol=1e-6, rtol=1e-5)
    norms = np.sqrt(np.sum(e.astype(np.float64)**2, axis=-1))
    np.testing.assert_allclose(norms, z[version+'_norm'], atol=1e-12, rtol=1e-12)
    assert int(np.sum(norms == 0)) == model['zero_norms'] == 0
    unit = e.astype(np.float64)/norms[..., None]
    cosine = np.einsum('sead,sed->sea', unit, unit[:, :, 0])
    np.testing.assert_allclose(cosine, z[version+'_quiet_cosine'], atol=1e-12, rtol=1e-12)
    for phrase in ['hey_nyx', 'hey_nick', 'hey_next', 'hey_mix']:
        ids = [i for i, a in enumerate(r['sources']) if a['target'] == phrase]
        assert len(ids) == 2016
        for endpoint in range(2):
            for arm, name in enumerate(r['arms']):
                group = r['summaries'][version][f'{phrase}/{endpoint}/{name}']
                for key, data in [('norm', norms), ('logit', logits), ('quiet_cosine', cosine)]:
                    check_stats(data[ids, endpoint, arm], group[key])
    positive = [i for i, a in enumerate(r['sources']) if a['target'] == 'hey_nyx']
    for phrase in ['hey_nick', 'hey_next', 'hey_mix']:
        negative = [lookup[r['sources'][i]['voice'], r['sources'][i]['length_multiplier'], phrase] for i in positive]
        margin = logits[positive]-logits[negative]
        pair_cosine = np.einsum('sead,sead->sea', unit[positive], unit[negative])
        np.testing.assert_array_equal(margin, z[version+'_'+phrase+'_logit_margin'])
        np.testing.assert_allclose(pair_cosine, z[version+'_'+phrase+'_target_cosine'], atol=1e-12, rtol=1e-12)
        for endpoint in range(2):
            for arm, name in enumerate(r['arms']):
                group = r['summaries'][version][f'target_vs_{phrase}/{endpoint}/{name}']
                check_stats(margin[:, endpoint, arm], group['logit_margin'])
                check_stats(pair_cosine[:, endpoint, arm], group['target_cosine'])
                check_stats(margin[:, endpoint, arm]-margin[:, endpoint, 0], group['logit_margin_change_from_quiet'])
audit = dict(report_sha256=sha(root/'report.json'), arrays_sha256=r['arrays_sha256'],
             script_sha256=sha(Path(__file__)), verified_models=2, windows_per_model=80640,
             embedding_dimensions=300, all_arrays_finite=True, zero_norms=0,
             all_saved_heads_reconstruct=True, summary_quantile_sets_recomputed=checked,
             limits='Saved-array arithmetic audit; not independent model inference or raw/frontend regeneration.')
Path('research/wakeword_p4/results/4090-factorized-embedding-audit-v1.json').write_text(json.dumps(audit, indent=2)+'\n')
print(json.dumps(audit))
