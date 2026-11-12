"""All-source frozen window embedding diagnostic; see preregistered protocol."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

base = Path('.local/wakeword')
pack = base / 'factorized-training-features-v1'
out = base / 'factorized-embedding-diagnostic-v1'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
assert not out.exists(), 'Review original output; do not overwrite'
assert sha(pack/'report.json') == '7b798653f73e2dcd3b01fd5b7092c947a26b9c8ec2525ff295de2e6478042e44'
metadata = json.loads((pack/'report.json').read_text())
records = metadata['records']
assert len(records) == 40320 and not metadata['final_data_used']
arms = ['quiet', 'noise25', 'combined', 'noise15', 'rir']
windows = np.empty((8064, 2, 5, 150, 40), dtype=np.uint16)
endpoints = np.empty((8064, 2), dtype=np.int32)
source_rows = []
for i in range(8064):
    rows = records[5*i:5*i+5]
    assert [r['arm'] for r in rows] == arms
    assert all(r['source_index'] == i and r['split'] == 'training' for r in rows)
    assert len({(r['source'], r['voice'], r['target'], r['length_multiplier']) for r in rows}) == 1
    low = max(r['feature_crop_end'] for r in rows)
    high = min(350, *(r['raw_shape'][0] for r in rows))
    assert 150 <= low <= high
    endpoints[i] = [low, high]
    for j, r in enumerate(rows):
        path = pack/r['raw']
        assert sha(path) == r['raw_file_sha256']
        raw = np.load(path, allow_pickle=False)
        assert list(raw.shape) == r['raw_shape'] and raw.dtype == np.uint16
        assert hashlib.sha256(raw.tobytes()).hexdigest() == r['raw_sha256']
        for e, end in enumerate([low, high]):
            assert (end-150)*160 <= 32000 and (end+1)*160 >= 32000+r['signal_samples']
            windows[i, e, j] = raw[end-150:end]
    source_rows.append({k: rows[0][k] for k in ['source_index', 'source', 'voice', 'target', 'length_multiplier']})
    if i % 1000 == 0:
        print(json.dumps(dict(stage='authenticate_windows', sources=i+1)), flush=True)

sys.path.insert(0, str((base/'microWakeWord').resolve()))
from microwakeword import mixednet

out.mkdir()
arrays = dict(endpoints=endpoints)
report = dict(protocol_sha256=sha(Path(__file__).with_suffix('.md')),
              script_sha256=sha(Path(__file__)), pack_report_sha256=sha(pack/'report.json'),
              arms=arms, endpoints=['earliest_common', 'latest_common'], sources=source_rows,
              raw_files_authenticated=40320, models={}, summaries={}, final_data_used=False,
              limits='Training fit only; fixed window embeddings, not streaming acceptance, causal mechanism or loss selection.')
pins = {'v46': '5ac62943e862b2a8b7c92d97b1b4e89a096a2bb9d76f27b1b4b72b25a427f417',
        'v47': 'c6c9037c50c66230a783788678328d12a73e011dec5c97b94f74deba5e58cdd9'}
def stats(a):
    a = np.asarray(a)
    return dict(count=int(a.size), quantiles=np.quantile(a, [0, .05, .5, .95, 1]).tolist())

for version, digest in pins.items():
    tf.keras.backend.clear_session()
    root = base/f'model-gpu-{version}-ami-extra'
    assert sha(root/'hey_nyx.tflite') == digest
    training = json.loads((root/'training.json').read_text())
    parser = argparse.ArgumentParser(); mixednet.model_parameters(parser)
    model = mixednet.model(parser.parse_args(training['architecture_args']), (150, 40), None)
    model.load_weights(root/'best.weights.h5')
    head = model.layers[-1]
    assert isinstance(head, tf.keras.layers.Dense) and head.units == 1
    assert isinstance(model.layers[-2], tf.keras.layers.Flatten)
    extractor = tf.keras.Model(model.inputs, [head.input, model.output])
    kernel, bias = head.get_weights()
    flat = windows.reshape(-1, 150, 40)
    embeddings, probabilities = [], []
    max_error = 0.
    for at in range(0, len(flat), 128):
        x = np.clip(flat[at:at+128].astype(np.float32)/25.6, 0, 26)
        e, p = extractor(x, training=False)
        e, p = e.numpy(), p.numpy().reshape(-1)
        logits = e@kernel+bias
        recovered = tf.math.sigmoid(logits).numpy().reshape(-1)
        np.testing.assert_allclose(recovered, p, rtol=1e-5, atol=1e-6)
        assert np.isfinite(e).all() and np.isfinite(p).all()
        max_error = max(max_error, float(np.max(np.abs(recovered-p))))
        embeddings.append(e); probabilities.append(p)
        if at % 12800 == 0:
            print(json.dumps(dict(stage=version, windows=min(at+128, len(flat)), total=len(flat))), flush=True)
    e = np.concatenate(embeddings).reshape(8064, 2, 5, -1)
    p = np.concatenate(probabilities).reshape(8064, 2, 5)
    logits = (e@kernel+bias)[..., 0]
    norms = np.linalg.norm(e.astype(np.float64), axis=-1)
    unit = e.astype(np.float64)/np.maximum(norms[..., None], 1e-12)
    cosine = np.sum(unit*unit[:, :, :1], axis=-1)
    arrays.update({version+'_embedding': e, version+'_probability': p,
                   version+'_logit': logits, version+'_norm': norms,
                   version+'_quiet_cosine': cosine})
    report['models'][version] = dict(model_sha256=digest, weights_sha256=sha(root/'best.weights.h5'),
        training_sha256=sha(root/'training.json'), selected_step=training['selected_step'],
        head_kernel=kernel.tolist(), head_bias=bias.tolist(), embedding_shape=list(e.shape),
        zero_norms=int(np.sum(norms == 0)), head_reconstruction_max_error=max_error)
    groups = report['summaries'][version] = {}
    for phrase in ['hey_nyx', 'hey_nick', 'hey_next', 'hey_mix']:
        ids = [i for i, r in enumerate(source_rows) if r['target'] == phrase]
        assert len(ids) == 2016
        for endpoint in range(2):
            for arm, name in enumerate(arms):
                groups[f'{phrase}/{endpoint}/{name}'] = dict(norm=stats(norms[ids, endpoint, arm]),
                    logit=stats(logits[ids, endpoint, arm]), quiet_cosine=stats(cosine[ids, endpoint, arm]))
    lookup = {(r['voice'], r['length_multiplier'], r['target']): i for i, r in enumerate(source_rows)}
    positive = [i for i, r in enumerate(source_rows) if r['target'] == 'hey_nyx']
    for phrase in ['hey_nick', 'hey_next', 'hey_mix']:
        negative = [lookup[source_rows[i]['voice'], source_rows[i]['length_multiplier'], phrase] for i in positive]
        margin = logits[positive]-logits[negative]
        pair_cosine = np.sum(unit[positive]*unit[negative], axis=-1)
        arrays[version+'_'+phrase+'_logit_margin'] = margin
        arrays[version+'_'+phrase+'_target_cosine'] = pair_cosine
        for endpoint in range(2):
            for arm, name in enumerate(arms):
                groups[f'target_vs_{phrase}/{endpoint}/{name}'] = dict(
                    logit_margin=stats(margin[:, endpoint, arm]),
                    target_cosine=stats(pair_cosine[:, endpoint, arm]),
                    logit_margin_change_from_quiet=stats(margin[:, endpoint, arm]-margin[:, endpoint, 0]))
    print(json.dumps(dict(stage=version, embedding_shape=list(e.shape), head_error=max_error)), flush=True)
np.savez_compressed(out/'arrays.npz', **arrays)
report['arrays_sha256'] = sha(out/'arrays.npz')
(out/'report.json').write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps(dict(completed=True, arrays_sha256=report['arrays_sha256'])), flush=True)
