"""Frozen v30 same-checkpoint float/INT8 on consumed Mac350; no selection."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import soundfile as sf
import tensorflow as tf
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'training'))
from features import features
from evaluate import event_indices, smooth

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

tf.config.set_visible_devices([], 'GPU')
tf.config.threading.set_intra_op_parallelism_threads(2)
tf.config.threading.set_inter_op_parallelism_threads(2)
model = Path('.local/wakeword/model-gpu-v30-ami-extra')
root = Path('.local/wakeword/mac-evidence-import-v1/extracted')
sources = root/'workspace/hey-nyx-training-v3/paired-mac-regression-v1'
output = Path('.local/wakeword/mac-float-int8-v1')
assert not output.exists()
assert sha(model/'hey_nyx.tflite') == '04aa45646610f0e3a866a17d4098263adf5da912bb6ddfe03c44807a295baa0c'
assert sha(sources/'manifest.jsonl') == 'c776e3790d4df96774e5a9b4b3d0526135621d93e69c6caa25959f9772f0b39f'
cpu_path = Path('.local/wakeword/mac-cpu-replay-v1')
cpu = json.loads((cpu_path/'result.json').read_text())
assert sha(cpu_path/'scores.npz') == cpu['scores_sha256']
quantized = np.load(cpu_path/'scores.npz', allow_pickle=False)
loaded = tf.saved_model.load(str(model/'saved_model'))
signature = loaded.signatures['serving_default']
keys = list(signature.structured_input_signature[1]); assert len(keys) == 1
states = [v for v in loaded.variables if 'states' in v.name]; assert states

def predict(raw):
    for state in states:
        state.assign(tf.zeros_like(state))
    x = np.clip(raw.astype(np.float32)/25.6, 0, 26)
    scores = []
    for at in range(0, len(x)-2, 3):
        result = signature(**{keys[0]: tf.convert_to_tensor(x[None,at:at+3])})
        assert len(result) == 1
        scores.append(float(next(iter(result.values())).numpy().reshape(-1)[0]))
    return np.asarray(scores, dtype=np.float32)

def events(scores):
    # Original host float64 convolution; report one-based raw-score steps.
    return [int(i)+5 for i in event_indices(smooth(scores, 5), .91796875)]

output.mkdir()
rows = [json.loads(line) for line in (sources/'manifest.jsonl').read_text().splitlines()]
assert len(rows) == 350
details, arrays = [], {}
for i, row in enumerate(rows):
    path = sources/row['path']; assert sha(path) == row['sha256']
    pcm, rate = sf.read(path, dtype='int16')
    assert rate == 16000 and pcm.ndim == 1 and len(pcm) == row['samples']
    raw = features(np.pad(pcm, (56160, (-len(pcm)) % 480)).astype('<i2'))
    assert hashlib.sha256(raw.tobytes()).hexdigest() == cpu['details'][i]['frontend_sha256']
    f = predict(raw); q = quantized[f'output_{i}'].astype(np.float32)/256
    assert f.shape == q.shape and np.isfinite(f).all()
    if i == 0:
        np.testing.assert_array_equal(f, predict(raw))
    arrays[f'float_{i}'] = f
    fe, qe = events(f), events(q)
    details.append({**row, 'float_events': fe, 'int8_events': qe,
        'float_source_events': [s for s in fe if 480*s-56160 > 0],
        'int8_source_events': [s for s in qe if 480*s-56160 > 0],
        'float_score_sha256': hashlib.sha256(f.tobytes()).hexdigest(),
        'output_count': len(f), 'mae': float(np.mean(np.abs(f-q))),
        'max_error': float(np.max(np.abs(f-q))),
        'float_max_five_mean': float(max(smooth(f))),
        'int8_max_five_mean': float(max(smooth(q)))})
    if i % 25 == 0:
        print(json.dumps({'completed': i+1, 'total': len(rows)}), flush=True)
np.savez_compressed(output/'scores.npz', **arrays)
summary = {}
for target in sorted({r['target'] for r in rows}):
    group = [d for d in details if d['target'] == target]
    summary[target] = {'clips': len(group),
        'float_trigger_clips': sum(bool(d['float_source_events']) for d in group),
        'int8_trigger_clips': sum(bool(d['int8_source_events']) for d in group),
        'float_only': sum(bool(d['float_source_events']) and not d['int8_source_events'] for d in group),
        'int8_only': sum(bool(d['int8_source_events']) and not d['float_source_events'] for d in group)}
result = {'scope': 'Consumed Mac350 fixed-threshold diagnostic, not acceptance or threshold selection.',
    'model_sha256': sha(model/'hey_nyx.tflite'), 'training_sha256': sha(model/'training.json'),
    'saved_model_files': {str(p.relative_to(model)): sha(p) for p in sorted((model/'saved_model').rglob('*')) if p.is_file()},
    'script_sha256': sha(Path(__file__)), 'tensorflow': tf.__version__,
    'reset_policy': 'All float state buffers zeroed per stream; same Linux native frontend authenticated against prior CPU replay; first stream reset repeat exact.',
    'threshold': .91796875, 'smoothing_steps': 5,
    'int8_scores_sha256': sha(cpu_path/'scores.npz'),
    'float_scores_sha256': sha(output/'scores.npz'), 'summary': summary, 'details': details,
    'limits': 'Same-checkpoint original SavedModel, not a retrain. Original frozen INT8 threshold is descriptive for float; no float optimum/FAR inference. No P4 or new/final data. Synthetic intention labels do not prove phonetic correctness.'}
(output/'result.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(summary), flush=True)
