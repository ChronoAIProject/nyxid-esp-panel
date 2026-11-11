"""Independent CPU replay of consumed Mac350 data; no calibration or training."""
import hashlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import platform
import sys

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'training'))
from evaluate import Detector
from features import features

def sha(data):
    return hashlib.sha256(data).hexdigest()

root = Path('.local/wakeword/mac-evidence-import-v1/extracted')
sources = root / 'workspace/hey-nyx-training-v3/paired-mac-regression-v1'
expected = json.loads((root / 'study/result.json').read_text())
manifest = (sources / 'manifest.jsonl').read_bytes()
assert sha(manifest) == 'c776e3790d4df96774e5a9b4b3d0526135621d93e69c6caa25959f9772f0b39f'
model = root / 'model/hey_nyx.tflite'
assert len(model.read_bytes()) == 54440
assert sha(model.read_bytes()) == '04aa45646610f0e3a866a17d4098263adf5da912bb6ddfe03c44807a295baa0c'
reference = np.load(root / 'study/scores.npz', allow_pickle=False)
rows = [json.loads(line) for line in manifest.splitlines()]
assert len(rows) == len(expected['source_details']) == 350
detector = Detector(model)
details, arrays = [], {}
for i, (row, exp) in enumerate(zip(rows, expected['source_details'], strict=True)):
    assert row['path'] == exp['path']
    path = sources / row['path']
    assert sha(path.read_bytes()) == row['sha256']
    pcm, rate = sf.read(path, dtype='int16')
    assert rate == 16000 and pcm.ndim == 1 and len(pcm) == row['samples']
    padded = np.pad(pcm, (56160, (-len(pcm)) % 480)).astype('<i2')
    raw = features(padded)
    detector.reset()
    scores = detector.predict(raw)
    quantized = np.rint(scores * 256).astype(np.uint8)
    np.testing.assert_array_equal(scores, quantized.astype(np.float32)/256)
    ref = reference[f'output_{i}']
    same_shape = quantized.shape == ref.shape
    difference = np.flatnonzero(quantized != ref) if same_shape else []
    details.append({'path': row['path'], 'frontend_sha256': sha(raw.tobytes()),
        'frontend_shape': list(raw.shape), 'frontend_dtype': str(raw.dtype),
        'frontend_exact': sha(raw.tobytes()) == exp['frontend_sha256'] and list(raw.shape) == exp['frontend_shape'] and str(raw.dtype) == exp['frontend_dtype'],
        'output_sha256': sha(quantized.tobytes()), 'output_count': len(quantized),
        'output_exact': bool(same_shape and np.array_equal(quantized, ref)),
        'first_output_difference': None if same_shape and not len(difference) else
            {'index': int(difference[0]), 'cpu': int(quantized[difference[0]]), 'mac': int(ref[difference[0]])} if len(difference) else {'shape_mismatch': True}})
    arrays[f'output_{i}'] = quantized
    if i % 25 == 0:
        print(json.dumps({'completed': i+1}), flush=True)
output = Path('.local/wakeword/mac-cpu-replay-v1')
output.mkdir(exist_ok=False)
np.savez_compressed(output / 'scores.npz', **arrays)
result = {'scope': 'Consumed Mac350 cross-host CPU diagnostic only; no new acceptance, calibration, or P4 replay.',
    'platform': platform.platform(), 'python': sys.version,
    'versions': {name: metadata.version(name) for name in ('numpy', 'scipy', 'soundfile', 'pymicro-features', 'ai-edge-litert')},
    'model_sha256': sha(model.read_bytes()), 'manifest_sha256': sha(manifest),
    'reference_result_sha256': sha((root/'study/result.json').read_bytes()),
    'reference_scores_sha256': sha((root/'study/scores.npz').read_bytes()),
    'script_sha256': sha(Path(__file__).read_bytes()),
    'sources': len(details), 'outputs': sum(d['output_count'] for d in details),
    'all_frontends_exact': all(d['frontend_exact'] for d in details),
    'all_outputs_exact': all(d['output_exact'] for d in details),
    'scores_sha256': sha((output/'scores.npz').read_bytes()), 'details': details}
(output/'result.json').write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps({k: v for k, v in result.items() if k != 'details'}), flush=True)
