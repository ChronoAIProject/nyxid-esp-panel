"""Fixed first/last paired block phase/reset preflight; no model updates."""
import argparse
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
import tensorflow as tf

base = Path('.local/wakeword'); root = base/'model-gpu-v47-ami-extra'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(root/'hey_nyx.tflite') == 'c6c9037c50c66230a783788678328d12a73e011dec5c97b94f74deba5e58cdd9'
schedule_path = Path('research/wakeword_p4/results/4090-paired-factorized-schedule-precheck-v1.json')
schedule = json.loads(schedule_path.read_text()); assert len(schedule['blocks']) == 50
pack = base/'factorized-training-features-v1'
assert sha(pack/'report.json') == schedule['report_sha256']
records = json.loads((pack/'report.json').read_text())['records']
training = json.loads((root/'training.json').read_text()); assert training['validation_window_phase'] == 1
sys.path.insert(0, str((base/'microWakeWord').resolve()))
from microwakeword import mixednet
parser = argparse.ArgumentParser(); mixednet.model_parameters(parser)
model = mixednet.model(parser.parse_args(training['architecture_args']), (150, 40), None)
model.load_weights(root/'best.weights.h5')
loaded = tf.saved_model.load(str(root/'saved_model'))
signature = loaded.signatures['serving_default']
key = next(iter(signature.structured_input_signature[1]))
states = [v for v in loaded.variables if 'states' in v.name]; assert states
weight_before = sha(root/'best.weights.h5')
details = []
for block in [schedule['blocks'][0], schedule['blocks'][-1]]:
    end = block['endpoint_inclusive']; assert end%3 == 0
    for view in block['views']:
        row = records[view['pack_index']]; path = pack/row['raw']
        assert sha(path) == view['raw_file_sha256']
        raw = np.load(path, allow_pickle=False)
        assert hashlib.sha256(raw[end-149:end+1].tobytes()).hexdigest() == view['window_sha256']
        x = np.clip(raw.astype(np.float32)/25.6, 0, 26)
        probability = float(model(x[None, end-149:end+1], training=False).numpy().reshape(-1)[0])
        scores = []
        for repeat in range(2):
            for state in states: state.assign(tf.zeros_like(state))
            sequence = []
            for at in range(0, end, 3):
                y = signature(**{key: tf.convert_to_tensor(x[None, at:at+3])})
                sequence.append(float(next(iter(y.values())).numpy().reshape(-1)[0]))
            scores.append(np.asarray(sequence))
        np.testing.assert_allclose(scores[0], scores[1], rtol=0, atol=1e-7)
        np.testing.assert_allclose(probability, scores[0][-1], rtol=1e-5, atol=1e-5)
        details.append(dict(block=block['block'], pack_index=view['pack_index'], endpoint_inclusive=end,
            window_probability=probability, stream_probability=float(scores[0][-1]),
            absolute_error=abs(probability-scores[0][-1]), reset_max_error=float(np.abs(scores[0]-scores[1]).max())))
    print(json.dumps(dict(completed_views=len(details), planned=80)), flush=True)
assert len(details) == 80 and sha(root/'best.weights.h5') == weight_before
result = dict(model_sha256=sha(root/'hey_nyx.tflite'), weights_sha256=weight_before,
    schedule_sha256=sha(schedule_path), script_sha256=sha(Path(__file__)), views=80,
    max_probability_error=max(r['absolute_error'] for r in details),
    max_reset_error=max(r['reset_max_error'] for r in details), details=details,
    limits='First/last paired blocks only, frozen v47 float phase/reset preflight; not all-source equivalence, training, INT8/FAR acceptance or Mac independent v46 measurement.')
Path('research/wakeword_p4/results/4090-v47-paired-phase-preflight-v1.json').write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k != 'details'}), flush=True)
