"""Certify later actual smoke initialization as derived original-probe definition.

Not an original historical probe snapshot. No model updates or probe rerun.
"""
import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path
import numpy as np
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')
tf.config.threading.set_intra_op_parallelism_threads(2)
base = Path('.local/wakeword')
smoke = base/'model-paired-control-smoke-v1'
out = base/'derived-probe-initialization-v1'; assert not out.exists()
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
initializer = base/'mww-pretrained-reconstruction-v1/reconstructed.weights.h5'
assert sha(initializer) == '8e7de8447bb95660844d687c5e44cd7d5c93f3106d08681e2ceda7cd6a28643b'
training = json.loads((smoke/'training.json').read_text())
sys.path.insert(0, str((base/'microWakeWord').resolve()))
from microwakeword import mixednet
parser = argparse.ArgumentParser(); mixednet.model_parameters(parser)
model = mixednet.model(parser.parse_args(training['architecture_args']), (150, 40), None)
head = model.layers[-1]; head_ids = {id(v) for v in head.variables}
model.load_weights(initializer)
initial_values = [v.numpy().copy() for v in model.variables]
snapshot = smoke/'actual-initialization.weights.h5'
model.load_weights(snapshot)
compared = 0
for original, variable in zip(initial_values, model.variables):
    if id(variable) not in head_ids:
        np.testing.assert_array_equal(original, variable.numpy()); compared += 1
head_values = head.get_weights()
head_sha = hashlib.sha256(b''.join(v.tobytes() for v in head_values)).hexdigest()
assert head_sha == 'a41cc84a5e15a42a13ee9fe02e38242fca4dd9afe67a19ba736587970b46fc54'
inventory = [dict(path=v.path, shape=list(v.shape), dtype=str(v.dtype),
    value_sha256=hashlib.sha256(v.numpy().tobytes()).hexdigest()) for v in model.variables]
assert inventory == json.loads((smoke/'actual-initialization-inventory.json').read_text())
out.mkdir()
shutil.copyfile(snapshot, out/'derived-probe-initialization.weights.h5')
shutil.copyfile(initializer, out/'original-reconstructed.weights.h5')
np.savez_compressed(out/'certified-head.npz', kernel=head_values[0], bias=head_values[1])
(out/'variable-inventory.json').write_text(json.dumps(inventory, indent=2)+'\n')
result = dict(role='Derived restoration of original probe numerical initialization definition; not original probe instance snapshot',
    script_sha256=sha(Path(__file__)), encoder_initializer_sha256=sha(initializer), head_value_sha256=head_sha,
    derived_h5_sha256=sha(out/'derived-probe-initialization.weights.h5'), head_npz_sha256=sha(out/'certified-head.npz'),
    inventory_sha256=sha(out/'variable-inventory.json'), nonhead_variables_exactly_equal_initializer=compared,
    original_probe_initialization_sha256=sha(base/'paired-consistency-init-gradient-probe-v1/initialization.json'),
    source_snapshot=str(snapshot), source_snapshot_inventory_exact=True,
    limits='Certified nonhead initializer equality and original head hash, not historical probe-instance provenance or independent original-gradient reproduction. No reinitialization seed is used to recover head values; later saved pre-update snapshot supplies certified head.')
(out/'certification.json').write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps(result))
