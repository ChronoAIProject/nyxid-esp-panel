"""Authenticate completed streaming export before any downstream scoring."""
import argparse
import hashlib
import json
from pathlib import Path
import h5py
import numpy as np
from ai_edge_litert.interpreter import Interpreter


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-dir', type=Path, required=True)
    p.add_argument('--expected-steps', type=int, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        p.error('Fresh audit output required')
    root = a.model_dir
    training = json.loads((root/'training.json').read_text())
    model = root/'hey_nyx.tflite'
    assert training['steps'] == a.expected_steps
    assert training['model_sha256'] == sha(model)
    assert training['batchnorm_policy'] == 'frozen'
    assert training['batchnorm_initial'] == training['batchnorm_selected']
    assert training['parameters'] == 22801
    with h5py.File(root/'best.weights.h5', 'r') as f:
        def check(name, item):
            if isinstance(item, h5py.Dataset):
                assert np.isfinite(item[()]).all(), name
        f.visititems(check)
    interpreter = Interpreter(model_path=str(model), num_threads=1)
    interpreter.allocate_tensors()
    inputs = interpreter.get_input_details()
    outputs = interpreter.get_output_details()
    assert len(inputs) == len(outputs) == 1
    assert inputs[0]['shape'].tolist() == [1, 3, 40]
    assert outputs[0]['shape'].tolist() == [1, 1]
    assert inputs[0]['dtype'] == np.int8 and outputs[0]['dtype'] == np.uint8
    assert inputs[0]['quantization'][0] > 0 and outputs[0]['quantization'][0] > 0
    report = dict(model_sha256=sha(model), model_bytes=model.stat().st_size,
                  training_sha256=sha(root/'training.json'),
                  weights_sha256=sha(root/'best.weights.h5'),
                  steps=training['steps'], selected_step=training['selected_step'],
                  seconds=training['seconds'], parameters=training['parameters'],
                  gpu_details=training['gpu_details'], all_saved_weights_finite=True,
                  frozen_bn_unchanged=True, input_shape=[1, 3, 40],
                  output_shape=[1, 1], input_dtype='int8', output_dtype='uint8',
                  source_sha256=sha(Path(__file__)), candidate_promoted=False,
                  limits='Export contract only; not accuracy, FAR or MCU parity.')
    a.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
