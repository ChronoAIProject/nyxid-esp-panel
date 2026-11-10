"""Check CUDA fitting, INT8 export and reference inference without corpus data."""
import json
import argparse
from pathlib import Path
import sys
import tempfile

import numpy as np
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'wakeword_p4/embedding'))
from train_head import make_model, export
from ai_edge_litert.interpreter import Interpreter, OpResolverType


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--allow-cpu', action='store_true', help='Check code on a CPU host; does not verify CUDA')
    args = parser.parse_args()
    devices = tf.config.list_physical_devices('GPU')
    if not devices and not args.allow_cpu:
        raise RuntimeError('No TensorFlow GPU detected; check the driver and CUDA environment')
    for device in devices:
        tf.config.experimental.set_memory_growth(device, True)
    tf.keras.utils.set_random_seed(727)
    rng = np.random.default_rng(727)
    x = rng.uniform(-1, 127 / 128, (128, 16, 96, 1)).astype(np.float32)
    y = rng.integers(0, 2, (128, 1)).astype(np.float32)
    model = make_model('temporal32-bn-dropout')
    model.compile(optimizer='adam', loss='binary_crossentropy')
    for _ in range(3):
        loss = model.train_on_batch(x, y)
    with tempfile.TemporaryDirectory(prefix='nyx-gpu-smoke-') as name:
        path = Path(name)
        report = export(model, path, x)
        interpreter = Interpreter(model_path=str(path / 'head.tflite'),
                                  experimental_op_resolver_type=OpResolverType.BUILTIN_REF)
        interpreter.allocate_tensors()
        input_info = interpreter.get_input_details()[0]
        interpreter.set_tensor(input_info['index'], np.rint(x[:1] * 128).clip(-128, 127).astype(np.int8))
        interpreter.invoke()
    print(json.dumps({'gpus': [d.name for d in devices], 'cuda_verified': bool(devices),
                      'loss': float(loss), 'export': report,
                      'scope': 'Random-data environment smoke only; no wake accuracy result'}))


if __name__ == '__main__':
    main()
