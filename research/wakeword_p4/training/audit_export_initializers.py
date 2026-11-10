"""Compare TFLite resource initializers with the exact saved float states.

This checks initialization constants, not inference accuracy or acceptance.
No waveform, wake score, classifier threshold or final corpus is consumed.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.lite.python import schema_py_generated as schema


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-dir', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        p.error('Fresh report required')
    tf.config.set_visible_devices([], 'GPU')
    loaded = tf.saved_model.load(str(a.model_dir/'saved_model'))
    states = {v.name: v.numpy() for v in loaded.variables if 'states' in v.name}
    if not states:
        raise ValueError('No saved float stream states')
    path = a.model_dir/'hey_nyx.tflite'
    model = schema.Model.GetRootAsModel(path.read_bytes(), 0)
    records, seen = [], set()
    for gi in range(1, model.SubgraphsLength()):
        graph = model.Subgraphs(gi)
        handles = {}
        for oi in range(graph.OperatorsLength()):
            op = graph.Operators(oi)
            code = model.OperatorCodes(op.OpcodeIndex()).BuiltinCode()
            if code == schema.BuiltinOperator.VAR_HANDLE:
                table = op.BuiltinOptions()
                options = schema.VarHandleOptions()
                options.Init(table.Bytes, table.Pos)
                if op.OutputsLength() != 1:
                    raise ValueError('Unexpected handle outputs')
                handles[op.Outputs(0)] = options.SharedName().decode()
        for oi in range(graph.OperatorsLength()):
            op = graph.Operators(oi)
            if model.OperatorCodes(op.OpcodeIndex()).BuiltinCode() != schema.BuiltinOperator.ASSIGN_VARIABLE:
                continue
            if op.InputsLength() != 2 or op.Inputs(0) not in handles:
                raise ValueError('Unexpected initializer assignment')
            handle = handles[op.Inputs(0)]
            candidates = [name for name in states
                          if name.removesuffix(':0') in (handle, handle.removesuffix('_1'))]
            if len(candidates) != 1 or candidates[0] in seen:
                raise ValueError('Ambiguous or repeated state handle')
            name = candidates[0]
            seen.add(name)
            tensor = graph.Tensors(op.Inputs(1))
            q = tensor.Quantization()
            if tensor.Type() != schema.TensorType.INT8 or q.ScaleLength() != 1 or q.ZeroPointLength() != 1:
                raise ValueError('Expected per-tensor INT8 state initializer')
            shape = tuple(int(v) for v in tensor.ShapeAsNumpy())
            values = model.Buffers(tensor.Buffer()).DataAsNumpy()
            if not isinstance(values, np.ndarray) or shape != states[name].shape:
                raise ValueError('Missing initializer constant or mismatched shape')
            values = values.view(np.int8).reshape(shape)
            scale, zero = float(q.Scale(0)), int(q.ZeroPoint(0))
            if scale <= 0:
                raise ValueError('Invalid state quantization scale')
            dequantized = (values.astype(np.float64)-zero)*scale
            error = np.abs(dequantized-states[name])
            expected = np.clip(np.rint(states[name].astype(np.float64)/scale)+zero, -128, 127).astype(np.int8)
            records.append(dict(float_state=name, tflite_handle=handle, shape=shape,
                                float_sha256=hashlib.sha256(states[name].tobytes()).hexdigest(),
                                initializer_sha256=hashlib.sha256(values.tobytes()).hexdigest(),
                                scale=scale, zero_point=zero,
                                float_nonzero=int(np.count_nonzero(states[name])),
                                dequantized_nonzero=int(np.count_nonzero(dequantized)),
                                max_initial_value_error=float(error.max()),
                                half_quantization_step=scale/2,
                                within_half_step=bool(np.all(error <= scale/2+1e-6)),
                                exactly_nearest_quantized_saved_values=bool(np.array_equal(expected, values))))
    if seen != set(states):
        raise ValueError('Incomplete initial state coverage')
    result = dict(role='Saved float versus TFLite initializer audit, not numerical inference equivalence',
                  source_sha256=sha(Path(__file__)), model_sha256=sha(path), tensorflow=tf.__version__,
                  saved_model_files_sha256={str(p.relative_to(a.model_dir/'saved_model')): sha(p)
                      for p in sorted((a.model_dir/'saved_model').rglob('*')) if p.is_file()},
                  records=records, states=len(records), all_within_half_step=all(r['within_half_step'] for r in records),
                  all_exact_nearest_values=all(r['exactly_nearest_quantized_saved_values'] for r in records),
                  audio_accessed=False, wake_scores_accessed=False, independent_final_audio_accessed=False,
                  limits='Only exported initial constants and saved float buffers. Matching nearest INT8 initializers does not prove the runtime buffers, reset API, streaming kernels, quantized predictions or MCU behavior correct. Large inference drift can remain after normal initial-value rounding; no samples/events discarded and no acceptance or latency claim.')
    a.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k: result[k] for k in ('states', 'all_within_half_step', 'all_exact_nearest_values')}))


if __name__ == '__main__':
    main()
