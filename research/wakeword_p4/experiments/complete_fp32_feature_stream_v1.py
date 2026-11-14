"""Whole-input FP32 traversal; old UINT8 helpers and their guards stay separate."""
import hashlib

import numpy as np

POLICY = 'whole-features-final-zero-pad-stride3-fp32-v1'


def reference_interpreter(model):
    from ai_edge_litert.interpreter import Interpreter, OpResolverType
    return Interpreter(model_content=model, num_threads=1,
                       experimental_op_resolver_type=OpResolverType.BUILTIN_REF)


def predict_complete_fp32_features(model, expected_sha256, raw, samples,
                                   interpreter_factory=reference_interpreter):
    """Return all probability outputs and clocks from one fresh interpreter.

    Caller must separately admit model/material/runtime identities under a
    published execution protocol. This helper does no candidate admission,
    calibration or source selection. It consumes uint16 frontend features,
    not audio, and resets only by creating the whole-source interpreter here.
    """
    if not isinstance(model, bytes) or hashlib.sha256(model).hexdigest() != expected_sha256:
        raise ValueError('Expected the complete execution-bound model bytes')
    if type(samples) is not int or samples <= 0:
        raise ValueError('Expected positive original PCM sample count')
    frames = max(0, (samples + 159) // 160 - 2)
    if not isinstance(raw, np.ndarray) or raw.dtype != np.uint16 or raw.shape != (frames, 40):
        raise ValueError('Expected every original uint16 frontend frame')
    padding = (-frames) % 3
    padded = np.pad(raw, ((0, padding), (0, 0))) if padding else raw
    values = np.clip(padded.astype(np.float32) / 25.6, 0, 26)
    interpreter = interpreter_factory(model)
    interpreter.allocate_tensors()
    inputs, outputs = interpreter.get_input_details(), interpreter.get_output_details()
    if len(inputs) != 1 or len(outputs) != 1:
        raise ValueError('Expected one model input and output')
    source, dest = inputs[0], outputs[0]
    if (source['dtype'] != np.float32 or dest['dtype'] != np.float32
            or list(source['shape']) != [1, 3, 40] or list(dest['shape']) != [1, 1]):
        raise ValueError('Expected explicit FP32 [1,3,40] input and [1,1] output')
    for tensor in (source, dest):
        params = tensor.get('quantization_parameters', {})
        if len(params.get('scales', [])) or len(params.get('zero_points', [])):
            raise ValueError('FP32 profile must not reinterpret quantized tensors')
    calls = len(values) // 3
    probabilities = np.empty(calls, dtype=np.float32)
    for index, start in enumerate(range(0, len(values), 3)):
        interpreter.set_tensor(source['index'], values[None, start:start + 3])
        interpreter.invoke()
        output = interpreter.get_tensor(dest['index'])
        if (output.dtype != np.float32 or output.shape != (1, 1)
                or not np.isfinite(output).all() or not 0 <= output[0, 0] <= 1):
            raise ValueError('Invalid FP32 probability output')
        probabilities[index] = output[0, 0]
    support = (3 * np.arange(1, calls + 1, dtype=np.int64) + 2) * 160
    clocks = np.minimum(support, samples)
    metadata = dict(policy=POLICY, original_samples=samples,
        original_feature_frames=frames, final_feature_padding=padding,
        input_feature_frames=len(values), raw_steps=calls,
        normalized_input_sha256=hashlib.sha256(values.tobytes()).hexdigest(),
        clock='one-based s: min((3*s+2)*160, original_samples); no phase shift',
        output_encoding='float32 probability; no quantization')
    return probabilities, clocks, metadata
