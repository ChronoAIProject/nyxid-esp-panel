"""Replace unit-stride INT8 SAME convolution padding with PAD + VALID.

ESP-NN 1.1.2's P4 dispatcher uses its scalar fallback for implicit padding.
This graph rewrite makes the identical zero-point padding explicit so the
existing vector path can be measured. It changes no learned bytes or cadence.
Host equality is necessary; the generated LUT graph still needs P4 proof.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
from tensorflow.lite.python import schema_py_generated as s
from ai_edge_litert.interpreter import Interpreter, OpResolverType

from fuse_activations import fold, pack


def transform(data):
    model = s.ModelT.InitFromObj(s.Model.GetRootAsModel(data, 0))
    before_buffers = [None if b.data is None else b.data.tobytes() for b in model.buffers]
    code = s.OperatorCodeT()
    code.builtinCode = code.deprecatedBuiltinCode = s.BuiltinOperator.PAD
    code.version = 2  # INT8 PAD
    pad_code = len(model.operatorCodes)
    model.operatorCodes.append(code)
    changes = []
    for graph_index, graph in enumerate(model.subgraphs):
        rewritten = []
        for op_index, op in enumerate(graph.operators):
            if model.operatorCodes[op.opcodeIndex].builtinCode != s.BuiltinOperator.CONV_2D:
                rewritten.append(op)
                continue
            params = op.builtinOptions
            if params.padding != s.Padding.SAME:
                rewritten.append(op)
                continue
            if (params.strideH != 1 or params.strideW != 1
                    or params.dilationHFactor != 1 or params.dilationWFactor != 1):
                raise ValueError('Only reviewed unit-stride/dilation SAME convolutions are supported')
            input_index, filter_index, bias_index = map(int, op.inputs)
            x, weight, bias = [graph.tensors[n] for n in (input_index, filter_index, bias_index)]
            output = graph.tensors[int(op.outputs[0])]
            if (x.type != s.TensorType.INT8 or weight.type != s.TensorType.INT8
                    or output.type != s.TensorType.INT8 or bias.type != s.TensorType.INT32
                    or len(x.shape) != 4 or len(weight.shape) != 4
                    or list(output.shape[:3]) != list(x.shape[:3])
                    or x.quantization.scale.size != 1 or x.quantization.zeroPoint.size != 1
                    or weight.quantization.quantizedDimension != 0
                    or np.any(weight.quantization.zeroPoint != 0)):
                raise ValueError('Unsupported fixed-shape symmetric-weight INT8 convolution')
            batch, height, width, channels = map(int, x.shape)
            out_channels, kh, kw, wc = map(int, weight.shape)
            if (batch != 1 or channels != wc or kh != 1 or kw != 3
                    or out_channels != int(output.shape[3])):
                raise ValueError('Expected reviewed horizontal 1x3 convolution')
            values = np.frombuffer(model.buffers[bias.buffer].data.tobytes(), dtype='<i4').astype(np.int64)
            if (len(values) != out_channels
                    or np.any(np.abs(values) + kh * kw * channels * 255 * 128 > 2**31 - 1)):
                raise ValueError('Exact int32 accumulator bound failed')
            paddings = np.array([[0, 0], [0, 0], [1, 1], [0, 0]], dtype='<i4')
            buffer = s.BufferT()
            buffer.data = np.frombuffer(paddings.tobytes(), dtype=np.uint8).copy()
            pad_tensor = s.TensorT()
            pad_tensor.shape = np.array([4, 2], np.int32)
            pad_tensor.type = s.TensorType.INT32
            pad_tensor.buffer = len(model.buffers)
            pad_tensor.name = f'explicit_padding_{graph_index}_{op_index}'.encode()
            model.buffers.append(buffer)
            pad_index = len(graph.tensors)
            graph.tensors.append(pad_tensor)
            padded = copy.deepcopy(x)
            padded.shape = np.array([batch, height, width + 2, channels], np.int32)
            padded.shapeSignature = None
            padded.buffer = 0
            padded.name = f'explicit_padded_input_{graph_index}_{op_index}'.encode()
            padded_index = len(graph.tensors)
            graph.tensors.append(padded)
            pad = s.OperatorT()
            pad.opcodeIndex = pad_code
            pad.inputs = np.array([input_index, pad_index], np.int32)
            pad.outputs = np.array([padded_index], np.int32)
            rewritten.append(pad)
            op.inputs = op.inputs.copy()
            op.inputs[0] = padded_index
            params.padding = s.Padding.VALID
            rewritten.append(op)
            changes.append({'subgraph': graph_index, 'operator': op_index,
                            'input_shape': list(map(int, x.shape)),
                            'padded_shape': list(map(int, padded.shape)),
                            'padding_byte': int(x.quantization.zeroPoint[0])})
        graph.operators = rewritten
    if len(changes) != 9:
        raise ValueError('Unexpected implicit-padding convolution count')
    result = pack(model)
    after = s.ModelT.InitFromObj(s.Model.GetRootAsModel(result, 0))
    if before_buffers != [None if b.data is None else b.data.tobytes()
                          for b in after.buffers[:len(before_buffers)]]:
        raise ValueError('An existing constant buffer changed')
    return result, changes


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('model', 'inputs', 'expected'):
        p.add_argument('--' + name, type=Path, required=True)
        p.add_argument('--' + name + '-sha256', required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    for name in ('model', 'inputs', 'expected'):
        if hashlib.sha256(getattr(args, name).read_bytes()).hexdigest() != getattr(args, name + '_sha256'):
            raise ValueError('Frozen fixture identity differs')
    source = args.model.read_bytes()
    result, changes = transform(source)
    inputs = np.frombuffer(args.inputs.read_bytes(), dtype=np.int8).reshape(-1, 1, 8, 32, 1)
    expected = np.frombuffer(args.expected.read_bytes(), dtype=np.int8).reshape(-1, 96)
    if len(inputs) != len(expected) or len(inputs) < 100:
        raise ValueError('Expected matched warmed streaming fixtures')
    runners = [Interpreter(model_content=data, experimental_op_resolver_type=OpResolverType.BUILTIN_REF)
               for data in (source, result)]
    for runner in runners:
        runner.allocate_tensors()
    mismatches = [0, 0]
    for x, reference in zip(inputs, expected):
        for n, runner in enumerate(runners):
            runner.set_tensor(runner.get_input_details()[0]['index'], x)
            runner.invoke()
            actual = runner.get_tensor(runner.get_output_details()[0]['index']).reshape(-1)
            mismatches[n] += int(np.count_nonzero(actual != reference))
    if any(mismatches):
        raise ValueError(f'Streaming backbone differs from frozen outputs: {mismatches}')
    folded, tables = fold(result)
    if len(tables) != 19:
        raise ValueError('Unexpected activation count')
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'backbone-unfolded.tflite').write_bytes(result)
    (args.output / 'backbone.tflite').write_bytes(folded)
    report = {'scope': 'Equivalent graph dispatch optimization; no recognition improvement claim',
              'source_sha256': args.model_sha256,
              'unfolded_sha256': hashlib.sha256(result).hexdigest(),
              'output_sha256': hashlib.sha256(folded).hexdigest(),
              'inputs_sha256': args.inputs_sha256, 'expected_sha256': args.expected_sha256,
              'cases': len(inputs), 'output_values': int(expected.size),
              'source_mismatches': mismatches[0], 'rewritten_mismatches': mismatches[1],
              'existing_constant_buffers_unchanged': True, 'accumulator_bound_safe': True,
              'sample_rate_or_cadence_changed': False, 'changes': changes,
              'activation_tables': tables, 'p4_verified': False, 'accepted': False}
    (args.output / 'padding-report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k not in ('changes', 'activation_tables')}))


if __name__ == '__main__':
    main()
