"""Move a full-feature temporal convolution from spatial width to channels.

INT8 input/weight bytes, quantization, biases, output shape and accumulation
order are unchanged: [H,96,1] x [5,96,1] becomes [H,1,96] x [5,1,96].
This removes an unnecessary large ESP-NN workspace, without retraining.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path

import flatbuffers
import numpy as np
from tensorflow.lite.python import schema_py_generated as schema
from ai_edge_litert.interpreter import Interpreter, OpResolverType


def transform(data):
    model = schema.ModelT.InitFromObj(schema.Model.GetRootAsModel(data, 0))
    graph = model.subgraphs[0]
    first = graph.operators[0]
    if model.operatorCodes[first.opcodeIndex].builtinCode != schema.BuiltinOperator.CONV_2D:
        raise ValueError('Expected first operator to be the full-feature Conv2D')
    input_index, filter_index, bias_index = map(int, first.inputs)
    x, weight = graph.tensors[input_index], graph.tensors[filter_index]
    if list(graph.inputs) != [input_index] or list(x.shape) != [1, 16, 96, 1] or list(weight.shape) != [32, 5, 96, 1]:
        raise ValueError('Unreviewed temporal layout')
    params = first.builtinOptions
    if (params.padding != schema.Padding.VALID or params.strideH != 1 or params.strideW != 1
            or params.dilationHFactor != 1 or params.dilationWFactor != 1):
        raise ValueError('Only full-width valid unit-stride convolution is equivalent')
    if x.type != schema.TensorType.INT8 or weight.type != schema.TensorType.INT8 or x.quantization.scale.size != 1:
        raise ValueError('Expected per-tensor INT8 input and INT8 filter')
    if weight.quantization.quantizedDimension != 0 or np.any(weight.quantization.zeroPoint != 0):
        raise ValueError('Expected symmetric output-channel filter quantization')
    if sum(filter_index in op.inputs for op in graph.operators) != 1:
        raise ValueError('Filter must not have other consumers')
    output = graph.tensors[int(first.outputs[0])]
    if list(output.shape) != [1, 12, 1, 32]:
        raise ValueError('Unexpected first output shape')
    weights = model.buffers[weight.buffer].data
    if weights.size != 32 * 5 * 96:
        raise ValueError('Expected fixed dense filter bytes')
    bias = model.buffers[graph.tensors[bias_index].buffer].data
    bias_values = np.frombuffer(bias.tobytes(), dtype='<i4').astype(np.int64)
    if len(bias_values) != 32 or (np.abs(bias_values) + 5 * 96 * 255 * 128 > 2**31 - 1).any():
        raise ValueError('Accumulator bound does not establish exact int32 sums')
    new_input = copy.deepcopy(x)
    new_input.name = b'feature_channels_reshape'
    new_input.shape = np.array([1, 16, 1, 96], np.int32)
    new_input.shapeSignature = None
    new_index = len(graph.tensors)
    graph.tensors.append(new_input)
    code = schema.OperatorCodeT()
    code.builtinCode = schema.BuiltinOperator.RESHAPE
    code.deprecatedBuiltinCode = schema.BuiltinOperator.RESHAPE
    code.version = 1
    reshape = schema.OperatorT()
    reshape.opcodeIndex = len(model.operatorCodes)
    reshape.inputs = np.array([input_index], np.int32)
    reshape.outputs = np.array([new_index], np.int32)
    reshape.builtinOptionsType = schema.BuiltinOptions.ReshapeOptions
    reshape.builtinOptions = schema.ReshapeOptionsT()
    reshape.builtinOptions.newShape = np.array([1, 16, 1, 96], np.int32)
    model.operatorCodes.append(code)
    first.inputs = first.inputs.copy()
    first.inputs[0] = new_index
    weight.shape = np.array([32, 5, 1, 96], np.int32)
    weight.shapeSignature = None
    graph.operators.insert(0, reshape)
    builder = flatbuffers.Builder(1024)
    builder.Finish(model.Pack(builder), file_identifier=b'TFL3')
    result = bytes(builder.Output())
    after = schema.ModelT.InitFromObj(schema.Model.GetRootAsModel(result, 0))
    if any(a.data.tobytes() != b.data.tobytes() for a, b in zip(model.buffers, after.buffers) if a.data is not None):
        raise ValueError('A constant buffer changed')
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--expected-model-sha256', required=True)
    p.add_argument('--inputs', type=Path, required=True)
    p.add_argument('--expected-inputs-sha256', required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    source = args.model.read_bytes()
    if hashlib.sha256(source).hexdigest() != args.expected_model_sha256 or hashlib.sha256(args.inputs.read_bytes()).hexdigest() != args.expected_inputs_sha256:
        raise ValueError('Frozen source/input identity mismatch')
    result = transform(source)
    inputs = np.frombuffer(args.inputs.read_bytes(), dtype=np.int8).reshape(-1, 1, 16, 96, 1)
    interpreters = [Interpreter(model_content=data, experimental_op_resolver_type=OpResolverType.BUILTIN_REF) for data in (source, result)]
    for interpreter in interpreters:
        interpreter.allocate_tensors()
    mismatches = 0
    for x in inputs:
        outputs = []
        for interpreter in interpreters:
            interpreter.set_tensor(interpreter.get_input_details()[0]['index'], x)
            interpreter.invoke()
            outputs.append(interpreter.get_tensor(interpreter.get_output_details()[0]['index']))
        mismatches += int(np.count_nonzero(outputs[0] != outputs[1]))
    if mismatches:
        raise ValueError('Whole-head reference outputs differ after layout rewrite')
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'head.tflite').write_bytes(result)
    report = {'scope': 'Exact-layout resource optimization, no recognition improvement claim',
              'source_sha256': args.expected_model_sha256,
              'output_sha256': hashlib.sha256(result).hexdigest(),
              'inputs_sha256': args.expected_inputs_sha256, 'cases': len(inputs), 'mismatches': mismatches,
              'constant_buffer_bytes_unchanged': True, 'accumulator_bound_safe': True,
              'ordered_products_equivalent': True, 'p4_compatibility_verified': False}
    (args.output / 'layout-report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
