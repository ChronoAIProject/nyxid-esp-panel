"""Exactly fold adjacent fixed INT8 LeakyRelu/Maximum into custom byte LUTs.

Each table exhaustively evaluates the original reference kernels for all 256
input byte values. A device implementation of NYX_INT8_LUT must interpret the
256 raw option bytes as signed outputs indexed by signed_input + 128. This
does not establish whole-model/device equality: run those numeric probes too.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import zlib

import flatbuffers
import numpy as np
from tensorflow.lite.python import schema_py_generated as schema
from ai_edge_litert.interpreter import Interpreter, OpResolverType


def pack(model):
    builder = flatbuffers.Builder(1024)
    builder.Finish(model.Pack(builder), file_identifier=b'TFL3')
    return bytes(builder.Output())


def fold(data):
    model = schema.ModelT.InitFromObj(schema.Model.GetRootAsModel(data, 0))
    graph = model.subgraphs[0]

    def code(op):
        return model.operatorCodes[op.opcodeIndex].builtinCode

    custom = schema.OperatorCodeT()
    custom.builtinCode = schema.BuiltinOperator.CUSTOM
    custom.deprecatedBuiltinCode = schema.BuiltinOperator.CUSTOM
    custom.customCode = b'NYX_INT8_LUT'
    custom.version = 1
    custom_index = len(model.operatorCodes)
    replacements, tables = [], []
    number = 0
    while number < len(graph.operators):
        first = graph.operators[number]
        if code(first) != schema.BuiltinOperator.LEAKY_RELU:
            replacements.append(first)
            number += 1
            continue
        if number+1 >= len(graph.operators):
            raise ValueError('Unpaired LeakyRelu operator')
        second = graph.operators[number+1]
        if code(second) != schema.BuiltinOperator.MAXIMUM or second.inputs[0] != first.outputs[0]:
            raise ValueError('Unreviewed activation topology')
        input_index, intermediate, output_index = map(int, [first.inputs[0], first.outputs[0], second.outputs[0]])
        if sum(intermediate in op.inputs for op in graph.operators) != 1 or intermediate in graph.outputs:
            raise ValueError('Intermediate activation has another consumer')
        if not all(graph.tensors[n].type == schema.TensorType.INT8 for n in [input_index, intermediate, output_index]):
            raise ValueError('Only fixed INT8 activations are supported')
        constant = graph.tensors[int(second.inputs[1])]
        if int(np.prod(constant.shape)) != 1 or not model.buffers[constant.buffer].data.size:
            raise ValueError('Maximum must have a fixed scalar second input')
        mini = copy.deepcopy(model)
        g = mini.subgraphs[0]
        g.inputs, g.outputs = [input_index], [output_index]
        g.operators = copy.deepcopy([first, second])
        for index in [input_index, intermediate, output_index]:
            g.tensors[index].shape = [256]
            g.tensors[index].shapeSignature = None
        interpreter = Interpreter(model_content=pack(mini),
                                  experimental_op_resolver_type=OpResolverType.BUILTIN_REF)
        interpreter.allocate_tensors()
        inputs = np.arange(-128, 128, dtype=np.int16).astype(np.int8)
        interpreter.set_tensor(input_index, inputs)
        interpreter.invoke()
        values = interpreter.get_tensor(output_index)
        if values.shape != (256,) or values.dtype != np.int8:
            raise ValueError('Unexpected reference activation result')
        folded = schema.OperatorT()
        folded.opcodeIndex = custom_index
        folded.inputs, folded.outputs = first.inputs.copy(), second.outputs.copy()
        folded.customOptions = np.frombuffer(values.tobytes(), dtype=np.uint8).copy()
        folded.customOptionsFormat = schema.CustomOptionsFormat.FLEXBUFFERS
        replacements.append(folded)
        tables.append({'source_operator': number, 'input_tensor': input_index, 'output_tensor': output_index,
                       'cases': 256, 'table_sha256': hashlib.sha256(values.tobytes()).hexdigest()})
        number += 2
    if not tables:
        raise ValueError('No supported activation pairs found')
    model.operatorCodes.append(custom)
    graph.operators = replacements
    return pack(model), tables


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--expect-pairs', type=int, required=True)
    args = parser.parse_args()
    if args.expect_pairs <= 0:
        parser.error('--expect-pairs must be positive')
    source = args.model.read_bytes()
    data, tables = fold(source)
    if len(tables) != args.expect_pairs:
        raise ValueError('Activation pair count differs from the frozen graph')
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/'model.tflite').write_bytes(data)
    report = {'scope': 'Exhaustive fixed INT8 activation maps; whole-model P4 numeric proof still required',
              'source_sha256': hashlib.sha256(source).hexdigest(),
              'output_sha256': hashlib.sha256(data).hexdigest(),
              'source_bytes': len(source), 'output_bytes': len(data),
              'output_zlib_bytes': len(zlib.compress(data, 9)),
              'pairs': tables, 'exhaustive_input_cases': len(tables)*256, 'accepted': False}
    (args.output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({key: value for key, value in report.items() if key != 'pairs'}))


if __name__ == '__main__':
    main()
