"""Read a frozen explicit-cache INT8 export and its operation versions for Mac/P4."""
import argparse
from collections import Counter
import gzip
import hashlib
from importlib import metadata
import json
import math
from pathlib import Path

from ai_edge_litert import schema_py_generated as schema
from ai_edge_litert.interpreter import Interpreter, OpResolverType
import numpy as np


def sha(data):
    return hashlib.sha256(data).hexdigest()


def inspect(report_path, packed_path):
    report_bytes = report_path.read_bytes()
    report = json.loads(report_bytes)
    compressed = packed_path.read_bytes()
    model = gzip.decompress(compressed)
    identity = report['model']
    if len(model) != identity['bytes'] or sha(model) != identity['sha256']:
        raise ValueError('Compressed model does not match original export report')

    graph = schema.Model.GetRootAsModel(model, 0)
    names = {value: key for key, value in vars(schema.BuiltinOperator).items() if isinstance(value, int)}
    ops = Counter()
    for i in range(graph.SubgraphsLength()):
        subgraph = graph.Subgraphs(i)
        for j in range(subgraph.OperatorsLength()):
            op = graph.OperatorCodes(subgraph.Operators(j).OpcodeIndex())
            ops[(names[op.BuiltinCode()], int(op.Version()))] += 1
    totals = Counter()
    for (name, _), count in ops.items():
        totals[name] += count
    if totals != Counter(report['ops']):
        raise ValueError('Effective FlatBuffer operation inventory differs from export report')

    interpreter = Interpreter(model_content=model, num_threads=1,
                              experimental_op_resolver_type=OpResolverType.BUILTIN_REF)
    interpreter.allocate_tensors()
    runner = interpreter.get_signature_runner()
    inputs, outputs = runner.get_input_details(), runner.get_output_details()
    def layout(tensor):
        shape = [int(n) for n in tensor['shape']]
        dtype = np.dtype(tensor['dtype'])
        scale, zero = tensor['quantization']
        return dict(shape=shape, dtype=dtype.name, bytes=math.prod(shape)*dtype.itemsize,
                    scale=float(scale), zero=int(zero), index=int(tensor['index']))

    caches = {}
    for key in sorted(inputs.keys() - {'chunk'}):
        entry, feedback = layout(inputs[key]), layout(outputs[key+'_out'])
        if entry['shape'] != feedback['shape'] or entry['dtype'] != feedback['dtype'] or entry['dtype'] != 'int8':
            raise ValueError(f'Unsupported cache shape or type at {key}')
        caches[key] = dict(input=entry, output=feedback,
                           copy_feedback=(entry['scale'], entry['zero']) == (feedback['scale'], feedback['zero']))
    if set(outputs) != {'probability', *(k+'_out' for k in caches)} or len(caches) != 7:
        raise ValueError('Unexpected explicit-cache signature')
    required = [not caches[f'cache{i}']['copy_feedback'] for i in range(len(caches))]
    if required != report['cache_requantization_required']:
        raise ValueError('Export and allocated host signature disagree on cache quantization')
    return dict(model=dict(bytes=len(model),sha256=sha(model)), packed_sha256=sha(compressed),
                export_report_sha256=sha(report_bytes), inspector_sha256=sha(Path(__file__).read_bytes()),
                runtime=metadata.version('ai-edge-litert'), graph_version=graph.Version(),
                operations=[dict(name=n,version=v,count=c) for (n,v),c in sorted(ops.items())],
                subgraphs=graph.SubgraphsLength(), input_chunk=layout(inputs['chunk']),
                probability=layout(outputs['probability']), caches=caches,
                explicit_cache_input_bytes=sum(c['input']['bytes'] for c in caches.values()),
                explicit_cache_output_bytes=sum(c['output']['bytes'] for c in caches.values()),
                scope='Exact frozen bytes, FlatBuffer versions and single-thread host BUILTIN_REF allocation. '
                      'P4 resolver registration, kernels, peak arena and timing require device evidence.')


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('original_export_report', type=Path)
    cli.add_argument('compressed_model', type=Path)
    options = cli.parse_args()
    print(json.dumps(inspect(options.original_export_report, options.compressed_model), indent=2))
