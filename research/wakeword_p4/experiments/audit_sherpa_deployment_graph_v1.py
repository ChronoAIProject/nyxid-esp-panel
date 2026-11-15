"""Static original graph inventory only: no inference, conversion, tuning or audio."""
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import onnx

BASE = Path('.local/wakeword/sherpa-kws-reference-v1/sherpa-onnx-kws-zipformer-gigaspeech-3.3M-2024-01-01')
OUT = Path('research/wakeword_p4/results/4090-sherpa-deployment-graph-audit-v1.json')

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def tensor(v):
    dims = [d.dim_value if d.HasField('dim_value') else d.dim_param for d in v.type.tensor_type.shape.dim]
    # Only the explicit upstream batch symbol is resolved. All other unknowns retained.
    concrete = [1 if d == 'N' else d for d in dims]
    size = onnx.helper.tensor_dtype_to_np_dtype(v.type.tensor_type.elem_type).itemsize
    return dict(name=v.name, dtype=onnx.TensorProto.DataType.Name(v.type.tensor_type.elem_type), dimensions=dims,
                batch_one_bytes=math.prod(concrete)*size if all(isinstance(d,int) and d>0 for d in concrete) else None)

def main():
    graphs = []
    for p in sorted(BASE.glob('*.onnx')):
        m = onnx.load(p, load_external_data=False)
        assert not any(t.data_location == onnx.TensorProto.EXTERNAL for t in m.graph.initializer)
        inputs = [tensor(v) for v in m.graph.input]
        graphs.append(dict(path=str(p), bytes=p.stat().st_size, sha256=sha(p),
            opsets=[dict(domain=x.domain,version=x.version) for x in m.opset_import],
            operators=dict(sorted(Counter(n.op_type for n in m.graph.node).items())),
            inputs=inputs, outputs=[tensor(v) for v in m.graph.output],
            initializers=dict(count=len(m.graph.initializer),logical_bytes=sum(math.prod(t.dims)*onnx.helper.tensor_dtype_to_np_dtype(t.data_type).itemsize for t in m.graph.initializer)),
            metadata={x.key:x.value for x in m.metadata_props}))
    encoder = next(g for g in graphs if '/encoder-' in g['path'] and '.int8.' in g['path'])
    states = [v for v in encoder['inputs'] if v['name'] != 'x']
    assert all(v['batch_one_bytes'] is not None for v in states)
    mixed = sum(g['bytes'] for g in graphs if ('/encoder-' in g['path'] or '/joiner-' in g['path']) and '.int8.' in g['path']) + next(g['bytes'] for g in graphs if '/decoder-' in g['path'] and '.int8.' not in g['path'])
    quantized_triplet = sum(g['bytes'] for g in graphs if '.int8.' in g['path'])
    report = dict(schema='sherpa-deployment-graph-audit-v1',script_sha256=sha(Path(__file__)),onnx_version=onnx.__version__,graphs=graphs,
        original_reference_model_bytes=mixed,available_quantized_triplet_bytes=quantized_triplet,
        encoder_batch_one_explicit_state_bytes=sum(v['batch_one_bytes'] for v in states),
        encoder_state_tensor_count=len(states),encoder_feature_input_bytes=encoder['inputs'][0]['batch_one_bytes'],
        scope='Static original protobuf inventory only. No candidate creation, inference, audio, acceptance access, conversion or numerical comparison.',
        conclusions=[
            'The available INT8 decoder is a distinct original archive member, not the decoder used in the consumed native reference. No behavioral result transfers to it.',
            'INT8 filenames describe weight/dynamic matmul quantization, not all-integer execution: encoder external features/caches/output are FLOAT; Conv, MatMul, Softmax, Exp, Log and other float paths remain.',
            'Explicit input state bytes are a batch-one lower bound on live memory, excluding outgoing states, all intermediate activations, graph/runtime/search/frontend memory and model weights. Do not double-count aliased buffers or call it measured peak RAM.',
            'decode_chunk_len=32 and T=45 are feature-frame geometry, not an isolated measured acoustic latency. Existing host-return latency includes frontend/decoder effects.',
            'No P4 throughput/MAC count derived: dynamic graph and search need a separately specified implementation/profile. Static operator counts are not runtime cost.',
            'Direct all-integer TFLite Micro deployment is unproven. Requires explicit float-path/state/attention replacement or a supported mixed runtime, a conversion audit and whole-system measurement. No automatic admission based on archive size.',
            'Compact lexical-sequence research must preserve full nearword supervision, specify causal state and token timing/decoder events, and prove feasible operators before a new training budget. Prior small-encoder CTC failures remain relevant; copying a token head is insufficient.'
        ],candidate_promoted=False)
    with OUT.open('x') as f:
        f.write(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ['original_reference_model_bytes','available_quantized_triplet_bytes','encoder_batch_one_explicit_state_bytes','encoder_state_tensor_count','encoder_feature_input_bytes']}))

if __name__ == '__main__':
    main()
