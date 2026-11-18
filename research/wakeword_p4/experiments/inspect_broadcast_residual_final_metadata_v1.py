"""Read trained export metadata; no inference, training, or P4 arena claim."""
import hashlib
import json
from collections import Counter
from importlib import metadata
from pathlib import Path

import numpy as np
from ai_edge_litert.interpreter import Interpreter, OpResolverType


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    root = Path('research/wakeword_p4/results')
    source = root / '4090-broadcast-residual-joint-export-original-v1.json'
    export = json.loads(source.read_text())
    model = Path(export['model']['path'])
    assert model.stat().st_size == export['model']['bytes'] == 43000
    assert sha(model) == export['model']['sha256']
    it = Interpreter(model_path=str(model), num_threads=1,
                     experimental_op_resolver_type=OpResolverType.BUILTIN_REF)
    it.allocate_tensors()
    runner = it.get_signature_runner()
    inputs = runner.get_input_details()
    tensors = it.get_tensor_details()
    size = lambda d: int(np.prod(d['shape'])) * np.dtype(d['dtype']).itemsize
    ops = Counter(d['op_name'] for d in it._get_ops_details())
    assert ops == Counter(export['ops'])
    out = dict(
        model=export['model'], export_report_sha256=sha(source),
        inspector_sha256=sha(__file__),
        versions={n: metadata.version(n) for n in
                  ['torch', 'tensorflow', 'ai-edge-litert', 'pymicro-features', 'numpy']},
        ops=dict(sorted(ops.items())), explicit_cache_input_bytes=sum(
            size(d) for n, d in inputs.items() if n.startswith('cache')),
        largest_tensor_bytes=max(map(size, tensors)),
        sum_tensor_bytes=sum(map(size, tensors)), tensor_count=len(tensors),
        inference_calls=0, optimizer_updates=0,
        limits='Read-only trained host graph inspection. Tensor sum includes constants and '
               'overlapping lifetimes; excludes allocator/kernel scratch and firmware. '
               'Neither sum nor largest tensor proves P4 peak arena or kernel compatibility. '
               'Versions describe current inspection environment; original export source '
               'and report remain authoritative. No firmware or recognition qualification.')
    path = root / '4090-broadcast-residual-final-host-metadata-v1.json'
    with path.open('x') as f:
        json.dump(out, f, indent=2)
        f.write('\n')
    print(json.dumps(out))


if __name__ == '__main__':
    main()
