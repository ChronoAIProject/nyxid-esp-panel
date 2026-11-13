"""Read original tensor metadata and saved traces; never invoke or export."""
import collections
import importlib.metadata
import json
from pathlib import Path

import numpy as np
from ai_edge_litert.interpreter import Interpreter, OpResolverType
from plan_hey_broad_trained_quantization_v1 import sha

ROOT = Path('research/wakeword_p4/results')


def tensor(t):
    return dict(index=int(t['index']), name=t['name'], shape=t['shape'].tolist(),
                dtype=str(t['dtype']), scale=float(t['quantization'][0]),
                zero_point=int(t['quantization'][1]))


def main():
    lock_path = ROOT/'4090-hey-broad-trained-quantization-execution-v1.json'
    result_path = ROOT/'4090-hey-broad-trained-quantization-result-v1.json'
    lock = json.loads(lock_path.read_text()); result = json.loads(result_path.read_text())
    assert sha(lock_path) == result['execution_sha256']
    reports = {}
    for target, folder in lock['models'].items():
        model = Path(folder)/'hey_nyx.tflite'
        assert sha(model) == lock['files'][str(model)]
        interpreter = Interpreter(model_path=str(model), experimental_op_resolver_type=OpResolverType.BUILTIN_REF)
        interpreter.allocate_tensors()  # metadata only, no invoke
        tensors = {t['index']: t for t in interpreter.get_tensor_details()}
        ops = interpreter._get_ops_details()
        logistic, = [op for op in ops if op['op_name'] == 'LOGISTIC']
        logit = tensors[int(logistic['inputs'][0])]
        probability = tensors[int(logistic['outputs'][0])]
        read = {int(op['inputs'][0]): tensors[int(op['outputs'][0])]
                for op in ops if op['op_name'] == 'READ_VARIABLE'}
        write = {int(op['inputs'][0]): tensors[int(op['inputs'][1])]
                 for op in ops if op['op_name'] == 'ASSIGN_VARIABLE'}
        assert read.keys() == write.keys() and len(read) == 6
        states = []
        for handle in sorted(read):
            a, b = read[handle], write[handle]
            matches = bool(np.array_equal(a['shape'], b['shape']) and
                           a['quantization'] == b['quantization'] and a['dtype'] == b['dtype'])
            assert matches
            states.append(dict(handle=handle, read=tensor(a), assigned=tensor(b), metadata_matches=matches))
        scale, zero = logit['quantization']
        logit_values = (np.arange(-128, 128, dtype=np.float64)-zero)*scale
        sigmoid = np.empty_like(logit_values)
        positive = logit_values >= 0
        sigmoid[positive] = 1/(1+np.exp(-logit_values[positive]))
        exp = np.exp(logit_values[~positive]); sigmoid[~positive] = exp/(1+exp)
        # Mathematical diagnostic, not a claim to independently implement the
        # integer LOGISTIC kernel's rounding. Actual codes are counted below.
        output_scale, output_zero = interpreter.get_output_details()[0]['quantization']
        ideal_codes = np.clip(np.rint(sigmoid/output_scale)+output_zero, 0, 255).astype(np.uint8)
        counts = collections.Counter(); nearest = []; floats = 0; cases = 0
        for case in result['cases']:
            if case['target'] != target:
                continue
            path = Path(case['arrays_path']); assert sha(path) == case['arrays_sha256']
            with np.load(path, allow_pickle=False) as saved:
                counts.update(map(int, saved['uint8'])); values = saved['float']
                candidates = (np.unique(ideal_codes).astype(np.float64)-output_zero)*output_scale
                nearest.extend(np.min(np.abs(values[:, None]-candidates[None]), axis=1).tolist())
                floats += len(values); cases += 1
        reports[target] = dict(model_path=str(model), model_sha256=sha(model),
            model_bytes=model.stat().st_size, logit=tensor(logit), logistic_probability=tensor(probability),
            logit_representable_range=[float(logit_values.min()), float(logit_values.max())],
            ideal_sigmoid_output_codes=sorted(set(map(int, ideal_codes))),
            actual_saved_output_code_counts={str(k): v for k, v in sorted(counts.items())},
            all_saved_codes_in_ideal_set=set(counts) <= set(map(int, ideal_codes)),
            source_cases=cases, raw_outputs=floats,
            nearest_ideal_raw_probability_error=dict(mean=float(np.mean(nearest)),
                p99=float(np.percentile(nearest, 99)), maximum=float(np.max(nearest))),
            states=states)
    report = dict(schema='hey-broad-quantization-lattice-audit-v1', script_sha256=sha(__file__),
        original_execution_sha256=sha(lock_path), original_result_sha256=sha(result_path),
        litert_version=importlib.metadata.version('ai-edge-litert'), models=reports,
        model_invocations=0, model_exports=0,
        initial_exploratory_api_note='get_tensor_details lacks is_variable in installed LiteRT; initial metadata query raised KeyError before inference. Resource relationships here come from READ/ASSIGN ops, not that unavailable key.',
        conclusion='Final pre-sigmoid affine INT8 logit quantization is extremely coarse; actual saved outputs occupy only the corresponding small probability lattice. State read/write shapes and affine quantization agree. This is a concrete output-resolution bottleneck, not yet proof of why calibration produced that range or of all upstream errors.',
        limits='Tensor metadata plus saved training outputs only. Ideal sigmoid diagnostic is not an independent integer kernel oracle. No internal dynamic saturation measured; no PTQ retry, model mutation, newdev/acceptance or gate relaxation.')
    with (ROOT/'4090-hey-broad-quantization-lattice-audit-v1.json').open('x') as output:
        output.write(json.dumps(report, indent=2)+'\n')
    print(json.dumps({t: dict(logit_scale=m['logit']['scale'], codes=m['actual_saved_output_code_counts'],
                             state_metadata_matches=True) for t, m in reports.items()}))


if __name__ == '__main__':
    main()
