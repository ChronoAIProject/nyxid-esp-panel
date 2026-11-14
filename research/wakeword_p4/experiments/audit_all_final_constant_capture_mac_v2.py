"""Audit saved constant arrays against raw FlatBuffer bytes, without TensorFlow.

Does not independently repeat execution or prove the original training graph.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
from ai_edge_litert import schema_py_generated as schema


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(args):
    base = Path(__file__).resolve().parents[1]
    plan_path = base / 'results/mac-all-final-constant-capture-plan-v2.json'
    plan = json.loads(plan_path.read_text())
    source = base / 'experiments/diagnose_all_final_constant_capture_mac_v2.py'
    assert sha(source) == plan['script_sha256']
    template_path = 'research/wakeword_p4/results/4090-hey-broad-consistency-fixed-grid-compile-plan-v1.json'
    template_bytes = subprocess.check_output(['git', 'show', plan['source_gpu_ref'] + ':' + template_path])
    assert hashlib.sha256(template_bytes).hexdigest() == plan['source_sha256'][template_path]
    templates = json.loads(template_bytes)['arms']
    reports, datasets = {}, {}
    for setting, directory in [('default', args.default),
                               ('no_constant_folding_or_arithmetic', args.no_fold)]:
        report_path = directory / 'report.json'
        report = json.loads(report_path.read_text())
        assert report['plan_sha256'] == sha(plan_path) and report['setting'] == setting
        assert report['model_invocations'] == report['optimizer_updates'] == 0
        assert report['physical_gpus_visible'] == []
        assert len(report['cases']) == 4
        reports[setting] = dict(report_sha256=sha(report_path), cases=[])
        for arm in plan['arms']:
            target, condition = arm['target'], arm['condition']
            row, = [case for case in report['cases'] if
                    (case['target'], case['condition']) == (target, condition)]
            assert row['model_sha256'] == arm['model_sha256']
            assert row['final_h5_sha256'] == arm['weights_sha256']
            model_path = args.payload_root / arm['model_path']
            assert sha(model_path) == arm['model_sha256']
            assert sha(args.payload_root / arm['weights_path']) == arm['weights_sha256']
            parsed = schema.ModelT.InitFromObj(schema.Model.GetRootAsModel(model_path.read_bytes(), 0))
            template, = [entry['template'] for entry in templates if
                         (entry['target'], entry['condition']) == (target, condition)]
            ops = [op for op in template['ops'] if op['op'] in
                   ('CONV_2D', 'DEPTHWISE_CONV_2D', 'FULLY_CONNECTED')]
            assert len(ops) == 10
            arrays_path = directory / row['arrays']
            assert sha(arrays_path) == row['arrays_sha256']
            with np.load(arrays_path, allow_pickle=False) as saved:
                arrays = {key: saved[key].copy() for key in saved.files}
            assert len(arrays) == 360 and all(np.isfinite(value).all() for value in arrays.values())
            observations = []
            for method in plan['methods']:
                for op in ops:
                    for ordinal, index in enumerate(op['inputs'][1:]):
                        tensor = parsed.subgraphs[0].tensors[index]
                        assert tensor.type == (schema.TensorType.INT32 if ordinal else schema.TensorType.INT8)
                        raw = bytes(parsed.buffers[tensor.buffer].data)
                        actual = np.frombuffer(raw, dtype='<i4' if ordinal else 'i1').reshape(tensor.shape)
                        prefix = method + '_tensor' + str(index) + '_'
                        value, dequant, quotient, codes, compiled, saved_scale = [
                            arrays[prefix + label] for label in
                            ['value', 'dequant', 'quotient', 'codes', 'compiled', 'scale']]
                        scale = np.asarray(tensor.quantization.scale, np.float32)
                        zero = np.asarray(tensor.quantization.zeroPoint, np.float64)
                        if ordinal:
                            shape = [len(scale)]
                        else:
                            shape = [1] * value.ndim
                            shape[-2 if op['op'] == 'DEPTHWISE_CONV_2D' else -1] = len(scale)
                        scale, zero = scale.reshape(shape), zero.reshape(shape)
                        assert np.array_equal(saved_scale, scale)
                        reconstructed = np.rint(dequant.astype(np.float64)/scale.astype(np.float64) + zero).astype(np.int64)
                        low, high = (-(2**31), 2**31-1) if ordinal else (-127, 127)
                        from_quotient = np.clip(np.rint(quotient.astype(np.float64))+zero, low, high).astype(np.int64)
                        if not ordinal:
                            if op['op'] == 'FULLY_CONNECTED':
                                reconstructed, from_quotient = reconstructed.T, from_quotient.T
                            else:
                                reconstructed = np.transpose(reconstructed, (3, 0, 1, 2))
                                from_quotient = np.transpose(from_quotient, (3, 0, 1, 2))
                        assert np.array_equal(codes, reconstructed)
                        assert compiled.dtype == actual.dtype and np.array_equal(compiled, actual)
                        differing = np.argwhere(codes != actual)
                        differences = [dict(index=coordinate.tolist(), observed=int(codes[tuple(coordinate)]),
                                            compiled=int(actual[tuple(coordinate)])) for coordinate in differing]
                        original, = [observation for observation in row['observations'] if
                                     (observation['method'], observation['index']) == (method, index)]
                        assert original['values'] == codes.size and original['bias'] == bool(ordinal)
                        assert original['unequal'] == len(differing) and original['differences'] == differences
                        observations.append(dict(method=method, index=index,
                            compiled_unequal=len(differing),
                            quotient_code_vs_dequant_unequal=int(np.count_nonzero(from_quotient != codes))))
            assert len(observations) == len(row['observations']) == 60
            datasets[(setting, target, condition)] = arrays
            reports[setting]['cases'].append(dict(target=target, condition=condition,
                arrays_sha256=sha(arrays_path), observations=observations))
    cross_settings = []
    for arm in plan['arms']:
        target, condition = arm['target'], arm['condition']
        left = datasets[('default', target, condition)]
        right = datasets[('no_constant_folding_or_arithmetic', target, condition)]
        assert set(left) == set(right)
        changed = {key: int(np.count_nonzero(left[key] != right[key])) for key in left
                   if not np.array_equal(left[key], right[key])}
        cross_settings.append(dict(target=target, condition=condition, changed_array_elements=changed))
    result = dict(schema='mac-all-final-constant-capture-saved-array-audit-v2',
        script_sha256=sha(Path(__file__)), plan_sha256=sha(plan_path), settings=reports,
        cross_settings=cross_settings, constant_comparisons=480, saved_array_integrity_passed=True,
        limits='Independent saved-array arithmetic and raw FlatBuffer constant authentication only. No TensorFlow execution, model invoke, source-parameter regeneration, optimizer, or recognition. Original runner alone asserts repeat/variable checks because second repeats and variable snapshots are not separately saved; this audit does not claim to prove them independently. Isolated graphs do not establish original full-network or GPU parity.')
    with args.report.open('x') as stream:
        stream.write(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(report=str(args.report), constant_comparisons=480,
        saved_array_integrity_passed=True,
        compiled_unequal=sum(observation['compiled_unequal'] for setting in reports.values()
                             for case in setting['cases'] for observation in case['observations']),
        cross_settings=cross_settings)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--default', type=Path, required=True)
    parser.add_argument('--no-fold', type=Path, required=True)
    parser.add_argument('--payload-root', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    run(parser.parse_args())
