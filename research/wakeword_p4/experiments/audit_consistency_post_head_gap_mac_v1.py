"""Independently audit serialized fragments and saved post-head observations.

No interpreter/model invocation, TensorFlow, optimizer, or gate selection.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from ai_edge_litert import schema_py_generated as schema


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalized(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, bytes):
        return value.hex()
    if isinstance(value, (list, tuple)):
        return [normalized(item) for item in value]
    if hasattr(value, '__dict__'):
        return {key: normalized(item) for key, item in vars(value).items()}
    return value


def measures(error):
    count = len(error)-4
    window = np.abs(np.sum(np.stack([error[offset:offset+count] for offset in range(5)]), axis=0)/5)
    return dict(mean=float(np.mean(window)), p99=float(np.quantile(window, .99)),
                maximum=float(np.max(window)))


def run(args):
    base = Path(__file__).resolve().parents[1]
    plan_path = base / 'results/mac-consistency-post-head-gap-plan-v1.json'
    plan = json.loads(plan_path.read_text())
    assert sha(base/'experiments/diagnose_consistency_post_head_gap_mac_v1.py') == plan['script_sha256']
    report_path = args.root / 'report.json'
    report = json.loads(report_path.read_text())
    assert report['plan_sha256'] == sha(plan_path)
    assert len(report['cases']) == len(plan['cases']) == 4
    tables, cases = [], []
    for case in plan['cases']:
        row, = [entry for entry in report['cases'] if
                (entry['target'], entry['condition']) == (case['target'], case['condition'])]
        model_path = args.payload_root / case['model_path']
        fragment_path = args.root / row['fragment']
        arrays_path = args.root / row['arrays']
        native_path = args.native_root / case['arrays']
        assert sha(model_path) == case['model_sha256'] == row['model_sha256']
        assert sha(fragment_path) == row['fragment_sha256']
        assert sha(arrays_path) == row['arrays_sha256']
        assert sha(native_path) == case['arrays_sha256'] == row['original_arrays_sha256']
        original = schema.ModelT.InitFromObj(schema.Model.GetRootAsModel(model_path.read_bytes(), 0))
        fragment = schema.ModelT.InitFromObj(schema.Model.GetRootAsModel(fragment_path.read_bytes(), 0))
        assert normalized(original.buffers) == normalized(fragment.buffers)
        assert normalized(original.operatorCodes) == normalized(fragment.operatorCodes)
        expected = normalized(original.subgraphs[0])
        expected['operators'] = [expected['operators'][entry['ordinal']] for entry in case['selected']]
        expected['inputs'] = [case['fc_output']]
        assert len(fragment.subgraphs) == 1 and normalized(fragment.subgraphs[0]) == expected
        assert not fragment.signatureDefs
        with np.load(arrays_path, allow_pickle=False) as data:
            saved = {key: data[key].copy() for key in data.files}
        with np.load(native_path, allow_pickle=False) as data:
            native = {key: data[key].copy() for key in ['actual_fc', 'uint8', 'float']}
        assert len(saved) == 20 and all(np.isfinite(value).all() for value in saved.values())
        assert np.array_equal(saved['enum_fc_codes'].astype(np.int16), np.arange(-128, 128))
        for saved_key, original_key in [('witness_fc', 'actual_fc'), ('witness_uint8', 'uint8'),
                                         ('witness_float', 'float')]:
            assert np.array_equal(saved[saved_key], native[original_key])
        indices = native['actual_fc'].astype(np.int16)+128
        stages = [entry['outputs'][0] for entry in case['selected']]
        assert row['stage_ids'] == stages
        for stage in stages:
            values = saved['r0_stage'+str(stage)]
            assert values.shape == (256+case['steps'],)
            assert np.array_equal(values, saved['r1_stage'+str(stage)])
            assert np.array_equal(values[:256][indices], values[256:])
        uint8 = saved['r0_stage'+str(stages[-1])]
        assert np.array_equal(uint8[256:], native['uint8'])
        fcq = original.subgraphs[0].tensors[case['fc_output']].quantization
        oq = original.subgraphs[0].tensors[stages[-1]].quantization
        assert row['fc_quantization'] == [float(fcq.scale[0]), int(fcq.zeroPoint[0])]
        assert row['output_quantization'] == [float(oq.scale[0]), int(oq.zeroPoint[0])]
        dequantized = np.asarray((np.arange(-128, 128, dtype=np.float64)-int(fcq.zeroPoint[0]))*
                                float(fcq.scale[0]), np.float32)
        centered = np.subtract(dequantized, np.float32(3), dtype=np.float32)
        coordinate = np.minimum(np.float32(8), np.maximum(np.float32(-8),
                        np.multiply(centered, np.float32(8/3), dtype=np.float32)))
        nominal = 1/(1+np.exp(-coordinate.astype(np.float64)))
        actual = (uint8[:256].astype(np.float64)-int(oq.zeroPoint[0]))*float(oq.scale[0])
        assert np.array_equal(coordinate, saved['enum_nominal_logit'])
        assert np.array_equal(nominal, saved['enum_nominal_probability'])
        assert np.array_equal(actual, saved['enum_actual_probability'])
        assert np.array_equal(nominal[indices], saved['witness_nominal_probability'])
        bound = float(np.max(np.abs(actual-nominal)))
        assert bound == row['enum_max_abs_nominal_probability_gap']
        error = {
            'original_total': actual[indices]-native['float'],
            'post_head_component': actual[indices]-nominal[indices],
            'nominal_fc_vs_float_component': nominal[indices]-native['float']}
        for key, array_key in [('original_total', 'total_signed_error'),
                               ('post_head_component', 'post_head_signed_error'),
                               ('nominal_fc_vs_float_component', 'nominal_fc_vs_float_signed_error')]:
            assert np.array_equal(error[key], saved[array_key])
        metrics = {key: measures(value) for key, value in error.items()}
        for key, values in metrics.items():
            assert all(np.isclose(value, row['smoothed_abs_error'][key][name], rtol=1e-12, atol=1e-15)
                       for name, value in values.items())
        lower = max(0., metrics['original_total']['p99']-bound)
        assert np.isclose(lower, row['residual_five_step_p99_lower_bound_from_uniform_post_head_gap'],
                          rtol=1e-12, atol=1e-15)
        assert metrics['nominal_fc_vs_float_component']['p99']+1e-15 >= lower
        cases.append(dict(target=case['target'], condition=case['condition'],
            all_original_buffers_opcodes_tensors_and_selected_ops_exact=True,
            two_fresh_all_four_stage_repeats_exact=True,
            enumeration_reproduces_every_saved_witness_stage=True,
            witness_output_matches_original_wholemodel=True,
            enumeration_nominal_probability_max_gap=bound,
            five_step_metrics=metrics, residual_p99_lower_bound=lower,
            arrays_sha256=sha(arrays_path), all_checks_passed=True))
        tables.append(uint8[:256])
    table_match = all(np.array_equal(tables[0], table) for table in tables)
    result = dict(schema='mac-consistency-post-head-gap-independent-audit-v1',
        script_sha256=sha(Path(__file__)), plan_sha256=sha(plan_path),
        original_report_sha256=sha(report_path), cases=cases,
        all_four_post_head_uint8_tables_identical=table_match,
        all_checks_passed=True,
        limits='Actual serialized fragment buffers/opcode versions/all tensors/selected ops authenticated against original models. Saved two repeats, every stage and wholemodel witness outputs independently checked; all256 nominal/actual probabilities and signed rolling-window decomposition recomputed. No interpreter invoked by this audit. Fragment enumeration is not independent all-code TFLM/P4 execution, a trained solution, recognition/FAR, or all540 admission. Original failures remain.')
    with args.report.open('x') as stream:
        stream.write(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(report=str(args.report), all_checks_passed=True,
                         all_four_post_head_uint8_tables_identical=table_match)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--payload-root', type=Path, required=True)
    parser.add_argument('--native-root', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    run(parser.parse_args())
