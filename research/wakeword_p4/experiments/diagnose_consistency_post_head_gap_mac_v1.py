"""Enumerate every int8 FC code through the original quantized post-head.

Validate a derived stateless LiteRT fragment against all four predetermined
native witnesses before describing nominal-head versus deployed-head error.
No full model/encoder invocation, training, new examples, or board/audio IO.
"""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess


BASE = Path(__file__).resolve().parents[1]
REPO = BASE.parents[1]
PREFIX = 'research/wakeword_p4/'
PLAN = BASE / 'results/mac-consistency-post-head-gap-plan-v1.json'
PARENT = BASE / 'results/mac-consistency-fixed-grid-witness-replay-plan-v1.json'
NATIVE = BASE / 'results/mac-consistency-fixed-grid-native-four-witness-replay-v1.json'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tracked(ref, path):
    return subprocess.check_output(['git', 'show', ref + ':' + path])


def make_plan(args):
    parent, native = [json.loads(path.read_text()) for path in [PARENT, NATIVE]]
    for path in [PARENT, NATIVE]:
        assert tracked('origin/research/hey-nyx-4090', str(path.relative_to(REPO))) == path.read_bytes()
    assert native['plan_sha256'] == sha(PARENT) and native['all_operations_passed']
    assert sha(args.native_root / 'report.json') == sha(NATIVE)
    templates_path = PREFIX + 'results/4090-hey-broad-consistency-fixed-grid-compile-plan-v1.json'
    template_bytes = tracked(parent['gpu_ref'], templates_path)
    assert hashlib.sha256(template_bytes).hexdigest() == parent['source_sha256'][templates_path]
    templates = json.loads(template_bytes)['arms']
    cases = []
    for arm in parent['arms']:
        row, = [row for row in native['cases'] if
                (row['target'], row['condition']) == (arm['target'], arm['condition'])]
        assert sha(args.native_root / row['arrays']) == row['arrays_sha256']
        assert sha(args.payload_root / arm['model_path']) == arm['model_sha256']
        template, = [entry['template'] for entry in templates if
                     (entry['target'], entry['condition']) == (arm['target'], arm['condition'])]
        selected = [n for n, op in enumerate(template['ops']) if
                    op['op'] in ['MUL', 'ADD', 'LOGISTIC', 'QUANTIZE']]
        assert [template['ops'][n]['op'] for n in selected] == ['MUL', 'ADD', 'LOGISTIC', 'QUANTIZE']
        fc, = [op for op in template['ops'] if op['op'] == 'FULLY_CONNECTED']
        cases.append(dict(target=arm['target'], condition=arm['condition'],
                          model_path=arm['model_path'], model_sha256=arm['model_sha256'],
                          arrays=row['arrays'], arrays_sha256=row['arrays_sha256'],
                          steps=row['steps'], fc_output=int(fc['outputs'][0]),
                          original_ops_count=len(template['ops']),
                          selected=[dict(ordinal=n, **template['ops'][n]) for n in selected]))
    result = dict(schema='mac-consistency-post-head-gap-plan-v1',
        script_sha256=sha(Path(__file__)), parent_plan_sha256=sha(PARENT),
        native_report_sha256=sha(NATIVE), cases=cases,
        versions={name: importlib.metadata.version(name) for name in
                  ['numpy', 'ai-edge-litert', 'flatbuffers']},
        policy='All4 original final10000 compiled models and all4 predetermined native witnesses fixed by050bf88/2bd51b6, no selection. Derive only original4-op MUL/ADD/LOGISTIC/QUANTIZE stateless fragment with original buffers/types/grids/op versions unchanged. Enumerate every256 int8 FC input codes, then every original saved actualFC step; two fresh builtinref fragment interpreters, save all4 stage outputs and full repeats. Require witness UINT8 exactly reproduces original wholemodel outputs before reporting. Nominal coordinate uses original boundedC float32 (dequantized exactFC-3)*(8/3), clipped[-8,8], then float64 sigmoid for descriptive probabilities. Report whole-enumeration nominal-versus-fragment bound and signed five-step decomposition on every original witness. No error ceiling chosen, no original gate altered. Fragment fidelity on792steps is not full legal-domain comparison to another implementation or real P4. No fullmodel/encoder invocation, optimizer, recompile/requantize original model, candidate export/promotion, fresh data/development/acceptance, waveform or board operations.')
    with PLAN.open('x') as stream:
        stream.write(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(plan=str(PLAN), sha256=sha(PLAN))))


def run(args):
    import flatbuffers
    import numpy as np
    from ai_edge_litert import schema_py_generated as schema
    from ai_edge_litert.interpreter import Interpreter, OpResolverType

    lock = json.loads(PLAN.read_text())
    assert tracked('origin/research/hey-nyx-4090', str(PLAN.relative_to(REPO))) == PLAN.read_bytes()
    assert sha(Path(__file__)) == lock['script_sha256']
    assert sha(PARENT) == lock['parent_plan_sha256'] and sha(NATIVE) == lock['native_report_sha256']
    assert sha(args.native_root / 'report.json') == lock['native_report_sha256']
    for name, expected in lock['versions'].items():
        assert importlib.metadata.version(name) == expected
    args.output_dir.mkdir(exist_ok=False)
    results = []

    def smooth_metrics(signed):
        smooth = np.abs(np.convolve(signed, np.ones(5)/5, mode='valid'))
        return dict(mean=float(smooth.mean()), p99=float(np.quantile(smooth, .99)),
                    maximum=float(smooth.max()))

    def nominal(codes, scale, zero):
        dequantized = ((codes.astype(np.float64)-zero)*scale).astype(np.float32)
        coordinate = np.clip((dequantized-np.float32(3))*np.float32(8/3), -8, 8)
        return coordinate, 1/(1+np.exp(-coordinate.astype(np.float64)))

    for case in lock['cases']:
        path = args.payload_root / case['model_path']
        assert sha(path) == case['model_sha256']
        original = path.read_bytes()
        graph = schema.ModelT.InitFromObj(schema.Model.GetRootAsModel(original, 0))
        subgraph = graph.subgraphs[0]
        assert len(subgraph.operators) == case['original_ops_count']
        selected = []
        for description in case['selected']:
            op = subgraph.operators[description['ordinal']]
            assert graph.operatorCodes[op.opcodeIndex].builtinCode == getattr(schema.BuiltinOperator, description['op'])
            assert list(op.inputs) == description['inputs'] and list(op.outputs) == description['outputs']
            selected.append(op)
        stage_ids = [int(op.outputs[0]) for op in selected]
        fc_id = case['fc_output']
        assert int(selected[0].inputs[0]) == fc_id
        assert len(subgraph.outputs) == 1 and int(subgraph.outputs[0]) == stage_ids[-1]
        fc_quant = subgraph.tensors[fc_id].quantization
        assert len(fc_quant.scale) == len(fc_quant.zeroPoint) == 1
        scale, zero = float(fc_quant.scale[0]), int(fc_quant.zeroPoint[0])
        output_quant = subgraph.tensors[stage_ids[-1]].quantization
        output_scale, output_zero = float(output_quant.scale[0]), int(output_quant.zeroPoint[0])
        subgraph.operators = selected
        subgraph.inputs = np.asarray([fc_id], np.int32)
        graph.subgraphs = [subgraph]
        graph.signatureDefs = []
        builder = flatbuffers.Builder(0)
        builder.Finish(graph.Pack(builder), file_identifier=b'TFL3')
        fragment = bytes(builder.Output())
        name = case['target'] + '-' + case['condition']
        fragment_path = args.output_dir / (name + '-post-head.tflite')
        fragment_path.write_bytes(fragment)
        saved_path = args.native_root / case['arrays']
        assert sha(saved_path) == case['arrays_sha256']
        with np.load(saved_path, allow_pickle=False) as saved:
            witness_fc = saved['actual_fc'].copy()
            witness_output = saved['uint8'].copy()
            witness_float = saved['float'].copy()
        assert witness_fc.dtype == np.int8 and witness_output.dtype == np.uint8
        assert witness_fc.shape == witness_output.shape == witness_float.shape == (case['steps'],)
        enumeration = np.arange(-128, 128, dtype=np.int16).astype(np.int8)
        inputs = np.concatenate([enumeration, witness_fc])
        runs = []
        for repeat in range(2):
            interpreter = Interpreter(model_content=fragment,
                experimental_op_resolver_type=OpResolverType.BUILTIN_REF,
                experimental_preserve_all_tensors=True, num_threads=1)
            interpreter.allocate_tensors()
            inp, out = interpreter.get_input_details()[0], interpreter.get_output_details()[0]
            assert inp['index'] == fc_id and tuple(inp['shape']) == (1, 1) and inp['dtype'] == np.int8
            assert out['index'] == stage_ids[-1] and out['dtype'] == np.uint8
            stages = {index: [] for index in stage_ids}
            for code in inputs:
                interpreter.set_tensor(fc_id, np.asarray([[code]], np.int8))
                interpreter.invoke()
                for index in stage_ids:
                    stages[index].append(interpreter.get_tensor(index).copy().reshape(-1))
            runs.append({index: np.concatenate(values) for index, values in stages.items()})
        assert all(np.array_equal(runs[0][index], runs[1][index]) for index in stage_ids)
        assert np.array_equal(runs[0][stage_ids[-1]][256:], witness_output)
        assert np.array_equal(runs[0][stage_ids[-1]][:256][witness_fc.astype(np.int16)+128],
                              witness_output)
        assert sha(path) == case['model_sha256']
        enum_logit, enum_nominal = nominal(enumeration, scale, zero)
        witness_logit, witness_nominal = nominal(witness_fc, scale, zero)
        enum_actual = (runs[0][stage_ids[-1]][:256].astype(np.float64)-output_zero)*output_scale
        actual = (witness_output.astype(np.float64)-output_zero)*output_scale
        head_error = enum_actual-enum_nominal
        head_bound = float(np.max(np.abs(head_error)))
        total = actual-witness_float
        post = actual-witness_nominal
        residual = witness_nominal-witness_float
        assert np.allclose(total, post+residual, rtol=0, atol=2e-16)
        assert np.max(np.abs(post)) <= head_bound
        metrics = {key: smooth_metrics(value) for key, value in
                   [('original_total', total), ('post_head_component', post),
                    ('nominal_fc_vs_float_component', residual)]}
        lower = max(0., metrics['original_total']['p99']-head_bound)
        assert metrics['nominal_fc_vs_float_component']['p99']+2e-16 >= lower
        arrays = dict(enum_fc_codes=enumeration, enum_nominal_logit=enum_logit,
                      enum_nominal_probability=enum_nominal, enum_actual_probability=enum_actual,
                      witness_fc=witness_fc, witness_uint8=witness_output,
                      witness_float=witness_float, witness_nominal_logit=witness_logit,
                      witness_nominal_probability=witness_nominal,
                      total_signed_error=total, post_head_signed_error=post,
                      nominal_fc_vs_float_signed_error=residual)
        for repeat, stages in enumerate(runs):
            for index, value in stages.items():
                arrays['r' + str(repeat) + '_stage' + str(index)] = value
        destination = args.output_dir / (name + '.npz')
        np.savez_compressed(destination, **arrays)
        result = dict(target=case['target'], condition=case['condition'],
            model_sha256=case['model_sha256'], original_arrays_sha256=case['arrays_sha256'],
            witness_steps=case['steps'], fc_quantization=[scale, zero],
            output_quantization=[output_scale, output_zero], stage_ids=stage_ids,
            enum_codes=256, fragment_invocations=2*len(inputs),
            full_witness_uint8_reproduced=True, fresh_fragment_repeats_exact=True,
            enum_max_abs_nominal_probability_gap=head_bound,
            enum_mean_abs_nominal_probability_gap=float(np.mean(np.abs(head_error))),
            smoothed_abs_error=metrics,
            residual_five_step_p99_lower_bound_from_uniform_post_head_gap=lower,
            arrays=destination.name, arrays_sha256=sha(destination),
            fragment=fragment_path.name, fragment_sha256=sha(fragment_path))
        results.append(result)
        print(json.dumps(result), flush=True)
    report = dict(schema='mac-consistency-post-head-gap-result-v1',
        plan_sha256=sha(PLAN), cases=results, execution_checks_passed=True,
        total_wholemodel_invocations=0,
        total_fragment_invocations=sum(case['fragment_invocations'] for case in results),
        total_original_witness_steps=sum(case['witness_steps'] for case in results),
        optimizer_updates=0, versions=lock['versions'], limits=lock['policy'])
    with (args.output_dir / 'report.json').open('x') as stream:
        stream.write(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['plan', 'run'])
    parser.add_argument('--payload-root', type=Path, required=True)
    parser.add_argument('--native-root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    make_plan(args) if args.action == 'plan' else run(args)
