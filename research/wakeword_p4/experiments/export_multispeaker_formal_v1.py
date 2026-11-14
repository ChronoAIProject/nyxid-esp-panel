"""One fixed streaming export per formal fixed-final model, then fresh-interpreter ABI replay."""
import argparse
import ast
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import traceback

import numpy as np

R = Path('research/wakeword_p4/results')
LOCK = R / '4090-multispeaker-formal-export-plan-v1.json'
OUT = Path('.local/wakeword/multispeaker-formal-export-v1')
MATERIAL = Path('.local/wakeword/multispeaker-smoke-export-material-v1')
ORIGINAL = Path(__file__).with_name('export_engine_adversarial_completed_nyx_v1.py')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def lock():
    formal = R / '4090-multispeaker-formal-original-v1.json'
    audit_path = R / '4090-multispeaker-formal-independent-audit-v1.json'
    report, audit = [json.loads(p.read_text()) for p in [formal, audit_path]]
    assert audit['original_report_sha256'] == sha(formal)
    assert audit['all_saved_state_checks_passed'] and audit['all_batch_digests_labels_roles_geometry_and_paired_cells_match']
    assert audit['batches_reconstructed'] == 40000
    smoke_plan = R / '4090-multispeaker-smoke-export-plan-v1.json'
    previous = json.loads(smoke_plan.read_text())
    files = dict(previous['files'])
    material = previous['material']
    paths = [Path(__file__), ORIGINAL, formal, audit_path, smoke_plan]
    paths += [Path(__file__).with_name('train_multispeaker_formal_v1.py'),
              Path(__file__).with_name('audit_multispeaker_formal_v1.py'),
              R / '4090-multispeaker-formal-execution-v1.json']
    weights = []
    for arm in ['narrow', 'broad']:
        row = next(r for r in report['arms'] if r['arm'] == arm)
        assert row['steps'] == row['optimizer_iterations'] == 20000
        path = Path('.local/wakeword/multispeaker-formal-v1') / arm / 'final.weights.h5'
        assert sha(path) == row['files']['final.weights.h5']
        paths.append(path)
        weights.append(dict(target='hey_nyx', condition=arm, path=str(path), sha256=sha(path)))
    files.update({str(p): sha(p) for p in paths})
    for p, h in files.items():
        assert sha(p) == h, p
    plan = dict(schema='multispeaker-formal-export-plan-v1', files=files, weights=weights, material=material,
                versions={n: importlib.metadata.version(n) for n in ['tensorflow', 'keras', 'numpy', 'ai-edge-litert']},
                context_frames=309, expected_new_gib=.1, retries=0,
                contract='Two original fixedfinal20000 models only, broad sole candidate/narrow diagnostic. Unchanged rawsigmoid/frozenBN. OneFP32+onebuiltinINT8 each. Fixedsamefirst16actualtrainingwindows/arm from preformal smoke material plus zero/26sentinels, no intermediate calibrationreset. Sixrealzero states. StaticABI and 824freshinterpretercalls only; numerical fidelity/wholewaveartifact/recognition/freshdev/acceptance/P4resources remain separate. No weight/calibration/checkpoint rescue.')
    with LOCK.open('x') as stream:
        stream.write(json.dumps(plan, indent=2) + '\n')
    print(json.dumps(dict(plan_sha256=sha(LOCK), models=2, exports=4)))


def run():
    plan = json.loads(LOCK.read_text())
    subprocess.run(['/usr/bin/python3', 'research/wakeword_p4/tools/check_disk_budget.py',
                    '--expected-gib', str(plan['expected_new_gib'])], check=True)
    tree = ast.parse(ORIGINAL.read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'run')

    class Adapter(ast.NodeTransformer):
        def visit_Constant(self, node):
            if node.value == 225:
                return ast.copy_location(ast.Constant(309), node)
            return node

        def visit_Assign(self, node):
            if any(isinstance(t, ast.Name) and t.id == 'material' for t in node.targets):
                return ast.copy_location(ast.parse("material=next(m for m in p['material'] if m['condition']==weight['condition'])").body[0], node)
            return self.generic_visit(node)

        def visit_keyword(self, node):
            if node.arg == 'four_arm_complete':
                node.arg = 'formal_training_completed'
                node.value = ast.Constant(True)
            return self.generic_visit(node)

    env = dict(__file__=str(ORIGINAL), subprocess=subprocess, json=json, np=np,
               Path=Path, LOCK=LOCK, OUT=OUT, sha=sha, traceback=traceback,
               importlib=importlib)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[Adapter().visit(function)], type_ignores=[])), str(ORIGINAL), 'exec'), env)
    env['run']()
    # Preserve the original static result bytes; runtime results are additive.
    from ai_edge_litert.interpreter import Interpreter
    static = json.loads((OUT / 'result.json').read_text())
    assert static['all_static_passed'] and static['execution_sha256'] == sha(LOCK)
    rows = []
    for model in static['models']:
        material = next(m for m in plan['material'] if m['condition'] == model['condition'])
        values = np.load(material['path'], allow_pickle=False)[0, 0]
        for kind, artifact in model['artifacts'].items():
            assert sha(artifact['path']) == artifact['sha256']
            traces = []
            for repeat in range(2):
                interpreter = Interpreter(model_path=artifact['path'])
                interpreter.allocate_tensors()
                inputs, outputs = interpreter.get_input_details(), interpreter.get_output_details()
                assert len(inputs) == len(outputs) == 1
                a, b = inputs[0], outputs[0]
                assert list(a['shape']) == [1, 3, 40] and list(b['shape']) == [1, 1]
                if kind == 'int8':
                    assert a['dtype'] == np.int8 and b['dtype'] == np.uint8
                    scale, zero = a['quantization']
                    assert scale > 0 and b['quantization'] == (1 / 256, 0)
                    supplied = np.clip(np.rint(values / scale) + zero, -128, 127).astype(np.int8)
                else:
                    assert a['dtype'] == b['dtype'] == np.float32
                    supplied = values
                raw = []
                for at in range(0, 309, 3):
                    interpreter.set_tensor(a['index'], supplied[None, at:at + 3])
                    interpreter.invoke()
                    raw.append(interpreter.get_tensor(b['index'])[0, 0])
                traces.append(np.asarray(raw))
            assert np.array_equal(*traces) and len(traces[0]) == 103 and np.isfinite(traces[0]).all()
            if kind == 'fp32':
                assert np.all((traces[0] >= 0) & (traces[0] <= 1))
            path = Path(artifact['path']).with_name(kind + '-abi-replay.npz')
            np.savez(path, features=values, supplied=supplied, first=traces[0], second=traces[1])
            rows.append(dict(arm=model['condition'], kind=kind, calls_per_repeat=103,
                             repeats_exact=True, trace_path=str(path), trace_sha256=sha(path),
                             input_quantization=list(a['quantization']), output_quantization=list(b['quantization'])))
    result = dict(schema='multispeaker-formal-export-runtime-v1', execution_sha256=sha(LOCK),
                  static_result_sha256=sha(OUT / 'result.json'), artifacts=rows,
                  all_static_and_runtime_abi_checks_passed=True, limits=plan['contract'])
    (OUT / 'runtime.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(exports=4, actual_interpreter_invocations=824, repeats_exact=True)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['lock', 'run'])
    args = parser.parse_args()
    lock() if args.action == 'lock' else run()
