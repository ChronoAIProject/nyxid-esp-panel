"""One fixed streaming export per smoke model, then fresh-interpreter ABI replay."""
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
LOCK = R / '4090-multispeaker-smoke-export-plan-v1.json'
OUT = Path('.local/wakeword/multispeaker-smoke-export-v1')
MATERIAL = Path('.local/wakeword/multispeaker-smoke-export-material-v1')
ORIGINAL = Path(__file__).with_name('export_engine_adversarial_completed_nyx_v1.py')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def lock():
    smoke_path = R / '4090-multispeaker-gpu-smoke-v1.json'
    audit_path = R / '4090-multispeaker-gpu-smoke-saved-state-audit-v1.json'
    smoke, audit = [json.loads(p.read_text()) for p in [smoke_path, audit_path]]
    assert audit['original_report_sha256'] == sha(smoke_path) and audit['all_saved_state_checks_passed']
    source_plan = R / '4090-multispeaker-gpu-smoke-plan-v1.json'
    original_plan = json.loads(source_plan.read_text())
    files = dict(original_plan['files'])
    paths = [Path(__file__), ORIGINAL, smoke_path, audit_path, source_plan]
    weights, material = [], []
    MATERIAL.mkdir(exist_ok=False)
    for arm in ['narrow', 'broad']:
        row = next(r for r in smoke['arms'] if r['arm'] == arm)
        path = Path('.local/wakeword/multispeaker-gpu-smoke-v1') / arm / 'final.weights.h5'
        assert sha(path) == row['files']['final.weights.h5']
        paths.append(path)
        weights.append(dict(target='hey_nyx', condition=arm, path=str(path), sha256=sha(path)))
        batch = original_plan['batches'][arm][0]
        assert sha(batch['actual_tensor_path']) == batch['actual_tensor_sha256']
        with np.load(batch['actual_tensor_path'], allow_pickle=False) as saved:
            representative = saved['x'][:16, None].copy()
        assert representative.shape == (16, 1, 309, 40)
        path = MATERIAL / (arm + '.npy')
        np.save(path, representative, allow_pickle=False)
        paths.append(path)
        material.append(dict(target='hey_nyx', condition=arm, path=str(path), sha256=sha(path),
                             shape=list(representative.shape), source_actual_batch_sha256=batch['actual_tensor_sha256']))
    paths += list(Path('.local/wakeword/microWakeWord/microwakeword').rglob('*.py'))
    files.update({str(p): sha(p) for p in paths})
    for p, h in files.items():
        assert sha(p) == h, p
    plan = dict(schema='multispeaker-smoke-export-plan-v1', files=files, weights=weights, material=material,
                versions={n: importlib.metadata.version(n) for n in ['tensorflow', 'keras', 'numpy', 'ai-edge-litert']},
                context_frames=309, expected_new_gib=.1, retries=0,
                contract='Disposable fixed-final2 smoke models only. One raw-sigmoid FP32 and builtin INT8 export each, unchanged frozen BN, six zero-real states. Original exporter arithmetic reused with explicit context225→309, per-arm material selector, and metadata keyword substitutions. Representative fixed first16 actual training batch windows/arm plus original zero/26 sentinels; no intermediate resets. Fresh interpreter103calls×2repeats/artifact on first actual training window; save all raw outputs. ABI/reset repeatability only, no FP32/INT8 accuracy bound, true whole-wave artifact qualification, recognition, new threshold, holdout, formal training or P4 resource admission.')
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
    result = dict(schema='multispeaker-smoke-export-runtime-v1', execution_sha256=sha(LOCK),
                  static_result_sha256=sha(OUT / 'result.json'), artifacts=rows,
                  all_static_and_runtime_abi_checks_passed=True, limits=plan['contract'])
    (OUT / 'runtime.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(exports=4, actual_interpreter_invocations=824, repeats_exact=True)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['lock', 'run'])
    args = parser.parse_args()
    lock() if args.action == 'lock' else run()
