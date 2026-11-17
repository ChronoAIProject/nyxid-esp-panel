"""Reproduce and fix context omissions in the GPU's actual review block.

Uses tiny constructed JSON/NPZ fixtures only; no inference or acoustic inputs.
"""
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
GPU_COMMIT = '757962e71ed50a2f33cb11024e30947608b62448'
GPU_PATH = 'research/wakeword_p4/experiments/review_binary40_additive_carrier_consumed_v1.py'
NAMES = ['builtin', 'melo-US', 'melo-BR']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value))


def block(source):
    nodes = ast.parse(source).body
    def assigns(node, name):
        return isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets)
    start = next(i for i, node in enumerate(nodes) if assigns(node, 'context_checks'))
    end = next(i for i in range(start + 1, len(nodes)) if assigns(nodes[i], 'out'))
    return compile(ast.fix_missing_locations(ast.Module(body=nodes[start:end], type_ignores=[])), 'actual-context-review-block', 'exec')


def fixture(root):
    timeline = [dict(kind='clip', group='constructed', start=i * 100, end=(i + 1) * 100) for i in range(48)]
    source = root / '4090-parallel-context-streams-plan-v1.json'
    reference = root / '4090-parallel-context-streams-original-v1.json'
    write(source, dict(streams=[dict(name=n, items=list(range(48))) for n in NAMES]))
    write(reference, dict(plan_sha256=sha(source), streams=[dict(name=n, timeline=timeline, timing=dict(calls=8)) for n in NAMES]))
    for arm in ['control', 'additive']:
        model = root / (arm + '.tflite'); model.write_bytes(('identity-only-' + arm).encode())
        export = root / f'4090-binary40-additive-carrier-{arm}-export-original-v1.json'
        model_info = dict(path=str(model), bytes=model.stat().st_size, sha256=sha(model))
        write(export, dict(arm=arm, model=model_info))
        plan = root / f'4090-binary40-additive-carrier-{arm}-context-plan-v1.json'
        write(plan, dict(arm=arm, model=model_info, files={str(p): sha(p) for p in [source, reference, export, model]}))
        rows = []
        for name in NAMES:
            trace = root / f'{arm}-{name}.npz'
            np.savez(trace, uint8=np.zeros(8, np.uint8), relative_block_return=np.arange(8, dtype=np.int64) * 480 - 81600)
            rows.append(dict(name=name, timeline=timeline, events=[], groups=dict(constructed=dict(clips=48, hits=0)), trace=dict(path=str(trace), bytes=trace.stat().st_size, sha256=sha(trace))))
        write(root / f'4090-binary40-additive-carrier-{arm}-context-original-v1.json', dict(arm=arm, all_passed=True, plan_sha256=sha(plan), streams=rows))


def run(code, mutation=None):
    with tempfile.TemporaryDirectory(prefix='nyx-context-contract-') as tmp:
        root = Path(tmp); fixture(root)
        report = root / '4090-binary40-additive-carrier-control-context-original-v1.json'
        data = json.loads(report.read_text())
        if mutation:
            mutation(root, data)
            write(report, data)
        env = dict(R=root, Path=Path, np=np, sha=sha, read=lambda path: json.loads(Path(path).read_text()))
        exec(code, env)
        return env['context_checks']


def main():
    original = subprocess.check_output(['git', 'show', GPU_COMMIT + ':' + GPU_PATH], cwd=ROOT, text=True)
    patch = ROOT / 'patches/research_additive_context_completeness.patch'
    with tempfile.TemporaryDirectory(prefix='nyx-review-patch-') as tmp:
        work = Path(tmp); target = work / GPU_PATH; target.parent.mkdir(parents=True); target.write_text(original)
        subprocess.run(['git', 'apply', '--check', str(patch)], cwd=work, check=True)
        subprocess.run(['git', 'apply', str(patch)], cwd=work, check=True)
        fixed = target.read_text()
    before, after = block(original), block(fixed)
    def empty(root, data): data['streams'] = []
    assert run(before, empty)['control'] == []
    assert len(run(after)['control']) == 3
    def truncated(root, data):
        row = data['streams'][0]; path = Path(row['trace']['path'])
        np.savez(path, uint8=np.zeros(7, np.uint8), relative_block_return=np.arange(7, dtype=np.int64))
        row['trace'].update(bytes=path.stat().st_size, sha256=sha(path))
    def wrong_model(root, data):
        path = root / '4090-binary40-additive-carrier-control-context-plan-v1.json'; plan = json.loads(path.read_text())
        plan['model'] = dict(plan['model'], sha256='0' * 64); write(path, plan); data['plan_sha256'] = sha(path)
    def missing_pin(root, data):
        path = root / '4090-binary40-additive-carrier-control-context-plan-v1.json'; plan = json.loads(path.read_text())
        del plan['files'][str(root / '4090-parallel-context-streams-original-v1.json')]
        write(path, plan); data['plan_sha256'] = sha(path)
    mutations = dict(empty=empty, missing_stream=lambda r,d: d['streams'].pop(),
                     duplicate_stream=lambda r,d: d['streams'].__setitem__(1, d['streams'][0]),
                     wrong_timeline=lambda r,d: d['streams'][0]['timeline'].pop(),
                     wrong_plan=lambda r,d: d.__setitem__('plan_sha256', '0' * 64),
                     truncated_trace=truncated, wrong_model=wrong_model, missing_reference_pin=missing_pin)
    for name, mutation in mutations.items():
        try: run(after, mutation)
        except (AssertionError, KeyError): pass
        else: raise AssertionError('Accepted incomplete/misidentified context: ' + name)
    print(json.dumps(dict(original_empty_context_accepted=True, patched_complete_six_contexts_passed=True,
                          invalid_cases_rejected=list(mutations), gpu_commit=GPU_COMMIT,
                          source_sha256=hashlib.sha256(original.encode()).hexdigest(), patch_sha256=sha(patch),
                          new_inference_training_baseline_tts_or_recording=False)))


if __name__ == '__main__': main()
