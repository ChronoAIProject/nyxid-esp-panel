"""Freeze then verify the FP32 helper on six already consumed numeric witnesses."""
import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import subprocess

import numpy as np

import complete_fp32_feature_stream_v1 as helper

REPO = Path(__file__).resolve().parents[3]


def sha(data):
    return hashlib.sha256(data).hexdigest()


def freeze(packet):
    intake = REPO / 'research/wakeword_p4/results/mac-fp32-internal-fullstream-intake-v1.json'
    manifest_bytes = (packet / 'manifest.json').read_bytes()
    assert sha(manifest_bytes) == json.loads(intake.read_text())['manifest_sha256']
    manifest = json.loads(manifest_bytes)

    def member(name):
        data = (packet / name).read_bytes()
        assert len(data) == manifest['files'][name]['bytes']
        assert sha(data) == manifest['files'][name]['sha256']
        return data

    lock_name = 'receipts/4090-fp32-internal-fullstream-execution-v1.json'
    lock = json.loads(member(lock_name))
    requests = lock['requests']
    assert len(requests) == 135
    name = 'hey_nyx-float_control'
    model_path = 'models/' + name + '.tflite'
    model = member(model_path)
    assert sha(model) == next(item['sha256'] for item in lock['exports']
                              if item['target'] + '-' + item['condition'] == name)
    cases = []
    for target in ('hey_willow', 'hey_nyx'):
        for engine in ('piper', 'kokoro', 'qwen'):
            index = next(i for i, row in enumerate(requests) if row['variant'] == 0
                         and row['source']['target'] == target and row['source']['engine'] == engine)
            path = f'arrays/{name}-{index:03}.npz'
            cases.append(dict(index=index, target=target, engine=engine, samples=requests[index]['samples'],
                              path=path, sha256=sha(member(path))))
    sources = [Path(__file__).resolve(), Path(helper.__file__).resolve()]
    runtime = Path(importlib.util.find_spec('ai_edge_litert').origin).parent
    runtime_files = sorted(path for path in runtime.rglob('*')
                           if path.is_file() and path.suffix in ('.py', '.so', '.dylib'))
    return dict(schema='mac-complete-fp32-helper-numeric-plan-v1',
        source_sha256={str(path.relative_to(REPO)): sha(path.read_bytes()) for path in sources},
        runtime_files={str(path): sha(path.read_bytes()) for path in runtime_files},
        versions={name: importlib.metadata.version(name) for name in ('numpy', 'ai-edge-litert')},
        original_manifest_sha256=sha(manifest_bytes), original_execution_sha256=sha(member(lock_name)),
        model=dict(path=model_path, sha256=sha(model)), cases=cases, repeats=2, max_abs=1e-4,
        scope='Six original consumed raw-feature/complete-clock numeric witnesses, one old raw model. No new weight, recognition/FAR, fresh development, acceptance, audio IO or P4 operation.')


def constructed():
    model = b'explicitly constructed model stand-in'
    signature = sha(model)
    instances = []

    class Fake:
        def __init__(self, _model):
            self.calls = 0
            instances.append(self)
        def allocate_tensors(self): pass
        def get_input_details(self):
            return [dict(index=0, shape=[1, 3, 40], dtype=np.float32)]
        def get_output_details(self):
            return [dict(index=1, shape=[1, 1], dtype=np.float32)]
        def set_tensor(self, _index, values): self.last = values.copy()
        def invoke(self): self.calls += 1
        def get_tensor(self, _index): return np.array([[self.calls / 1000]], np.float32)

    for samples in (1, 319, 320, 321, 479, 480, 481, 639, 640, 641, 999, 10000):
        frames = max(0, (samples + 159) // 160 - 2)
        raw = np.arange(frames * 40, dtype=np.uint16).reshape(frames, 40)
        first, clocks, tail = helper.predict_complete_fp32_features(model, signature, raw, samples, Fake)
        second, _, again = helper.predict_complete_fp32_features(model, signature, raw, samples, Fake)
        expected_calls = len(list(range(0, frames, 3)))
        assert first.tobytes() == second.tobytes() and tail == again
        assert instances[-1] is not instances[-2] and instances[-1].calls == expected_calls
        assert first.shape == (expected_calls,)
        assert clocks.tolist() == [min((3 * step + 2) * 160, samples) for step in range(1, expected_calls + 1)]
        assert tail['final_feature_padding'] == (-frames) % 3
        if expected_calls:
            expected = np.zeros((1, 3, 40), np.float32)
            start = (expected_calls - 1) * 3
            expected[0, :frames-start] = np.minimum(raw[start:].astype(np.float32) / 25.6, 26)
            assert instances[-1].last.tobytes() == expected.tobytes()
    invalid = [('', np.zeros((1, 40), np.uint16), 480),
               (signature, np.zeros((1, 40), np.int16), 480),
               (signature, np.zeros((2, 40), np.uint16), 480),
               (signature, np.zeros((1, 40), np.uint16), True)]
    for digest, raw, samples in invalid:
        try: helper.predict_complete_fp32_features(model, digest, raw, samples, Fake)
        except ValueError: pass
        else: raise AssertionError('Malformed input admitted')
    class BadInput(Fake):
        def get_input_details(self):
            return [dict(index=0, shape=[1, 3, 40], dtype=np.int8)]
    class BadOutput(Fake):
        def get_output_details(self):
            return [dict(index=1, shape=[1, 1], dtype=np.uint8)]
    class BadProbability(Fake):
        def get_tensor(self, _index): return np.array([[np.nan]], np.float32)
    for factory in (BadInput, BadOutput, BadProbability):
        try: helper.predict_complete_fp32_features(model, signature, np.zeros((1, 40), np.uint16), 480, factory)
        except ValueError: pass
        else: raise AssertionError('Invalid FP32 runtime admitted')
    return dict(constructed_lengths=12, constructed_corruptions_rejected=7,
                fresh_interpreters_and_final_pad_verified=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('freeze', 'run'))
    for name in ('packet', 'plan'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--published-ref')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    plan = freeze(args.packet)
    if args.action == 'freeze':
        with args.plan.open('x') as stream: stream.write(json.dumps(plan, indent=2) + '\n')
        print(json.dumps(dict(cases=6, repeats=2, model_calls=0, plan_sha256=sha(args.plan.read_bytes()))))
        return
    assert args.published_ref and args.output is not None and not args.output.exists()
    relative = args.plan.resolve().relative_to(REPO)
    published = subprocess.check_output(['git', 'show', args.published_ref + ':' + str(relative)], cwd=REPO)
    assert args.plan.read_bytes() == published and json.loads(published) == plan
    result = dict(schema='mac-complete-fp32-helper-numeric-result-v1',
                  plan_sha256=sha(published), cases=[], all_passed=False)
    try:
        result['constructed'] = constructed()
        model = (args.packet / plan['model']['path']).read_bytes()
        for case in plan['cases']:
            with np.load(args.packet / case['path'], allow_pickle=False) as arrays:
                raw = arrays['raw']
                expected_input = sha(arrays['input'].tobytes())
                first, clocks, tail = helper.predict_complete_fp32_features(model, plan['model']['sha256'], raw, case['samples'])
                second, other_clocks, other_tail = helper.predict_complete_fp32_features(model, plan['model']['sha256'], raw, case['samples'])
                assert first.tobytes() == second.tobytes() and tail == other_tail
                assert np.array_equal(clocks, other_clocks) and np.array_equal(clocks, arrays['sample_end'])
                assert tail['normalized_input_sha256'] == expected_input
                gaps = {key: float(np.max(np.abs(first.astype(float) - arrays[key].astype(float)))) for key in ('tf', 'litert')}
                assert all(gap <= plan['max_abs'] for gap in gaps.values())
                result['cases'].append(dict(**case, steps=len(first), tail=tail, max_abs=gaps,
                    repeat_exact=True, normalized_input_exact=True, complete_clocks_exact=True))
        result.update(all_passed=True, actual_invocations=2 * sum(case['steps'] for case in result['cases']),
                      limits=plan['scope'], new_candidate_qualified=False)
    except BaseException as error:
        result['error'] = dict(type=type(error).__name__, detail=str(error))
        raise
    finally:
        with args.output.open('x') as stream: stream.write(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(cases=6, actual_invocations=result['actual_invocations'], all_passed=True)))


if __name__ == '__main__':
    main()
