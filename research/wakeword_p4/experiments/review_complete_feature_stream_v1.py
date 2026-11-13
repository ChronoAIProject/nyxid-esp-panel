"""Constructed invocation/tail audit using the actual pinned detector class.

The fake interpreter records every real quantized input block and has a simple
stateful output. No trained-model inference, audio or development data is used.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import numpy as np

from complete_feature_stream_v1 import POLICY, predict_complete_features

DETECTOR = 'research/wakeword_p4/experiments/uint8_stream_detector_v1.py'
CASES = (0, 1, 2, 3, 4, 5, 6, 7, 8, 149, 150, 151, 224, 225, 226, 599, 600, 601)


class ObservedInterpreter:
    def __init__(self, **kwargs):
        self.blocks = []; self.state = 0; self.result = None

    def allocate_tensors(self):
        pass

    def get_input_details(self):
        return [dict(dtype=np.int8, shape=(1, 3, 40), quantization=(.125, -128), index=0)]

    def get_output_details(self):
        return [dict(dtype=np.uint8, shape=(1, 1), quantization=(1/256, 0), index=1)]

    def set_tensor(self, index, values):
        assert index == 0 and values.shape == (1, 3, 40) and values.dtype == np.int8
        self.current = values.copy()

    def invoke(self):
        self.blocks.append(self.current.copy())
        self.state = (self.state*3+int(self.current.astype(np.int64).sum())) % 256
        self.result = np.array([[self.state]], dtype=np.uint8)

    def get_tensor(self, index):
        assert index == 1
        return self.result.copy()


def refused(call):
    try:
        call()
    except ValueError:
        return True
    raise AssertionError('Invalid/truncated stream unexpectedly accepted')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--gpu-ref', default='7d34e9d')
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    commit = subprocess.check_output(['git', 'rev-parse', args.gpu_ref], text=True).strip()
    source = subprocess.check_output(['git', 'show', commit+':'+DETECTOR])
    tree = ast.parse(source)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Uint8StreamDetector')
    env = dict(np=np, Interpreter=ObservedInterpreter,
               OpResolverType=SimpleNamespace(BUILTIN_REF='constructed-reference'))
    exec(compile(ast.Module(body=[cls], type_ignores=[]), DETECTOR, 'exec'), env)
    Detector = env['Uint8StreamDetector']
    results = []
    for frames in CASES:
        raw = (np.arange(frames*40).reshape(frames, 40) % 600).astype(np.uint16)
        saved = raw.copy(); raw.flags.writeable = False
        samples = (frames+2)*160-17
        old = Detector('constructed'); historical = old.predict(raw)
        detector = Detector('constructed')
        output, clocks, meta = predict_complete_features(detector, raw, samples)
        assert len(historical) == frames//3 and len(output) == (frames+2)//3
        assert np.array_equal(output[:len(historical)], historical)
        assert all(np.array_equal(a, b) for a, b in zip(
            detector.interpreter.blocks[:len(historical)], old.interpreter.blocks))
        assert np.array_equal(saved, raw)
        actual = (np.concatenate(detector.interpreter.blocks, axis=1)[0]
                  if len(output) else np.empty((0, 40), dtype=np.int8))
        q = np.clip(np.rint(raw.astype(np.float32)/25.6/.125)-128, -128, 127).astype(np.int8)
        assert np.array_equal(actual[:frames], q)
        assert np.array_equal(actual[frames:], np.full(((-frames)%3, 40), -128, dtype=np.int8))
        assert len(actual) == frames+(-frames)%3
        assert np.array_equal(clocks, np.array(
            [min((3*s+2)*160, samples) for s in range(1, len(output)+1)], dtype=np.int64))
        detector.reset()
        repeat, repeat_clocks, repeat_meta = predict_complete_features(detector, raw, samples)
        assert np.array_equal(output, repeat) and np.array_equal(clocks, repeat_clocks) and meta == repeat_meta
        tail_changes_only_last_step = None
        if frames % 3:
            changed = raw.copy(); changed[-1, 0] += 26
            detector.reset(); altered, _, _ = predict_complete_features(detector, changed, samples)
            assert np.array_equal(output[:-1], altered[:-1]) and output[-1] != altered[-1]
            tail_changes_only_last_step = True
        results.append(dict(frames=frames, samples=samples, historical_calls=len(historical),
            complete_calls=len(output), final_feature_padding=meta['final_feature_padding'],
            all_real_frames_in_order_once=True, quantized_padding_is_zero_point=True,
            historical_output_prefix_exact=True, fresh_reset_repeat_exact=True,
            tail_changes_only_last_step=tail_changes_only_last_step,
            output_sha256=hashlib.sha256(output.tobytes()).hexdigest()))
    detector = Detector('constructed')
    raw = np.zeros((4, 40), dtype=np.uint16)
    invalid = [refused(lambda: predict_complete_features(detector, raw.astype(np.float32), 943)),
               refused(lambda: predict_complete_features(detector, raw[:, :39], 943)),
               refused(lambda: predict_complete_features(detector, raw, True)),
               refused(lambda: predict_complete_features(detector, raw, 1103))]
    class TruncatedDetector:
        def predict(self, padded):
            return np.zeros(len(padded)//3-1, dtype=np.uint8)
    invalid.append(refused(lambda: predict_complete_features(TruncatedDetector(), raw, 943)))
    helper = Path(__file__).with_name('complete_feature_stream_v1.py')
    result = dict(schema='complete-feature-stream-constructed-review-v1', policy=POLICY,
        gpu_commit=commit, original_detector_sha256=hashlib.sha256(source).hexdigest(),
        helper_sha256=hashlib.sha256(helper.read_bytes()).hexdigest(),
        audit_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        cases=results, invalid_or_truncated_inputs_rejected=len(invalid),
        historical_scorers_modified=False,
        limits='Actual pinned detector quantization/iteration code, constructed features and fake stateful interpreter only. No trained model, frontend/audio traversal, real development/acceptance, FAR/latency or P4 inference certification.')
    with args.output.open('x') as out:
        out.write(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(cases=len(results),invalid_rejections=len(invalid),all_checks_pass=True)))


if __name__ == '__main__':
    main()
