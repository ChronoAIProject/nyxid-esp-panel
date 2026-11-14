"""Replay an existing complete WAV with and without the panel's zero-PCM priming."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
import time
import wave

import numpy as np
from ai_edge_litert.interpreter import Interpreter, OpResolverType

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'training'))
from features import features
from uint8_event_policy_v1 import EVENT_ARITHMETIC, event_steps


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def replay(model, wav, context_frames, threshold_grid, output):
    if type(context_frames) is not int or context_frames <= 0:
        raise ValueError('Expected the actual training context length')
    grid = json.loads(Path(threshold_grid).read_text())['thresholds']
    if not grid or len(set(grid)) != len(grid):
        raise ValueError('Expected a nonempty frozen threshold list')
    for threshold in grid:
        event_steps([], threshold)
    with wave.open(str(wav), 'rb') as source:
        if (source.getframerate(), source.getnchannels(), source.getsampwidth(), source.getcomptype()) != (16000, 1, 2, 'NONE'):
            raise ValueError('Expected existing 16 kHz mono PCM16 WAV')
        expected_samples = source.getnframes()
        pcm = np.frombuffer(source.readframes(expected_samples), dtype='<i2')
        if not len(pcm) or len(pcm) != expected_samples:
            raise ValueError('Empty or truncated source')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    requested = (context_frames + 200) * 160
    primed_samples = ((requested + 479) // 480) * 480
    result = dict(schema='mac-existing-panel-priming-replay-v1', model_sha256=sha(model),
                  wav_sha256=sha(wav), pcm_sha256=hashlib.sha256(pcm.tobytes()).hexdigest(),
                  original_samples=len(pcm), context_frames=context_frames,
                  requested_prime_samples=requested, rounded_prime_samples=primed_samples,
                  threshold_grid_sha256=sha(threshold_grid), event_arithmetic=EVENT_ARITHMETIC,
                  script_sha256=sha(__file__), event_policy_sha256=sha(Path(__file__).with_name('uint8_event_policy_v1.py')),
                  frontend_wrapper_sha256=sha(Path(__file__).resolve().parents[1] / 'training/features.py'),
                  versions={n: importlib.metadata.version(n) for n in ('numpy', 'ai-edge-litert', 'pymicro-features')},
                  phases=[], candidate_promoted=False,
                  limits='Existing consumed-WAV initialization diagnosis only. Full input retained; no threshold selection, new audio, hardware, fresh development or acceptance. Does not establish recognition quality or P4 behavior.')
    (output / 'inputs.json').write_text(json.dumps(result, indent=2) + '\n')
    for name, prime in [('cold', 0), ('primed', primed_samples)]:
        started = time.monotonic()
        # A single frontend processes both segments. No flush/reset at the join.
        combined = np.concatenate((np.zeros(prime, np.int16), pcm))
        raw = features(combined)
        expected_frames = max(0, (len(combined) + 159) // 160 - 2)
        if raw.shape != (expected_frames, 40):
            raise ValueError('Incomplete frontend traversal')
        values = np.clip(np.pad(raw, ((0, (-len(raw)) % 3), (0, 0))).astype(np.float32) / 25.6, 0, 26)
        interpreter = Interpreter(model_path=str(model), num_threads=1,
                                  experimental_op_resolver_type=OpResolverType.BUILTIN_REF)
        interpreter.allocate_tensors()
        inputs, outputs = interpreter.get_input_details(), interpreter.get_output_details()
        if len(inputs) != 1 or len(outputs) != 1:
            raise ValueError('Expected internal streaming states')
        a, b = inputs[0], outputs[0]
        if (list(a['shape']) != [1, 3, 40] or a['dtype'] != np.int8 or
                list(b['shape']) != [1, 1] or b['dtype'] != np.uint8 or
                b['quantization'] != (1 / 256, 0)):
            raise ValueError('Unsupported streaming ABI or score encoding')
        scale, zero = a['quantization']
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError('Invalid input quantization')
        supplied = np.clip(np.rint(values / scale) + zero, -128, 127).astype(np.int8)
        codes = np.empty(len(supplied) // 3, np.uint8)
        for at in range(0, len(supplied), 3):
            interpreter.set_tensor(a['index'], supplied[None, at:at + 3])
            interpreter.invoke()
            codes[at // 3] = interpreter.get_tensor(b['index'])[0, 0]
        support = (3 * np.arange(1, len(codes) + 1, dtype=np.int64) + 2) * 160
        global_end = np.minimum(support, len(combined))
        source_end = global_end - prime
        trace = output / (name + '.npz')
        np.savez_compressed(trace, uint8=codes, support_unclamped=support,
                            global_sample_end=global_end, source_sample_end=source_end)
        points = []
        for threshold in grid:
            # Event state starts with inference, including priming. The app
            # ignores returns during priming without clearing detector state.
            events = [dict(step=s, global_sample=int(global_end[s - 1]),
                           source_sample=int(source_end[s - 1])) for s in event_steps(codes, threshold)]
            points.append(dict(threshold=threshold,
                               ignored_priming_events=[e for e in events if e['source_sample'] <= 0],
                               real_input_events=[e for e in events if e['source_sample'] > 0]))
        result['phases'].append(dict(name=name, prime_samples=prime, calls=len(codes),
                                     final_feature_padding=(-len(raw)) % 3,
                                     frontend_frames=len(raw), trace_sha256=sha(trace),
                                     input_quantization=[scale, zero], points=points,
                                     elapsed_seconds=time.monotonic() - started))
        (output / 'progress.json').write_text(json.dumps(result, indent=2) + '\n')
    (output / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--wav', type=Path, required=True)
    parser.add_argument('--context-frames', type=int, required=True)
    parser.add_argument('--threshold-grid', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = replay(args.model, args.wav, args.context_frames, args.threshold_grid, args.output)
    print(json.dumps(dict(model_sha256=result['model_sha256'], original_samples=result['original_samples'],
                          phases=[dict(name=p['name'], calls=p['calls']) for p in result['phases']])))
