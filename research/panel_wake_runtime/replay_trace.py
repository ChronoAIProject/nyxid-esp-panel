"""Compare a complete input trace with host frontend and reference inference.

Reject lost/duplicate fragments rather than filling gaps with invented audio.
The original unfused backbone and bundled exact-layout heads are explicit.
"""
import argparse
import collections
import hashlib
import json
from pathlib import Path
import sys
import zlib

import numpy as np
from ai_edge_litert.interpreter import Interpreter, OpResolverType

from trace_capture import records
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'wakeword_p4/embedding'))
from native_mel import NativeMel


def assemble(path):
    groups = collections.defaultdict(dict)
    for header, payload in records(path):
        _, instance, step, part, *_ = header
        row = groups[instance].setdefault(step, {'header': header, 'parts': {}})
        # Header metadata must agree across all fragments, except part number.
        if header[:3] + header[4:] != row['header'][:3] + row['header'][4:]:
            raise ValueError('Conflicting trace metadata')
        if part in row['parts']:
            raise ValueError('Duplicate trace fragment')
        row['parts'][part] = payload
    if not groups:
        raise ValueError('Empty trace')
    for instance, steps in groups.items():
        if sorted(steps) != list(range(1, max(steps) + 1)):
            raise ValueError('Missing initial/intermediate trace steps')
        for step, row in steps.items():
            if set(row['parts']) != {0, 1, 2}:
                raise ValueError('Missing trace fragment')
            a, b, embedding = (row['parts'][part] for part in range(3))
            pcm = a[:1280] + b[:1280]
            if zlib.crc32(pcm) != row['header'][5]:
                raise ValueError('PCM trace checksum differs')
            row['pcm'] = np.frombuffer(pcm, '<i2')
            row['features'] = np.frombuffer(a[1280:] + b[1280:], np.int8)
            row['embedding'] = np.frombuffer(embedding, np.int8)
    return groups


def interpreter(path):
    value = Interpreter(model_path=str(path), num_threads=1,
                        experimental_op_resolver_type=OpResolverType.BUILTIN_REF)
    value.allocate_tensors()
    return value


def invoke(model, array):
    source, target = model.get_input_details()[0], model.get_output_details()[0]
    model.set_tensor(source['index'], array.reshape(source['shape']))
    model.invoke()
    return model.get_tensor(target['index']).reshape(-1)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--capture', type=Path, required=True)
    p.add_argument('--backbone', type=Path, required=True)
    p.add_argument('--bundle', type=Path, required=True)
    p.add_argument('--window', type=Path, required=True)
    p.add_argument('--mel-filter', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        p.error('Use a fresh output file')
    manifest = json.loads((args.bundle / 'manifest.json').read_text())
    expected = manifest.get('backbone_layout', {}).get('source_sha256')
    if expected is None or hashlib.sha256(args.backbone.read_bytes()).hexdigest() != expected:
        raise ValueError('Explicit unfused backbone identity differs from bundle provenance')
    for name in ('dense.tflite', 'temporal.tflite', 'native-coefficients.h', 'backbone.tflite'):
        if hashlib.sha256((args.bundle / name).read_bytes()).hexdigest() != manifest['files'][name]:
            raise ValueError('Bundle file identity differs')
    frontend = NativeMel(np.load(args.window), np.load(args.mel_filter))
    report = {'scope': 'Captured development input numeric diagnostic; not recognition acceptance',
              'capture_sha256': hashlib.sha256(args.capture.read_bytes()).hexdigest(),
              'bundle_sha256': hashlib.sha256((args.bundle / 'manifest.json').read_bytes()).hexdigest(),
              'threshold': manifest['threshold'], 'instances': [], 'accepted': False}
    for instance, blocks in assemble(args.capture).items():
        model = interpreter(args.backbone)
        source = model.get_input_details()[0]
        scale, zero = source['quantization']
        heads = [interpreter(args.bundle / (name + '.tflite')) for name in ('dense', 'temporal')]
        history = np.zeros((16, 96), np.int8)
        previous = np.zeros(480, np.int16)
        scores = collections.deque(maxlen=3)
        row = {'instance': instance, 'steps': len(blocks), 'frontend_different_bytes': 0,
               'frontend_max_byte_delta': 0, 'embedding_different_bytes': 0,
               'head_different_outputs': 0, 'numerator_mismatches': 0,
               'event_mismatches': 0, 'device_events': [], 'host_events': [],
               'maximum_score': 0, 'live_samples': max(0, len(blocks) - 30) * 1280}
        live_energy, live_peak, clipped = 0, 0, 0
        armed, last = True, -1000
        for step in sorted(blocks):
            block = blocks[step]
            header = block['header']
            pcm = block['pcm']
            if step <= 30 and pcm.any():
                raise ValueError('Priming blocks are not exact zero samples')
            if step > 30:
                wide = pcm.astype(np.int64)
                live_energy += int(wide @ wide)
                live_peak = max(live_peak, int(np.abs(wide).max()))
                clipped += int(np.count_nonzero((wide == -32768) | (wide == 32767)))
            signal = np.concatenate((previous, pcm))
            previous = pcm[-480:].copy()
            computed = np.clip(np.rint(frontend.block(signal) / scale) + zero, -128, 127).astype(np.int8).reshape(-1)
            delta = np.abs(computed.astype(np.int16) - block['features'].astype(np.int16))
            row['frontend_different_bytes'] += int(np.count_nonzero(delta))
            row['frontend_max_byte_delta'] = max(row['frontend_max_byte_delta'], int(delta.max()))
            actual = invoke(model, block['features'])
            row['embedding_different_bytes'] += int(np.count_nonzero(actual != block['embedding']))
            history[:-1] = history[1:]
            history[-1] = block['embedding']
            if step >= 25:
                probabilities = [int(invoke(h, history)[0]) + 128 for h in heads]
                row['head_different_outputs'] += sum(a != b for a, b in zip(probabilities, header[7:9]))
                scores.append(sum(probabilities))
            numerator = sum(scores) if len(scores) == 3 else 0
            row['numerator_mismatches'] += numerator != header[9]
            score = numerator / 1536
            row['maximum_score'] = max(row['maximum_score'], score)
            if score < manifest['threshold'] * .5 and step - last >= 25:
                armed = True
            event = len(scores) == 3 and armed and score >= manifest['threshold'] and step - last >= 25
            if event:
                last, armed = step, False
                row['host_events'].append(step)
            device_event = bool(header[4] & 2)
            if device_event:
                row['device_events'].append(step)
            row['event_mismatches'] += event != device_event
        row['live_pcm_peak'] = live_peak
        row['live_pcm_rms'] = (live_energy / max(1, row['live_samples'])) ** .5
        row['clipped_samples'] = clipped
        row['frontend_exact'] = row['frontend_different_bytes'] == 0
        row['models_exact_on_device_inputs'] = all(row[k] == 0 for k in (
            'embedding_different_bytes', 'head_different_outputs'))
        row['policy_exact_on_device_heads'] = all(row[k] == 0 for k in (
            'numerator_mismatches', 'event_mismatches'))
        report['instances'].append(row)
        print(json.dumps(row), flush=True)
    report['frontend_exact'] = all(row['frontend_exact'] for row in report['instances'])
    report['models_exact_on_device_inputs'] = all(
        row['models_exact_on_device_inputs'] for row in report['instances'])
    report['policy_exact_on_device_heads'] = all(
        row['policy_exact_on_device_heads'] for row in report['instances'])
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    if any(any(row[k] for k in ('embedding_different_bytes', 'head_different_outputs',
                               'numerator_mismatches', 'event_mismatches')) for row in report['instances']):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
