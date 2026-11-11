"""Replay one frozen INT8 model and inspect information lost at its probability output."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np
from ai_edge_litert.interpreter import Interpreter, OpResolverType
from scipy.special import expit

from detector_policy import smooth

POLICIES = ('uint8_probability', 'reconstructed_probability', 'logit')
CEILING = 255/256


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def predict(model, data):
    i = Interpreter(model_path=str(model), experimental_preserve_all_tensors=True,
                    experimental_op_resolver_type=OpResolverType.BUILTIN_REF, num_threads=1)
    i.allocate_tensors()
    inputs, outputs = i.get_input_details()[0], i.get_output_details()[0]
    ops = [op for op in i._get_ops_details() if op['op_name'] == 'LOGISTIC']
    if len(ops) != 1 or len(ops[0]['inputs']) != 1:
        raise ValueError('Expected one binary classifier logit')
    tensor = next(t for t in i.get_tensor_details() if t['index'] == ops[0]['inputs'][0])
    if tensor['dtype'] != np.int8 or tensor['shape'].tolist() != [1, 1] or outputs['quantization'] != (1/256, 0):
        raise ValueError('Unexpected logit/probability geometry')
    scale, zero = inputs['quantization']
    x = np.clip(np.rint(np.clip(data.astype(np.float32)/25.6, 0, 26)/scale)+zero, -128, 127).astype(np.int8)
    probabilities, logits = np.empty(len(x)//3, np.uint8), np.empty(len(x)//3, np.int8)
    for step in range(len(logits)):
        i.set_tensor(inputs['index'], x[None, step*3:step*3+3])
        i.invoke()
        probabilities[step] = i.get_tensor(outputs['index'])[0, 0]
        logits[step] = i.get_tensor(tensor['index'])[0, 0]
    scale, zero = tensor['quantization']
    dequantized = (logits.astype(np.float64)-zero)*scale
    return (smooth(probabilities.astype(np.float32)/256, 5), smooth(expit(dequantized), 5),
            smooth(dequantized, 5)), {'scale': scale, 'zero': zero}


def gate(clips, ambient):
    positive = defaultdict(list)
    negative = defaultdict(list)
    for row, value in clips:
        engine = row['engine']
        if row['label']:
            positive[engine].append(value)
        else:
            negative[(engine,)].append(value)
            negative[(engine, row['path'].split('/')[0])].append(value)
    # For >= threshold, the (allowance+1)-th highest negative must be strictly
    # below threshold. Zero continuous events likewise requires > every score.
    limiting = max([(max(ambient), 'continuous')] +
                   [(sorted(values, reverse=True)[len(values)//100], ':'.join(group))
                    for group, values in negative.items()])
    threshold = float(np.nextafter(limiting[0], np.inf))
    total = sum(len(v) for v in positive.values())
    hits = sum(sum(x >= threshold for x in v) for v in positive.values())
    macro = float(np.mean([np.mean(np.asarray(v) >= threshold) for v in positive.values()]))
    worst = max(sum(x >= threshold for x in v)/len(v) for v in negative.values())
    return {'min_threshold': threshold, 'limiting_group': limiting[1], 'positive_hits': hits,
            'positive_variants': total, 'macro_recall': macro, 'max_negative_group_rate': worst,
            'ambient_maximum': max(ambient)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('model', 'features', 'validation', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    expected = json.loads((args.validation/'threshold.json').read_bytes())
    index = args.features/'features.json'
    if (sha(args.model) != expected['model_sha256'] or sha(index) != expected['feature_index_sha256']
            or expected['smoothing_steps'] != 5):
        raise ValueError('Frozen model/feature/validation identity mismatch')
    details = json.loads((args.validation/'validation-details.json').read_bytes())
    raw = np.memmap(args.features/'features.bin', mode='r', dtype='<u2').reshape(-1, 40)
    rows = [r for r in json.loads(index.read_bytes()) if r['split'] == 'validation']
    if len(rows) != len(details['validation']):
        raise ValueError('Validation stream count mismatch')
    clips, ambient = [[] for _ in POLICIES], [[] for _ in POLICIES]
    ceiling_counts = defaultdict(lambda: [0, 0])
    for n, (row, ref) in enumerate(zip(rows, details['validation']), 1):
        if (row['path'], row['label'], row['variant']) != (ref['source'], ref['label'], ref['variant']):
            raise ValueError('Validation stream order mismatch')
        tracks, quantization = predict(args.model, raw[row['offset']:row['offset']+row['frames']])
        if not np.array_equal(tracks[0], ref['scores']):
            raise ValueError('Output bytes do not reproduce the frozen validation')
        times = (3*(np.arange(len(tracks[0]))+4)+5)*.01
        end = ref['keyword_end']
        mask = (times >= end-.10) & (times <= end+.80) if row['label'] else np.ones(len(times), dtype=bool)
        for dst, values in zip(clips, tracks):
            dst.append((row, float(values[mask].max()) if mask.any() else float('-inf')))
        if clips[0][-1][1] >= CEILING:
            group = row['engine']+':'+row['path'].split('/')[0]
            ceiling_counts[group][row['label']] += 1
        if n % 1000 == 0:
            print(json.dumps({'scored_clips': n, 'total': len(rows)}), flush=True)
    ambient_rows = json.loads((args.features/'human.json').read_bytes()) + json.loads((args.features/'ambient.json').read_bytes())
    for n, row in enumerate(ambient_rows):
        if row['split'] != 'validation':
            continue
        tracks, _ = predict(args.model, np.load(args.features/row['path'], mmap_mode='r'))
        ref = np.load(args.validation/f'ambient-validation-{n:03d}.npy')
        if not np.array_equal(tracks[0], ref):
            raise ValueError('Continuous output bytes differ')
        for dst, values in zip(ambient, tracks):
            dst.append(float(values.max()))
    result = {'scope': 'Previously scored validation only; retrospective best threshold, no new operating point selected',
              'model_sha256': expected['model_sha256'], 'probability_ceiling': CEILING,
              'logit_tensor_quantization': quantization, 'count_order': ['negative', 'positive'],
              'at_probability_ceiling_by_engine_corpus': ceiling_counts,
              'strict_gate_best': {key: gate(c, a) for key, c, a in zip(POLICIES, clips, ambient)},
              'limits': 'Internal logits are still INT8. Reconstructed probability is a sigmoid of dequantized logits, '
                        'not float-model inference. No held-out test, board firmware or audio changed.'}
    with args.output.open('x') as f:
        json.dump(result, f, indent=2)
        f.write('\n')
    print(json.dumps(result['strict_gate_best'], indent=2))


if __name__ == '__main__':
    main()
