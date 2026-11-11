"""Training-only diagnostic of exact-window versus keyword-occurrence CTC.

No fitting, threshold selection, held-out inference, audio device or firmware.
Retain logits so the probability/accounting can be independently reconstructed.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np

from compact_sequence_inference import Detector, target_log_probability
from detector_policy import smooth
from keyword_occurrence import log_probability
from phone_sequences import collapse
from phrase_ctc_readout import sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('plan', 'model', 'features', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Use a fresh diagnostic output')
    plan = json.loads(args.plan.read_bytes())
    if any(sha(Path(__file__).parent/name) != value for name, value in plan['source_sha256'].items()):
        raise ValueError('Diagnostic source differs')
    if sha(args.model) != plan['model_sha256']:
        raise ValueError('Dual model differs')
    if (sha(args.features/'features.json') != plan['feature_index_sha256']
            or sha(args.features/'features.bin') != plan['feature_data_sha256']):
        raise ValueError('Actual feature index/data differs')
    rows = json.loads((args.features/'features.json').read_bytes())
    by_key = {(row['path'], row['variant']): row for row in rows}
    reserved = {row['sha256'] for row in rows if row['split'] != 'training'}
    selected = []
    for item in plan['sources']:
        row = by_key[item['path'], item['variant']]
        if (row['split'] != 'training' or row['sha256'] in reserved
                or any(row[k] != item[k] for k in ('engine', 'sha256', 'label', 'text'))):
            raise ValueError('Training source identity/role differs')
        selected.append((item, row))
    args.output.mkdir(exist_ok=False)
    (args.output/'plan.json').write_bytes(args.plan.read_bytes())
    raw = np.memmap(args.features/'features.bin', mode='r', dtype='<u2').reshape(-1, 40)
    detector = Detector(args.model)
    reports, grouped = [], defaultdict(list)
    for number, (item, row) in enumerate(selected):
        detector.reset()
        p, logits = detector.raw_outputs(raw[row['offset']:row['offset']+row['frames']])
        exact = np.exp(target_log_probability(logits))
        occurrence = np.exp(log_probability(logits))
        if (not np.isfinite(occurrence).all() or np.any(occurrence < exact-1e-12)
                or np.any(occurrence > 1) or np.any(occurrence < 0)):
            raise ValueError('CTC event probability bounds/dominance fail')
        scores = {name: smooth(value, 5) for name, value in
            [('scalar', p), ('exact_product', p*exact), ('occurrence_product', p*occurrence)]}
        times = (3*(np.arange(len(scores['scalar']))+4)+5)*.01
        end = (row['keyword_frame']+2)*.01 if row['label'] else None
        valid = ((times >= end-.1) & (times <= end+.8) if end is not None
                 else np.ones(len(times), bool))
        if not valid.any():
            raise ValueError('Selected source has no diagnostic endpoint window')
        peak = np.flatnonzero(valid)[int(np.argmax(scores['scalar'][valid]))]
        tokens = collapse(logits[peak+4].argmax(-1))
        contains = any(tokens[i:i+2] == [1, 2] for i in range(len(tokens)-1))
        trace = args.output/f'trace-{number:03d}.npz'
        np.savez_compressed(trace, scalar=p, logits=logits, exact=exact, occurrence=occurrence,
                            times=times, **{name+'_smoothed': value for name, value in scores.items()})
        report = {'source': item, 'steps': len(p), 'keyword_end': end,
            'maxima': {name: float(value[valid].max()) for name, value in scores.items()},
            'at_scalar_peak': {'time': float(times[peak]), 'greedy_tokens': tokens,
                'greedy_exact': tokens == [1, 2], 'greedy_contains': contains,
                'exact': float(exact[peak+4]), 'occurrence': float(occurrence[peak+4])},
            'array': trace.name, 'array_sha256': sha(trace)}
        reports.append(report); grouped[item['cohort']].append(report)
        print(json.dumps({'source': number+1, 'total': len(selected), 'cohort': item['cohort'],
                          'maxima': report['maxima']}), flush=True)
    summary = {}
    for cohort, values in grouped.items():
        summary[cohort] = {'sources': len(values),
            'median_maxima': {name: float(np.median([r['maxima'][name] for r in values])) for name in scores},
            'at_scalar_peak_greedy_exact': sum(r['at_scalar_peak']['greedy_exact'] for r in values),
            'at_scalar_peak_greedy_contains': sum(r['at_scalar_peak']['greedy_contains'] for r in values),
            'reference_0_97_views': {name: sum(r['maxima'][name] >= .97 for r in values) for name in scores}}
    neutral = np.zeros((1, 67, 6))
    result = {'plan_sha256': sha(args.plan), 'model_sha256': sha(args.model), 'accepted': False,
        'source_count': len(selected), 'sources': reports, 'cohorts': summary,
        'neutral_67_step_logits': {'exact': float(np.exp(target_log_probability(neutral))[0]),
                                 'occurrence': float(np.exp(log_probability(neutral))[0])},
        'limits': plan['limits']}
    (args.output/'result.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({'phase': 'complete', 'cohorts': summary,
                      'neutral_67_step_logits': result['neutral_67_step_logits']}), flush=True)


if __name__ == '__main__':
    main()
