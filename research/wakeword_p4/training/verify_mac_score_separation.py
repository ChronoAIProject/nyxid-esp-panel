"""Verify the integer separation certificate with alternate window accounting."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('study', 'analysis', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    result = json.loads((args.analysis/'result.json').read_bytes())
    plan = json.loads((args.analysis/'plan.json').read_bytes())
    assert sha(args.study/'result.json') == result['input_result_sha256'] == plan['full_result_sha256']
    assert sha(args.study/'scores.npz') == result['input_scores_sha256'] == plan['scores_sha256']
    assert sha(args.analysis/'plan.json') == result['plan_sha256']
    assert sha(args.analysis/'source.py') == result['source_sha256']
    parent = json.loads((args.study/'result.json').read_bytes())
    traces = np.load(args.study/'scores.npz', allow_pickle=False)
    assert len(parent['source_details']) == len(result['source_scores']) == 350
    checked, witnesses = [], []
    for index, (source, saved) in enumerate(zip(parent['source_details'], result['source_scores'], strict=True)):
        assert all(saved[k] == source[k] for k in ('path','target','voice','variant','text','sha256'))
        raw = traces[f'output_{index}']
        assert raw.dtype == np.uint8 and raw.ndim == 1
        assert hashlib.sha256(raw.tobytes()).hexdigest() == source['quantized_output_sha256']
        windows = np.lib.stride_tricks.sliding_window_view(raw, 5)
        sums = windows.sum(axis=1, dtype=np.int64)
        # Obtain return positions by first constructing *all raw-score* block
        # ends, then select the end of each five-score window.
        raw_ends = np.arange(2, len(raw)+2, dtype=np.int64)*480-56160
        positions = raw_ends[4:]
        assert np.all(sums[positions <= 0] == 0)
        valid = np.flatnonzero(positions > 0)
        top = valid[np.argmax(sums[valid])]
        maximum = int(sums[top])
        assert maximum == saved['max_five_score_sum'] and saved['denominator'] == 1280
        assert (maximum >= 1175) == bool(source['firmware_arithmetic']['events'])
        checked.append((source, maximum))
        witnesses.append({'path': source['path'], 'target': source['target'],
            'sha256': source['sha256'], 'five_output_bytes': windows[top].tolist(),
            'five_score_sum': maximum, 'denominator': 1280,
            'window_start_raw_score_index_zero_based': int(top),
            'source_relative_detector_block_end': int(positions[top])})
    positive = [(row, score) for row, score in checked if row['target'] == 'hey_nyx']
    negative = [(row, score) for row, score in checked
                if row['target'] == 'none' and row['text'].startswith('Hey ')]
    assert len(positive) == 50 and len(negative) == 90
    max_positive = max(score for _, score in positive)
    max_negative = max(score for _, score in negative)
    cap = sum(score > max_negative for _, score in positive)
    nonzero = sum(score > 0 for _, score in positive)
    cert = result['score_separation']
    assert cap == cert['positive_upper_bound_at_zero_confuser_false_clips']
    assert max_negative == cert['highest_confuser_five_score_sum']
    assert nonzero == cert['positive_sources_with_nonzero_score']
    worst_paths = {row['path'] for row, score in negative if score == max_negative}
    best_paths = {row['path'] for row, score in positive if score == max_positive}
    audit = {'scope': __doc__, 'passed': True, 'accepted': False,
        'analysis_sha256': sha(args.analysis/'result.json'),
        'auditor_sha256': sha(Path(__file__)), 'sources_checked': len(checked),
        'highest_positive_sum': max_positive, 'highest_confuser_sum': max_negative,
        'denominator': 1280, 'zero_confuser_false_clips_positive_upper_bound': cap,
        'nonzero_positive_upper_bound': nonzero,
        'proof': 'At any positive scalar threshold, excluding every confuser requires exceeding its largest observed score; only strictly higher positive maxima could then trigger. Zero-output positives cannot trigger. Zero-score prime leaves the first real-source event armed.',
        'strongest_positive_witnesses': [row for row in witnesses if row['path'] in best_paths],
        'strongest_confuser_witnesses': [row for row in witnesses if row['path'] in worst_paths],
        'limits': 'Exact certificate for these saved INT8 traces and synthesis-intent labels. No new inference, float-checkpoint comparison, phonetic truth, population FAR/recall, physical latency or active-call acceptance.'}
    with args.output.open('x') as out:
        out.write(json.dumps(audit, indent=2)+'\n')
    print(json.dumps(audit))


if __name__ == '__main__':
    main()
