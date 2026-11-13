"""Independently recheck all saved continuation events using integer sums."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from diagnose_bryce_v30_saved_scores_v1 import analyze


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('study', 'output'):
        parser.add_argument('--' + name, required=True, type=Path)
    args = parser.parse_args()
    assert not args.output.exists()
    report = json.loads((args.study / 'report.json').read_text())
    assert sha(args.study / 'scores.npz') == report['scores_sha256']
    assert sha(args.study / 'source.py') == report['source_sha256']
    assert sha(args.study / 'protocol.json') == report['protocol_sha256']
    variants = ['original', 'dry_stop_0ms', 'dry_stop_100ms', 'dry_stop_200ms']
    checked, changed_clips, prefix_score_count = 0, {}, 0
    with np.load(args.study / 'scores.npz', allow_pickle=False) as saved:
        assert set(saved.files) == {f'output_{i}__{v}' for i in range(126) for v in variants}
        for index, row in enumerate(report['records']):
            for variant in variants:
                item = row['variants'][variant]
                scores = saved[f'output_{index}__{variant}']
                reference = dict(output_count=item['output_count'],
                    quantized_output_sha256=item['quantized_output_sha256'],
                    firmware_policy=item['policy'], host_policy=item['policy'])
                result = analyze(scores, reference)
                for key in ('raw_max_uint8', 'five_score_max_sum_uint8'):
                    assert result[key] == item[key]
                if variant != 'original':
                    count = item['preservation']['causal_output_scores_exact']
                    np.testing.assert_array_equal(scores[:count], saved[f'output_{index}__original'][:count])
                    prefix_score_count += count
                checked += 1
        for variant in variants[1:]:
            changed_clips[variant] = sum(
                row['variants'][variant]['preservation']['changed_pcm_samples'] > 0
                for row in report['records'])
    output = dict(schema='bryce-same-prefix-saved-score-audit-v1',
        source_sha256=sha(Path(__file__)),
        independent_integer_event_helper_sha256=sha(Path(__file__).with_name('diagnose_bryce_v30_saved_scores_v1.py')),
        report_sha256=sha(args.study / 'report.json'), scores_sha256=report['scores_sha256'],
        traces_checked=checked, all_event_sequences_and_peaks_match=True,
        causal_score_prefix_values_rechecked=prefix_score_count,
        clips_with_changed_pcm=changed_clips, model_inference=False,
        acceptance_claim=False,
        limits='Recomputes saved score events and causal score-prefix equality only. '
               'PCM/frontend preservation is asserted by original experiment, not independently re-rendered here. '
               'No threshold selection or independent recall/FAR test.')
    args.output.write_text(json.dumps(output, indent=2) + '\n')
    print(json.dumps(output))


if __name__ == '__main__':
    main()
