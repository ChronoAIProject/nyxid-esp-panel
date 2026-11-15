"""Describe score overlap in saved development traces; never select a threshold.

Inputs are the compact original 5ee4516c positive-coverage evidence files. No
audio, feature extraction, model inference or numerical agreement test is used.
Peak ceilings deliberately ignore refractory behavior and response timing, so
they are not actual recognized-clip counts or a deployable calibration.
"""
import argparse
from collections import Counter
import gzip
import json
from pathlib import Path

import numpy as np

MODEL = '5557ab94f485b909791dcf4167d004a2961a7146cbf3679f2db2191da09b373b'


def load_cohort(root, tag):
    prefix = '4090-causal-positive-coverage-' + tag + '-'
    condition = 'multivoice-nearword' if tag == 'primed' else 'positive-coverage'
    metadata = json.loads(gzip.decompress(
        (root / (prefix + 'hey_nyx-' + condition + '-original-v1.json.gz')).read_bytes()))
    if metadata['model']['sha256'] != MODEL:
        raise ValueError('Expected original positive-coverage model metadata')
    index = json.loads(gzip.decompress((root / (prefix + 'uint8-index-v1.json.gz')).read_bytes()))
    lookup = {(r['phase'], r['index']): r for r in index}
    with np.load(root / (prefix + 'uint8-v1.npz'), allow_pickle=False) as archive:
        raw = archive[archive.files[0]].copy()
    data = []
    for rec in metadata['records']:
        if rec['phase'] != 'synthetic':
            continue
        info = lookup[(rec['phase'], rec['index'])]
        values = raw[info['offset']:info['offset'] + info['calls']]
        if len(values) != info['calls'] or values.dtype != np.uint8:
            raise ValueError('Missing or malformed original output trace')
        cumulative = np.concatenate(([0], np.cumsum(values, dtype=np.int64)))
        sums = cumulative[5:] - cumulative[:-5]
        end = (np.arange(5, len(values) + 1) * 3 + 2) * 160 - rec['prime_samples']
        source = rec['record']
        target = source['target_id']
        positive = target == 'hey_nyx'
        allowed = end >= source['lead_samples'] if positive else end >= 0
        data.append(dict(target=target, peak=int(sums[allowed].max()), positive=positive,
                         common=target not in ('hey_nyx', 'hey_willow', 'hi_esp')))
    return data


def inspect(root):
    cohorts = {tag: load_cohort(root, tag) for tag in ('primed', 'jenny', 'joe')}
    negative = [r['peak'] for r in cohorts['jenny'] if r['common']]
    rows = []
    # Descriptive false-peak budgets, not requirements or selected calibrations.
    for budget in (0, 4, 8, 16, 48, 66):
        cutoff = sorted(negative, reverse=True)[budget] + 1
        row = dict(jenny_common_peak_budget=budget, minimum_cutoff_sum=cutoff,
                   threshold=cutoff / 1280,
                   jenny_common_peak_exceedances=sum(v >= cutoff for v in negative))
        for tag, data in cohorts.items():
            positive = [r for r in data if r['positive']]
            row[tag + '_target_peak_ceiling'] = sum(r['peak'] >= cutoff for r in positive)
            row[tag + '_target_total'] = len(positive)
        rows.append(row)
    saturated = Counter(r['target'] for r in cohorts['jenny'] if r['common'] and r['peak'] == 1275)
    joe = [r['peak'] for r in cohorts['joe'] if r['positive']]
    return dict(schema=1, model_sha256=MODEL, source_commit='5ee4516c',
        scope='Saved consumed development only; no inference, new audio, training, '
              'numerical agreement test, threshold selection or independent acceptance.',
        method='Maximum original five-output integer sum after source onset for targets '
               'and after priming for negatives. Hysteresis, refractory and timing ignored. '
               'Positive counts are optimistic peak ceilings, not recognized clips. '
               'Generated supports are not used as phonetic boundaries.',
        jenny_shared_common_clips=len(negative), jenny_saturated_common_by_phrase=dict(saturated),
        joe_target_peak_sum_min_median_max=[min(joe), float(np.median(joe)), max(joe)],
        rows=rows, deployment_threshold_changed=False, candidate_promoted=False,
        conclusion='Some ordinary nearwords saturate the five-output score, while even '
                   'the strongest Joe target peak is much lower. This points to poor '
                   'acoustic separation, rather than a small sensitivity adjustment. '
                   'It does not rule out a different temporal policy or establish its usability.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('saved_evidence', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = inspect(args.saved_evidence)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({key: result[key] for key in
        ('jenny_saturated_common_by_phrase', 'joe_target_peak_sum_min_median_max', 'conclusion')}))


if __name__ == '__main__':
    main()
