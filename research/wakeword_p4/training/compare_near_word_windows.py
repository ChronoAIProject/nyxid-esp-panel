"""Check matched exports and summarize their reused development evaluation."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from near_word_windows import NEAR_WORDS


def sha(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('plan', 'artifacts', 'output'):
        parser.add_argument('--' + key, required=True, type=Path)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_bytes())
    known = Path(__file__).resolve().parents[1] / 'results/microwake-negative-focus-v1.json'
    baseline = json.loads(known.read_bytes())['arms']['control']['model_sha256']
    arms = {}
    identity = None
    for name in plan['arms']:
        model = args.artifacts / f'model-{name}'
        val = args.artifacts / f'validation-{name}'
        training = json.loads((model / 'training.json').read_bytes())
        calibration = json.loads((val / 'threshold.json').read_bytes())
        model_hash = sha(model / 'hey_nyx.tflite')
        if (training['model_sha256'] != model_hash or calibration['model_sha256'] != model_hash
                or training['source_files_sha256']['train_stream.py'] != plan['source_sha256']['train_stream.py']
                or training['initial_weights_sha256'] != plan['init']['sha256']
                or calibration['feature_index_sha256'] != plan['feature_index_sha256']['features.json']
                or calibration['smoothing_steps'] != 5 or calibration['threshold_grid'] != plan['threshold_grid']
                or calibration['negative_selection_metric'] != 'max_corpus_false_rate'):
            raise ValueError('Model, inputs, or evaluation differs from frozen comparison')
        scheduled = training['near_word_windows']
        if name == 'control':
            if model_hash != baseline or scheduled != {'sha256': None, 'scheduled': 0, 'changed': 0}:
                raise ValueError('Control did not reproduce the previous INT8 bytes')
            identity = training['representative_training_inputs_sha256']
        elif (scheduled['sha256'] != plan['windows_sha256'] or scheduled['changed'] <= 0
                or training['representative_training_inputs_sha256'] != identity):
            raise ValueError('Treatment missed the intervention or changed representative input bytes')
        details = json.loads((val / 'validation-details.json').read_bytes())
        ref = .97
        hits, totals, misses = Counter(), Counter(), Counter()
        for row in details['validation']:
            if row['label'] == 0:
                phrase = row['text'].strip().lower().rstrip('.!?')
                totals[phrase] += 1
                hits[phrase] += row['max_score'] >= ref
                if row['max_score'] >= ref:
                    misses[f"{row['engine']}:{row['source'].split('/')[0]}"] += 1
        fields = ('threshold', 'macro_recall', 'recall', 'ambient_hours', 'ambient_false_events',
                  'ambient_false_per_hour', 'ambient_false_per_hour_upper_95', 'max_source_false_rate',
                  'max_corpus_false_rate', 'hard_negative_false_rate')
        grid = [{k: row[k] for k in fields} for row in calibration['grid']]
        ref_row = next(r for r in grid if r['threshold'] == ref)
        if (sum(r['events'][str(ref)] for r in details['ambient']) != ref_row['ambient_false_events']
                or sum(hits.values()) / sum(totals.values()) != ref_row['hard_negative_false_rate']):
            raise ValueError('Raw validation counts do not reproduce frozen reference')
        arms[name] = {'model_sha256': model_hash, 'model_bytes': (model / 'hey_nyx.tflite').stat().st_size,
                      'training_sha256': sha(model / 'training.json'), 'calibration_sha256': sha(val / 'threshold.json'),
                      'details_sha256': sha(val / 'validation-details.json'), 'windows': scheduled,
                      'reference_0_97': ref_row, 'false_views_by_engine_corpus_at_0_97': misses,
                      'near_words_at_0_97': {w: [hits[w], totals[w]] for w in sorted(NEAR_WORDS)},
                      'grid': grid}

    def best(arm, limit):
        return max((r for r in arms[arm]['grid'] if r['ambient_false_events'] == 0
                    and max(r['max_source_false_rate'], r['max_corpus_false_rate']) <= limit),
                   key=lambda r: (r['macro_recall'], -r['threshold']), default=None)

    control = best('control', .01)
    relaxed_control = best('control', .02)
    treatment = best('whole_phrase', .01)
    comparator = control or relaxed_control
    result = {'scope': plan['scope'], 'plan_sha256': sha(args.plan), 'control_reproduced_int8': True,
              'matched_representative_inputs_sha256': identity,
              'arms': arms, 'control_best_strict_zero_event': control, 'control_best_relaxed_zero_event': relaxed_control,
              'whole_phrase_best_strict_zero_event': treatment,
              'advance': bool(treatment and comparator and treatment['macro_recall'] >= comparator['macro_recall'] + .03),
              'limits': plan['limits']}
    with args.output.open('x') as out:
        out.write(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'advance': result['advance'], 'ref': {k: v['reference_0_97'] for k, v in arms.items()},
                      'near_words': {k: v['near_words_at_0_97'] for k, v in arms.items()}}, indent=2))


if __name__ == '__main__':
    main()
