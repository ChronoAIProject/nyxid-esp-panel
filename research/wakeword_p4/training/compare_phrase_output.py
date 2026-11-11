"""Verify the matched scalar phrase-output experiment and its development gate."""
import argparse
from collections import Counter
import json
from pathlib import Path

from compare_paired_ranking import interface, sha
from paired_training import PAIRS_PER_BATCH


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('plan', 'artifacts', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    a = p.parse_args()
    plan = json.loads(a.plan.read_bytes())
    steps = int(plan['common_train_args'][plan['common_train_args'].index('--steps')+1])
    expected_windows = {'sha256': plan['windows_sha256'], 'scheduled': 0, 'changed': 0}
    shared, arms = {}, {}
    fields = ('threshold', 'macro_recall', 'ambient_hours', 'ambient_false_events',
              'ambient_false_per_hour_upper_95', 'max_source_false_rate',
              'max_corpus_false_rate', 'hard_negative_false_rate')
    for name in ('binary', 'competing'):
        root = a.artifacts/f'model-{name}'
        val = a.artifacts/f'validation-{name}'
        training = json.loads((root/'training.json').read_bytes())
        calibration = json.loads((val/'threshold.json').read_bytes())
        details = json.loads((val/'validation-details.json').read_bytes())
        digest = sha(root/'hey_nyx.tflite')
        if (training['model_sha256'] != digest or calibration['model_sha256'] != digest
                or training['initial_weights_sha256'] != plan['init']['sha256']
                or any(training['source_files_sha256'][k] != v for k, v in plan['source_sha256'].items())
                or training['paired_near_words']['draws'] != steps*PAIRS_PER_BATCH
                or training['paired_near_words']['gain'] != 0
                or training['training_prefix_batches'] != min(steps, 16)
                or training['near_word_windows'] != expected_windows
                or training['phrase_output'] != (name == 'competing')
                or training['paired_phrase_gain'] != (0 if name == 'binary' else None)
                or calibration['feature_index_sha256'] != plan['feature_index_sha256']['features.json']
                or calibration['smoothing_steps'] != 5 or calibration['threshold_grid'] != plan['threshold_grid']
                or calibration['negative_selection_metric'] != 'max_corpus_false_rate'):
            raise ValueError('Training, source labels or validation differs from the frozen plan')
        iface = interface(root/'hey_nyx.tflite')
        if (len(iface['inputs']) != 1 or len(iface['outputs']) != 1
                or iface['inputs'][0][:2] != [[1, 3, 40], 'int8']
                or iface['outputs'][0][:2] != [[1, 1], 'uint8']):
            raise ValueError('Expected unchanged scalar INT8 streaming input/output')
        error = training['initial_target_probability_max_error']
        if name == 'competing' and (error is None or not 0 <= error <= 1e-5):
            raise ValueError('Competing head did not preserve initial target probability')
        shared[name] = {
            'pairing': training['paired_near_words'],
            'prefix': training['training_prefix_sha256'],
            'representative': training['representative_training_inputs_sha256'],
            'input': iface['inputs'],
            'output_geometry': [r[:2] for r in iface['outputs']],
            'clips': [(r['engine'], r['source'], r['variant'], r['label']) for r in details['validation']],
            'ambient': [(r['source_type'], r['hours']) for r in details['ambient']],
        }
        grid = [{k: r[k] for k in fields} for r in calibration['grid']]
        ref = next(r for r in grid if r['threshold'] == .97)
        false = Counter(r['text'].strip().lower().rstrip('.!?') for r in details['validation']
                        if not r['label'] and r['max_score'] >= .97)
        if (sum(false.values())/sum(not r['label'] for r in details['validation']) != ref['hard_negative_false_rate']
                or sum(r['events']['0.97'] for r in details['ambient']) != ref['ambient_false_events']):
            raise ValueError('Fixed reference counts do not reproduce')
        arms[name] = {
            'model_sha256': digest, 'model_bytes': training['model_bytes'],
            'training_sha256': sha(root/'training.json'), 'calibration_sha256': sha(val/'threshold.json'),
            'details_sha256': sha(val/'validation-details.json'),
            'interface': iface, 'grid': grid, 'reference_0_97': ref,
            'reference_false_by_text': dict(false),
            'initial_target_probability_max_error': error,
        }
    if shared['binary'] != shared['competing']:
        raise ValueError('Input schedule, class labels, representative inputs, IO or exposure differ')

    def false_rate(row):
        return max(row['max_source_false_rate'], row['max_corpus_false_rate'])

    def best(name, fraction):
        return max((r for r in arms[name]['grid'] if r['ambient_false_events'] == 0
                    and false_rate(r) <= fraction), key=lambda r: (r['macro_recall'], -r['threshold']), default=None)

    strict = {name: best(name, .01) for name in arms}
    control = strict['binary'] or best('binary', .02)
    control_rule = 'strict' if strict['binary'] else 'relaxed_2_percent'
    if control is None:
        control = min((r for r in arms['binary']['grid'] if r['ambient_false_events'] == 0),
                      key=lambda r: (false_rate(r), -r['macro_recall'], r['threshold']), default=None)
        control_rule = 'minimum_error_zero_event'
    treatment = strict['competing']
    report = {
        'scope': plan['scope'], 'plan_sha256': sha(a.plan),
        'shared_training_prefix_sha256': shared['binary']['prefix'],
        'shared_pair_schedule_sha256': shared['binary']['pairing']['schedule_sha256'],
        'shared_representative_inputs_sha256': shared['binary']['representative'],
        'strict_gate': strict, 'control_comparator': control, 'control_comparator_rule': control_rule,
        'advance': bool(treatment and control and treatment['macro_recall'] >= control['macro_recall']+.03),
        'arms': arms, 'limits': plan['limits'],
    }
    with a.output.open('x') as f:
        json.dump(report, f, indent=2)
        f.write('\n')
    print(json.dumps({k: report[k] for k in ('advance', 'strict_gate', 'control_comparator_rule', 'control_comparator')}, indent=2))
    print(json.dumps({name: r['reference_0_97'] for name, r in arms.items()}, indent=2))


if __name__ == '__main__':
    main()
