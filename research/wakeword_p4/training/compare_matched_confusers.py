"""Verify the matched ambient-negative training comparison and its fixed gates."""
import argparse
import json
from pathlib import Path

from compare_paired_ranking import interface
from matched_confuser_windows import ARMS, INTERVAL, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('plan', 'artifacts', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_bytes())
    steps = int(plan['common_train_args'][plan['common_train_args'].index('--steps')+1])
    matched, arms = {}, {}
    fields = ('threshold', 'macro_recall', 'ambient_hours', 'ambient_false_events', 'ambient_false_per_hour_upper_95',
              'max_source_false_rate', 'max_corpus_false_rate', 'hard_negative_false_rate')
    for arm in ARMS:
        train_dir, val_dir = (args.artifacts/f'{kind}-{arm}' for kind in ('model', 'validation'))
        model, train, val = train_dir/'hey_nyx.tflite', train_dir/'training.json', val_dir/'threshold.json'
        training, calibration = (json.loads(p.read_bytes()) for p in (train, val))
        meta = training['matched_confusers']
        if (training['model_sha256'] != sha(model) or calibration['model_sha256'] != sha(model)
                or training['initial_weights_sha256'] != plan['init']['sha256']
                or any(training['source_files_sha256'][k] != v for k, v in plan['source_sha256'].items())
                or (meta['arm'], meta['bank_sha256'], meta['draws']) != (arm, plan['bank_sha256'], steps//INTERVAL)
                or training['near_word_windows']['sha256'] != plan['windows_sha256']
                or calibration['feature_index_sha256'] != plan['feature_index_sha256']['features.json']
                or calibration['threshold_grid'] != plan['threshold_grid'] or calibration['smoothing_steps'] != 5
                or calibration['negative_selection_metric'] != 'max_corpus_false_rate'):
            raise ValueError('Training or evaluation differs from the frozen plan')
        details = json.loads((val_dir/'validation-details.json').read_bytes())
        matched[arm] = {'draw_schedule': meta['schedule_sha256'], 'tts_pairing': training['paired_near_words'],
                        'ptq_input': training['representative_training_inputs_sha256'], 'interface': interface(model),
                        'validation_clips': [(r['source'], r['variant'], r['label']) for r in details['validation']],
                        'ambient_streams': [(r['source_type'], r['hours']) for r in details['ambient']]}
        grid = [{k: row[k] for k in fields} for row in calibration['grid']]
        arms[arm] = {'model_sha256': sha(model), 'training_sha256': sha(train), 'calibration_sha256': sha(val),
                     'grid': grid, 'reference_0_97': next(r for r in grid if r['threshold'] == .97)}
    if matched['ordinary'] != matched['hard']:
        raise ValueError('The pair schedule, INT8 interface, representative input or exposure differed')
    strict, closest = {}, {}
    for arm, data in arms.items():
        def false(r):
            return max(r['max_source_false_rate'], r['max_corpus_false_rate'])
        strict[arm] = max((r for r in data['grid'] if r['ambient_false_events'] == 0 and false(r) <= .01),
                          key=lambda r: r['macro_recall'], default=None)
        closest[arm] = min(data['grid'], key=lambda r: (r['ambient_false_events'], false(r), -r['macro_recall']))
    summary = {'scope': plan['scope'], 'plan_sha256': sha(args.plan), 'draws_per_arm': steps//INTERVAL,
               'pair_schedule_sha256': matched['hard']['draw_schedule'],
               'ptq_input_sha256': matched['hard']['ptq_input'], 'interface': matched['hard']['interface'],
               'strict_gate': strict, 'closest_grid_points': closest,
               'advance': bool(strict['hard'] and strict['ordinary'] and
                               strict['hard']['macro_recall'] >= strict['ordinary']['macro_recall']-.03),
               'arms': arms, 'limits': plan['limits']}
    with args.output.open('x') as f:
        json.dump(summary, f, indent=2)
        f.write('\n')
    print(json.dumps({'advance': summary['advance'], 'closest': closest,
                      'reference': {k: v['reference_0_97'] for k, v in arms.items()}}, indent=2))


if __name__ == '__main__':
    main()
