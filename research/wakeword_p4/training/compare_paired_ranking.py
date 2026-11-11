"""Verify matched schedules and report paired training development results."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from paired_training import CONFUSERS, PAIRS_PER_BATCH, phrase


def sha(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def interface(path):
    from ai_edge_litert.interpreter import Interpreter, OpResolverType

    model = Interpreter(model_path=str(path), experimental_op_resolver_type=OpResolverType.BUILTIN_REF)
    model.allocate_tensors()
    return {'inputs': [[r['shape'].tolist(), r['dtype'].__name__, list(r['quantization'])]
                       for r in model.get_input_details()],
            'outputs': [[r['shape'].tolist(), r['dtype'].__name__, list(r['quantization'])]
                        for r in model.get_output_details()],
            'operators': [r['op_name'] for r in model._get_ops_details()]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('plan', 'artifacts', 'output'):
        parser.add_argument('--' + key, required=True, type=Path)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_bytes())
    arms, schedules, ptq = {}, set(), set()
    phrase_study = set(plan['arms']) == {'phrase_control', 'phrase_aux'}
    exposure, interfaces = set(), set()
    fields = ('threshold', 'macro_recall', 'ambient_hours', 'ambient_false_events',
              'ambient_false_per_hour_upper_95', 'max_source_false_rate',
              'max_corpus_false_rate', 'hard_negative_false_rate')
    steps = int(plan['common_train_args'][plan['common_train_args'].index('--steps') + 1])
    for name, gain in plan['arms'].items():
        model = args.artifacts / f'model-{name}'
        val = args.artifacts / f'validation-{name}'
        training = json.loads((model / 'training.json').read_bytes())
        calibration = json.loads((val / 'threshold.json').read_bytes())
        digest = sha(model / 'hey_nyx.tflite')
        pairing = training['paired_near_words']
        if (training['model_sha256'] != digest or calibration['model_sha256'] != digest
                or training['source_files_sha256']['train_stream.py'] != plan['source_sha256']['train_stream.py']
                or training['source_files_sha256']['paired_training.py'] != plan['source_sha256']['paired_training.py']
                or training['initial_weights_sha256'] != plan['init']['sha256']
                or pairing['draws'] != steps * PAIRS_PER_BATCH
                or pairing['gain'] != (0 if phrase_study else gain)
                or (phrase_study and training['paired_phrase_gain'] != gain)
                or training['near_word_windows'] != {'sha256': plan['windows_sha256'], 'scheduled': 0, 'changed': 0}
                or calibration['feature_index_sha256'] != plan['feature_index_sha256']['features.json']
                or calibration['smoothing_steps'] != 5 or calibration['threshold_grid'] != plan['threshold_grid']
                or calibration['negative_selection_metric'] != 'max_corpus_false_rate'):
            raise ValueError('Model, inputs, or evaluation differ from the frozen comparison')
        schedules.add(pairing['schedule_sha256'])
        ptq.add(training['representative_training_inputs_sha256'])
        grid = [{k: r[k] for k in fields} for r in calibration['grid']]
        details = json.loads((val / 'validation-details.json').read_bytes())
        if phrase_study:
            exposure.add((tuple((r['engine'], r['source'], r['variant'], r['label'])
                                for r in details['validation']),
                          tuple((r['source_type'], r['hours']) for r in details['ambient']),
                          tuple(r['ambient_hours'] for r in grid)))
            interfaces.add(json.dumps(interface(model / 'hey_nyx.tflite'), sort_keys=True))
        ref = next(r for r in grid if r['threshold'] == .97)
        totals, false, strata = Counter(), Counter(), Counter()
        for row in details['validation']:
            if row['label'] == 0:
                text = phrase(row)
                totals[text] += 1
                if row['max_score'] >= ref['threshold']:
                    false[text] += 1
                    strata[f"{row['engine']}:{row['source'].split('/')[0]}"] += 1
        if (sum(false.values()) / sum(totals.values()) != ref['hard_negative_false_rate']
                or sum(r['events']['0.97'] for r in details['ambient']) != ref['ambient_false_events']):
            raise ValueError('Raw development counts do not reproduce fixed reference')
        arms[name] = {'model_sha256': digest, 'model_bytes': (model / 'hey_nyx.tflite').stat().st_size,
                      'training_sha256': sha(model / 'training.json'),
                      'calibration_sha256': sha(val / 'threshold.json'),
                      'details_sha256': sha(val / 'validation-details.json'),
                      'paired_training': pairing, 'grid': grid, 'fixed_reference_0_97': ref,
                      'reference_false_by_engine_corpus': strata,
                      'reference_confusers': {c: [false[c], totals[c]] for c in CONFUSERS}}
    if len(schedules) != 1 or len(ptq) != 1:
        raise ValueError('Matched training schedules or representative inputs differ')
    if phrase_study and (len(exposure) != 1 or len(interfaces) != 1):
        raise ValueError('Validation exposure or INT8 interface differs')

    def best(name, max_fraction):
        eligible = (r for r in arms[name]['grid'] if r['ambient_false_events'] == 0
                    and max(r['max_source_false_rate'], r['max_corpus_false_rate']) <= max_fraction)
        return max(eligible, key=lambda r: (r['macro_recall'], -r['threshold']), default=None)

    control_name, treatment_name = ('phrase_control', 'phrase_aux') if phrase_study else ('paired_control', 'paired_rank')
    control = best(control_name, .01)
    fallback = best(control_name, .02)
    treatment = best(treatment_name, .01)
    comparator = control or fallback
    report = {'scope': plan['scope'], 'plan_sha256': sha(args.plan), 'pair_schedule_sha256': schedules.pop(),
              'representative_training_inputs_sha256': ptq.pop(), 'arms': arms,
              'control_strict': control, 'control_relaxed': fallback,
              ('phrase_strict' if phrase_study else 'rank_strict'): treatment,
              'advance': bool(treatment and comparator and treatment['macro_recall'] >= comparator['macro_recall'] + .03),
              'limits': plan['limits']}
    if phrase_study:
        report['exported_interface'] = json.loads(interfaces.pop())
    with args.output.open('x') as out:
        out.write(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'advance': report['advance'],
                      'reference': {k: v['fixed_reference_0_97'] for k, v in arms.items()},
                      'confusers': {k: v['reference_confusers'] for k, v in arms.items()}}, indent=2))


if __name__ == '__main__':
    main()
