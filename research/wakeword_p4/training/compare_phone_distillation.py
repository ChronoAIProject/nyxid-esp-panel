"""Verify a matched training-only phoneme-supervision comparison."""
import argparse
import json
import math
from pathlib import Path

from compare_paired_ranking import interface, sha


def operating_points(arms, fallback_reference=None):
    """Choose frozen-grid points; optionally compare against a bad control."""
    def rate(row):
        return max(row['max_source_false_rate'], row['max_corpus_false_rate'])

    def best(name, fraction):
        return max((r for r in arms[name]['grid'] if r['ambient_false_events'] == 0 and rate(r) <= fraction),
                   key=lambda r: (r['macro_recall'], -r['threshold']), default=None)

    strict = {name: best(name, .01) for name in arms}
    control, policy = strict['control'], 'strict_gate'
    if control is None:
        control, policy = best('control', .02), 'two_percent_gate'
    if control is None:
        control = min((r for r in arms['control']['grid'] if r['ambient_false_events'] == 0),
                      key=lambda r: (rate(r), -r['macro_recall'], r['threshold']), default=None)
        policy = 'minimum_false_zero_event' if control else 'none'
    treatment = strict['phonetic']
    if control is None and fallback_reference is not None:
        control = next((r for r in arms['control']['grid'] if r['threshold'] == fallback_reference), None)
        if control is None:
            raise ValueError('Frozen control reference is absent from the grid')
        policy = 'frozen_reference_with_false_rate_dominance'
        treatment = best('phonetic', min(.01, rate(control)))
    return {'strict_gate': strict, 'control_comparator': control,
            'control_comparator_policy': policy, 'treatment_comparator': treatment,
            'advance': bool(treatment and control and treatment['macro_recall'] >= control['macro_recall']+.03)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('plan', 'artifacts', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    a = p.parse_args()
    plan = json.loads(a.plan.read_bytes())
    steps = int(plan['common_train_args'][plan['common_train_args'].index('--steps')+1])
    fields = ('threshold', 'macro_recall', 'ambient_hours', 'ambient_false_events',
              'ambient_false_per_hour_upper_95', 'max_source_false_rate',
              'max_corpus_false_rate', 'hard_negative_false_rate')
    arms, shared = {}, {}
    for name, gain in plan['arms'].items():
        model = a.artifacts/f'model-{name}'
        val = a.artifacts/f'validation-{name}'
        train = json.loads((model/'training.json').read_bytes())
        cal = json.loads((val/'threshold.json').read_bytes())
        details = json.loads((val/'validation-details.json').read_bytes())
        digest = sha(model/'hey_nyx.tflite')
        phone = train['phone_distillation']
        warmup = phone.get('warmup')
        if plan.get('warmup'):
            if (not warmup or any(warmup[k] != v for k, v in plan['warmup'].items())
                    or warmup['binary_weights_before_sha256'] != warmup['binary_weights_after_sha256']
                    or warmup['binary_optimizer_before_sha256'] != warmup['binary_optimizer_after_sha256']):
                raise ValueError('Phone warmup differs or changes the binary model/optimizer')
        elif warmup:
            raise ValueError('Unplanned phone-head warmup')
        if (train['model_sha256'] != digest or cal['model_sha256'] != digest
                or train['initial_weights_sha256'] != plan['init']['sha256']
                or any(train['source_files_sha256'][k] != v for k, v in plan['source_sha256'].items())
                or phone['bank_sha256'] != plan['phone_bank_sha256'] or phone['draws'] != steps*8
                or phone.get('objective', 'joint') != plan.get('phone_objective', 'joint')
                or phone.get('encoder_ratio_cap') != plan.get('encoder_ratio_cap')
                or phone['gain'] != gain or train['training_prefix_batches'] != 16
                or train['paired_near_words'] is not None or train['phrase_output']
                or cal['feature_index_sha256'] != plan['feature_index_sha256']['features.json']
                or cal['smoothing_steps'] != 5 or cal['threshold_grid'] != plan['threshold_grid']
                or cal['negative_selection_metric'] != 'max_corpus_false_rate'):
            raise ValueError('Training or validation differs from the frozen phone experiment')
        if plan.get('encoder_ratio_cap') is not None:
            balance = phone['encoder_balance']
            if gain and (not balance['enabled'] or balance['steps'] != steps
                         or not 0 < balance['balanced_steps'] <= steps
                         or not math.isfinite(balance['maximum_scaled_encoder_ratio'])
                         or balance['maximum_scaled_encoder_ratio'] > plan['encoder_ratio_cap']+1e-6):
                raise ValueError('Encoder balancing is missing or exceeds its frozen bound')
            if not gain and (balance['enabled'] or balance['steps']):
                raise ValueError('Binary-only control unexpectedly applies encoder balancing')
        iface = interface(model/'hey_nyx.tflite')
        if any(op in ('SOFTMAX',) for op in iface['operators']):
            raise ValueError('The training-only head leaked into the export')
        shared[name] = {
            'prefix': train['training_prefix_sha256'],
            'schedule': phone['schedule_sha256'],
            'ptq': train['representative_training_inputs_sha256'],
            'warmup': warmup,
            'interface': iface, 'parameters': train['parameters'],
            'clips': [(r['source'], r['variant'], r['label']) for r in details['validation']],
            'ambient': [(r['source_type'], r['hours']) for r in details['ambient']],
        }
        grid = [{k: row[k] for k in fields} for row in cal['grid']]
        ref = next(r for r in grid if r['threshold'] == .97)
        false_count = sum(not r['label'] and r['max_score'] >= .97 for r in details['validation'])
        if (false_count/sum(not r['label'] for r in details['validation']) != ref['hard_negative_false_rate']
                or sum(r['events']['0.97'] for r in details['ambient']) != ref['ambient_false_events']):
            raise ValueError('Fixed-reference counts differ')
        arms[name] = {'model_sha256': digest, 'model_bytes': train['model_bytes'],
                      'training_sha256': sha(model/'training.json'), 'calibration_sha256': sha(val/'threshold.json'),
                      'details_sha256': sha(val/'validation-details.json'), 'reference_0_97': ref,
                      'reference_false_clips': false_count, 'grid': grid}
    if shared['control'] != shared['phonetic']:
        raise ValueError('Input prefix, draw schedule, calibration input, exposure or export interface differs')

    report = {'scope': plan['scope'], 'plan_sha256': sha(a.plan),
              'shared_training_prefix_sha256': shared['control']['prefix'],
              'shared_draw_schedule_sha256': shared['control']['schedule'],
              'shared_ptq_input_sha256': shared['control']['ptq'],
              'shared_warmup': shared['control']['warmup'],
              'interface': shared['control']['interface'], 'inference_parameters': shared['control']['parameters'],
              **operating_points(arms, plan.get('control_fallback_reference')),
              'arms': arms, 'limits': plan['limits']}
    with a.output.open('x') as f:
        json.dump(report, f, indent=2)
        f.write('\n')
    print(json.dumps({k: v for k, v in report.items() if k not in ('arms', 'interface')}, indent=2))
    print(json.dumps({name: r['reference_0_97'] for name, r in arms.items()}, indent=2))


if __name__ == '__main__':
    main()
