"""Read-only input, checkpoint, logits and metric audit for the tap experiment."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

from audit_compact_study import require, sha
from audit_intent_phone_fit import classification, distance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('artifacts', 'repository', 'features', 'bank', 'initial-weights', 'parent-study', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--audit-correction', type=Path)
    args = parser.parse_args()
    root = args.artifacts.resolve()
    plan = json.loads((root/'plan.json').read_bytes())
    launch = json.loads((root/'launch.json').read_bytes())
    result = json.loads((root/'study/result.json').read_bytes())
    committed = subprocess.check_output(['git', 'show', launch['checkout_revision']+':'+
        plan.get('repository_plan_path', 'research/wakeword_p4/training/plans/intent-phone-local-tap-v1.json')], cwd=args.repository)
    require(hashlib.sha256(committed).hexdigest() == sha(root/'plan.json') == launch['plan_sha256']
            == sha(root/'study/plan.json') == result['plan_sha256'], 'Frozen plans differ')
    for name, digest in plan['source_sha256'].items():
        require(sha(root/'source'/name) == launch['source_sha256'][name] == digest, 'Source differs: '+name)
    for name, digest in plan['audit_source_sha256'].items():
        require(sha(root/'auditing'/name) == digest, 'Auditor differs: '+name)
    correction = None
    if args.audit_correction:
        correction = json.loads(args.audit_correction.read_bytes())
        require(correction['original_auditor_sha256'] == plan['audit_source_sha256'][Path(__file__).name]
                and correction['corrected_auditor_sha256'] == sha(Path(__file__))
                and correction['metric_roundoff_tolerance'] == 1e-12
                and correction['plan_sha256'] == sha(root/'plan.json'), 'Audit correction differs')
    else:
        require(sha(Path(__file__)) == plan['audit_source_sha256'][Path(__file__).name],
                'A changed auditor requires explicit correction provenance')
    for name, digest in plan['parent_study_sha256'].items():
        require(sha(args.parent_study/name) == digest, 'Parent evidence differs: '+name)
    command = launch['command']
    require(Path(command[1]).resolve() == root/'source'/plan.get('training_script', 'intent_phone_taps.py'),
            'Wrong execution source')
    require(command[2::2] == ['--plan', '--upstream', '--features', '--bank', '--initial-weights',
                            '--parent-study', '--output'], 'Wrong execution options')
    for option, path in (('--plan', root/'plan.json'), ('--features', args.features),
                        ('--bank', args.bank), ('--initial-weights', args.initial_weights),
                        ('--parent-study', args.parent_study), ('--output', root/'study')):
        require(Path(command[command.index(option)+1]).resolve() == path.resolve(), 'Wrong input: '+option)
    upstream = Path(command[command.index('--upstream')+1])
    require(subprocess.check_output(['git', '-C', str(upstream), 'rev-parse', 'HEAD'], text=True).strip()
            == plan['upstream_commit'] and not subprocess.check_output(['git', '-C', str(upstream), 'diff', 'HEAD']),
            'Upstream differs')
    require(sha(args.initial_weights) == plan['initializer_sha256']
            and sha(args.features/'features.bin') == plan['feature_data_sha256'], 'Input bytes differ')
    sys.path[:0] = [str(root/'source'), str(upstream)]
    import tensorflow as tf
    tf.config.threading.set_intra_op_parallelism_threads(3)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    from transfer import build_model
    from intent_phone_taps import encoder_tap
    from phrase_ctc_readout import load_bank, weights_sha
    from ctc_reference import loss_and_gradient
    raw, original_labels, original_rows = load_bank(args.bank, args.features, plan['compact_bank'])
    groups, rows, chosen = {}, [], []
    for i, (value, label, row) in enumerate(zip(raw, original_labels, original_rows, strict=True)):
        digest = hashlib.sha256(value.astype('<u2', copy=False).tobytes()).hexdigest()
        if digest not in groups:
            groups[digest] = len(rows)
            chosen.append(i)
            rows.append({**row, 'class': int(label), 'crop_sha256': digest, 'bank_rows': []})
        item = rows[groups[digest]]
        require(all(item[k] == row[k] for k in ('engine', 'sha256', 'variant'))
                and item['class'] == int(label), 'Conflicting duplicate identity')
        item['bank_rows'].append(i)
    require(rows == json.loads((root/'study/rows.json').read_bytes())
            == json.loads((args.parent_study/'study/rows.json').read_bytes())
            and len(rows) == plan['unique_crops'] and sha(root/'study/rows.json') == result['rows_sha256'],
            'Exact training crops differ')
    labels = original_labels[chosen]
    x = np.clip(raw[chosen].astype(np.float32)/25.6, 0, 26)
    ids, rng, schedule = np.arange(len(rows), dtype=np.int32), np.random.default_rng(plan['draw_seed']), hashlib.sha256()
    for _ in range(plan['steps']):
        schedule.update(rng.choice(ids, size=plan['batch_size'], replace=True).astype('<i4').tobytes())
    references = [[1, 2, 3, 4, 5, 6], [1, 2, 7, 4, 5, 6], [1, 2, 3, 8, 5, 6, 9], [1, 2, 3, 4, 5]]
    reports = {}
    settings = plan.get('arm_settings', {'deep': {'tap': 'deep', 'initialization': 'glorot'},
                                         'local': {'tap': 'local', 'initialization': 'glorot'}})
    for kind, setting in settings.items():
        info = result['arms'][kind]
        path = root/'study'/f'{kind}.json'
        require(info['result'] == path.name and sha(path) == info['result_sha256'], 'Result bytes differ')
        arm = json.loads(path.read_bytes())
        require(arm['draw_schedule_sha256'] == schedule.hexdigest()
                and [h['step'] for h in arm['history']] == [250, 500, 750, 1000], 'Training schedule differs')
        require(sha(root/'study'/arm['checkpoint']) == arm['checkpoint_sha256']
                and sha(root/'study'/arm['logits']) == arm['logits_sha256'], 'Checkpoint/logit bytes differ')
        retained = np.load(root/'study'/arm['logits'], allow_pickle=False)['logits']
        require(retained.shape == (len(rows), 67, 10) and np.isfinite(retained).all(), 'Invalid logits')
        tf.keras.utils.set_random_seed(726)
        binary, _ = build_model(335)
        binary.load_weights(args.initial_weights)
        require(weights_sha(binary.weights) == arm['binary_weights_before_sha256'], 'Initial binary tensors differ')
        encoder = encoder_tap(binary, setting['tap'])
        require(weights_sha(encoder.weights) == arm['encoder_weights_before_sha256'], 'Initial encoder differs')
        trainable_ids = {id(v) for v in encoder.trainable_variables}
        disconnected = [v for v in binary.weights if id(v) not in trainable_ids]
        require(weights_sha(disconnected) == arm['disconnected_weights_before_sha256'], 'Disconnected initial weights differ')
        h = tf.keras.Input((67, 60))
        require(setting['initialization'] in ('glorot', 'zero'), 'Unknown head initialization')
        initialization = (tf.keras.initializers.GlorotUniform(seed=plan['head_seed'])
                          if setting['initialization'] == 'glorot' else tf.keras.initializers.Zeros())
        head = tf.keras.Model(h, tf.keras.layers.Dense(10, kernel_initializer=initialization)(h))
        require(weights_sha(head.weights) == arm['head_weights_before_sha256'], 'Initial head differs')
        if plan.get('initial_diagnostic_crop_ids'):
            diagnostic = arm['initial_diagnostic']
            require(diagnostic == info['initial_diagnostic']
                    and sha(root/'study'/diagnostic['array']) == diagnostic['array_sha256'], 'Initial diagnostic differs')
            with np.load(root/'study'/diagnostic['array'], allow_pickle=False) as initial:
                initial_ids, initial_states, initial_logits = initial['ids'], initial['states'], initial['logits']
            require(initial_ids.tolist() == plan['initial_diagnostic_crop_ids']
                    and np.array_equal(encoder(x[initial_ids], training=False), initial_states)
                    and np.array_equal(head(initial_states, training=False), initial_logits),
                    'Actual initial representation/logits do not replay')
            probability = np.asarray(tf.nn.softmax(initial_logits, axis=-1))
            reconstructed = {'array': diagnostic['array'], 'array_sha256': diagnostic['array_sha256'],
                'minimum_logit': float(initial_logits.min()), 'maximum_logit': float(initial_logits.max()),
                'mean_entropy': float(-np.mean(np.sum(probability*np.log(np.maximum(probability, 1e-30)), axis=-1))),
                'mean_maximum_probability': float(probability.max(-1).mean()),
                'argmax_counts': np.bincount(initial_logits.argmax(-1).ravel(), minlength=10).tolist()}
            require(reconstructed == diagnostic, 'Initial diagnostic metric reconstruction differs')
            if setting['initialization'] == 'zero':
                require(np.count_nonzero(initial_logits) == 0, 'Zero head is not neutral')
        full = tf.keras.Model(encoder.input, head(encoder.output))
        full.load_weights(root/'study'/arm['checkpoint'])
        require(weights_sha(encoder.weights) == arm['encoder_weights_after_sha256']
                and weights_sha(head.weights) == arm['head_weights_after_sha256']
                and weights_sha(disconnected) == arm['disconnected_weights_before_sha256']
                and arm['disconnected_weights_unchanged'], 'Loaded tensors or frozen scope differ')
        require(full.count_params() == arm['parameters'] == info['parameters']
                and encoder.count_params() == arm['encoder_parameters'] and head.count_params() == arm['head_parameters'] == 610,
                'Parameter accounting differs')
        score_replay_delta = 0.
        for start in range(0, len(rows), plan['batch_size']):
            actual = np.asarray(full(x[start:start+plan['batch_size']], training=False))
            require(np.array_equal(actual, retained[start:start+len(actual)]), 'Checkpoint logit replay differs')
            target_rows = np.zeros((4, 7), np.int32)
            for j, reference in enumerate(references):
                target_rows[j, :len(reference)] = reference
            repeated = np.repeat(actual, 4, axis=0)
            losses = tf.nn.ctc_loss(np.tile(target_rows, (len(actual), 1)), repeated,
                np.tile([len(v) for v in references], len(actual)), [67]*len(repeated),
                logits_time_major=False, blank_index=0)
            replayed = -np.asarray(losses).reshape(len(actual), 4)
            stored = np.asarray(arm['scores'][start:start+len(actual)])
            require(np.array_equal(replayed, stored), 'Full class likelihood replay differs')
            score_replay_delta = max(score_replay_delta, float(np.max(np.abs(replayed-stored))))
        greedy = [[int(v) for j, v in enumerate(row) if v and (j == 0 or v != row[j-1])]
                  for row in retained.argmax(-1)]
        require(greedy == arm['decoded'], 'Greedy sequences differ')
        scores = np.asarray(arm['scores'])
        require(scores.shape == (len(rows), 4) and np.isfinite(scores).all(), 'Invalid class scores')
        checked, seen, max_delta = [], set(), 0.
        for i, row in enumerate(rows):
            key = row['engine'], row['class']
            if key in seen:
                continue
            seen.add(key)
            for j, reference in enumerate(references):
                expected, _ = loss_and_gradient(retained[i], reference)
                delta = abs(float(scores[i, j])+expected)
                require(delta <= plan['reference_loss_tolerance'], 'Independent CTC likelihood differs')
                max_delta = max(max_delta, delta)
            checked.append(i)
        require(checked == plan['reference_crop_ids'], 'Reference sample coverage differs')
        measured = classification(labels, scores.argmax(-1))
        lengths = np.array([len(references[label]) for label in labels])
        total_loss = sum(float(np.sum(-scores[start:start+plan['batch_size'], :]
            [np.arange(min(plan['batch_size'], len(rows)-start)), labels[start:start+plan['batch_size']]]
            /lengths[start:start+plan['batch_size']])) for start in range(0, len(rows), plan['batch_size']))
        measured.update(mean_ctc_nll_per_token=total_loss/len(rows),
            greedy_exact_sequences=sum(d == references[label] for d, label in zip(greedy, labels, strict=True)),
            target_tokens=int(lengths.sum()),
            greedy_edits=sum(distance(references[label], d) for d, label in zip(greedy, labels, strict=True)))
        measured['greedy_token_error_rate'] = measured['greedy_edits']/measured['target_tokens']
        metric_deltas = {}
        for key, value in measured.items():
            stored = arm['summary'][key]
            if isinstance(value, float):
                delta = abs(value-stored)
                require(delta <= (1e-12 if correction else plan.get('metric_roundoff_tolerance', 0)),
                        'Metric reconstruction differs: '+key)
                metric_deltas[key] = delta
            else:
                require(value == stored, 'Metric reconstruction differs: '+key)
        require(arm['summary'] == info['summary'], 'Result metric reference differs')
        gate = plan['fit_gate']
        passed = (measured['macro_accuracy'] >= gate['minimum_macro_accuracy']
            and measured['nyx_recall'] >= gate['minimum_nyx_recall']
            and measured['confuser_false_nyx_fraction'] <= gate['maximum_confuser_false_nyx_fraction']
            and measured['greedy_exact_sequences']/len(rows) >= gate['minimum_greedy_exact_sequence_fraction']
            and measured['mean_ctc_nll_per_token'] <= gate['maximum_mean_normalized_ctc_loss'])
        require(bool(passed) == arm['fit_gate_passed'] == info['fit_gate_passed'], 'Fit gate differs')
        reports[kind] = {'summary': measured, 'fit_gate_passed': bool(passed),
            'parameters': arm['parameters'], 'maximum_independent_ctc_loss_delta': max_delta,
            'metric_reconstruction_absolute_deltas': metric_deltas,
            'maximum_full_likelihood_replay_delta': score_replay_delta,
            'checkpoint_sha256': arm['checkpoint_sha256'], 'result_sha256': sha(path)}
        tf.keras.backend.clear_session()
    control = plan.get('control_arm', 'deep')
    treatment = plan.get('treatment_arm', 'local')
    deep = json.loads((root/'study'/f'{control}.json').read_bytes())
    local = json.loads((root/'study'/f'{treatment}.json').read_bytes())
    parent = json.loads((args.parent_study/'study/full-fit-phones.json').read_bytes())
    require(all(deep[a] == parent[b] for a, b in (
        ('encoder_weights_after_sha256', 'encoder_weights_after_sha256'),
        ('head_weights_after_sha256', 'head_weights_sha256'), ('summary', 'summary'),
        ('draw_schedule_sha256', 'draw_schedule_sha256'))), 'Parent control replay differs')
    paired_fields = ('binary_weights_before_sha256', 'draw_schedule_sha256', 'head_parameters')
    if settings[control]['initialization'] == settings[treatment]['initialization']:
        paired_fields += ('head_weights_before_sha256',)
    else:
        require(deep['head_weights_before_sha256'] != local['head_weights_before_sha256'],
                'Distinct head initializations unexpectedly agree')
    require(all(deep[k] == local[k] for k in paired_fields), 'Paired initial/draw/head identities differ')
    if settings[control]['tap'] == settings[treatment]['tap']:
        require(deep['encoder_weights_before_sha256'] == local['encoder_weights_before_sha256']
                and deep['parameters'] == local['parameters'], 'Matched encoder differs')
    require(result['parent_deep_control_replayed'] and result['accepted'] is False
            and result['advance'] == reports[treatment]['fit_gate_passed'], 'Final decision differs')
    report = {'plan_sha256': sha(root/'plan.json'), 'source_commit': launch['checkout_revision'],
        'actual_auditor_sha256': sha(Path(__file__)),
        'audit_correction_sha256': sha(args.audit_correction) if args.audit_correction else None,
        'result_sha256': sha(root/'study/result.json'), 'unique_crops': len(rows),
        'source_hashes': len({r['sha256'] for r in rows}), 'draw_schedule_sha256': schedule.hexdigest(),
        'all_crops_checkpoint_logits_metrics_and_gate_reconstructed': True,
        'parent_deep_control_replayed': True, 'arms': reports,
        'advance': result['advance'], 'accepted': False, 'limits': plan['limits']}
    with args.output.open('x') as out:
        out.write(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
