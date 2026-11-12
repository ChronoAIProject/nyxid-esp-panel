"""Authenticate original v46 source/recipe records and recompute float choices.

Recorded metadata and selection arithmetic only; no model or score-array replay.
This does not replace the pending original INT8 and old-stage joint evaluation.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess


HERE = Path(__file__).resolve().parent
REF = '65ebbcce0c87bb8a51cb5abfcd56a1c9bfe0ab4f'
MODEL = '5ac62943e862b2a8b7c92d97b1b4e89a096a2bb9d76f27b1b4b72b25a427f417'
THRESHOLDS = [.50, .70, .80, .90, .95, .97, .98, .985, .99, .992, .994]


def sha(data):
    return hashlib.sha256(data).hexdigest()


def blob(name):
    return subprocess.check_output(['git', '-C', str(HERE.parents[1]), 'show',
        REF + ':research/wakeword_p4/' + name])


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), 'Fresh output required')
    inputs = {name: blob('results/' + name) for name in (
        '4090-v46-training.json', '4090-v46-export-audit.json', '4090-v45-training-v1.json')}
    training, export, prior = [json.loads(value) for value in inputs.values()]
    require(sha(inputs['4090-v46-training.json']) == export['training_sha256']
        == 'bfa9cb5995c28987fc1e0f7b1b53382ce273efed6979b795122cca6df1a33858', 'Original training identity differs')
    require(training['model_sha256'] == export['model_sha256'] == MODEL, 'Model identity differs')
    sources = training['source_files_sha256']
    for name, digest in sources.items():
        require(Path(name).name == name and sha(blob('training/' + name)) == digest,
            'Source differs: ' + name)
    require(export['source_sha256'] == sources['audit_stream_export.py'], 'Export audit source differs')
    require(set(training) - set(prior) == {'balanced_coverage'} and not set(prior) - set(training),
        'Unexpected recipe keys changed')
    changed = {'selected_step', 'history', 'seconds', 'model_sha256', 'source_files_sha256'}
    require(all(training[k] == prior[k] for k in prior if k not in changed), 'Inherited recipe differs')
    require(training['batchnorm_initial'] == training['batchnorm_selected']
        and export['frozen_bn_unchanged'], 'Recorded frozen BN differs')
    balanced = training['balanced_coverage']
    require(balanced['seed'] == 207726 and balanced['temporal']['seed'] == 217726,
        'Balanced RNG identities differ')
    require(balanced['pack_sha256'] == '37ea6ff5ffc8c85973afec3a63f231527e493ddeb6cee77c08d594e96bc15546'
        and balanced['report_sha256'] == '93733755fec50b41cf71fcc8e2efa9fdfc14956a61e8b50e98e4bd110d5248b3',
        'Balanced original inputs differ')
    require(balanced['source_sha256'] == sources['balanced_coverage.py']
        and balanced['positive_slots'] == balanced['temporal']['positive_slots'] == list(range(8, 16))
        and balanced['negative_slots'] == list(range(44, 56))
        and balanced['temporal']['complete_negative_slots'] == list(range(44, 50))
        and balanced['temporal']['timeline_negative_slots'] == list(range(50, 56)), 'Balanced slots differ')
    voices = None
    for phrase in ('hey_nyx', 'hey_nick', 'hey_next', 'hey_mix'):
        group = {int(k.split('/')[1]) for k in balanced['groups'] if k.startswith(phrase + '/')}
        require(len(group) == 672 and (voices is None or voices == group), 'Conditioning ID grid differs')
        voices = group
    require(len(balanced['groups']) == 2688 and set(balanced['groups'].values()) == {9}, 'Variant grid differs')
    history = training['history']
    require([r['step'] for r in history] == list(range(5000, 120001, 5000)), 'Incomplete checkpoint history')
    best, chosen, eligible_points = None, None, 0
    for item in history:
        require([r['threshold'] for r in item['grid']] == THRESHOLDS, 'Original float grid differs')
        eligible = [r for r in item['grid'] if r['ambient_faph'] <= .5 and r['max_source_false_rate'] <= .01]
        eligible_points += len(eligible)
        if eligible:
            selected = max(eligible, key=lambda r: (r['macro_recall'], -r['ambient_faph'], -r['max_source_false_rate']))
            rank = (1, selected['macro_recall'], -selected['ambient_faph'], -selected['max_source_false_rate'])
        else:
            selected = min(item['grid'], key=lambda r: (r['ambient_faph'], r['max_source_false_rate'], -r['macro_recall']))
            rank = (0, -selected['ambient_faph'], -selected['max_source_false_rate'], selected['macro_recall'])
        require(selected == item['selected'], 'Recorded checkpoint choice differs')
        if best is None or rank > best:
            best, chosen = rank, item
    require(chosen['step'] == training['selected_step'] == export['selected_step'] == 10000,
        'Original selected checkpoint differs')
    require(training['steps'] == export['steps'] == 120000 and training['parameters'] == export['parameters'] == 22801
        and training['model_bytes'] == export['model_bytes'] == 54440, 'Recorded geometry differs')
    require(export['input_shape'] == [1, 3, 40] and export['output_shape'] == [1, 1]
        and export['input_dtype'] == 'int8' and export['output_dtype'] == 'uint8'
        and not export['candidate_promoted'], 'Recorded export interface differs')
    result = dict(scope=__doc__, gpu_commit=REF, model_sha256=MODEL,
        input_reports_sha256={name: sha(value) for name, value in inputs.items()},
        original_training_sources_authenticated=len(sources), inherited_recipe_unchanged=True,
        us_coverage_unchanged=True, balanced_conditioning_ids=len(voices), balanced_groups=2688,
        balanced_variants_per_group=9, balanced_source_sha256=balanced['source_sha256'],
        checkpoints_recomputed=len(history), float_grid_points_recomputed=len(history) * len(THRESHOLDS),
        float_eligible_points=eligible_points, selected_step=chosen['step'], selected_float_point=chosen['selected'],
        original_export_record_contract_matches=True, all_checks_pass=True, accepted=False,
        script_sha256=sha(Path(__file__).read_bytes()),
        limits='Source/report bytes and saved-record selection arithmetic only. Recorded finite weights, frozen BN and export interface were not independently executed or rehashed against model bytes on Mac. No model/weights/feature/score-array replay, full trainer RNG proof, INT8 acceptance, unused-development/final access, P4 or audio operation. v46 combines source replacement with temporal-mixture change; no isolated causal or parity claim. Full old-stage evaluation requires separate evidence; do not restart or duplicate the original run.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
