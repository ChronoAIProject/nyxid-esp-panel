"""Read-only input/replay audit for the compact-sequence mixed-source study.

Run after both training jobs finish and before new INT8 wake validation. No
training, inference, audio-device access or model advancement occurs here.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def audit(args):
    root, repo = args.artifacts.resolve(), args.repository.resolve()
    plan = json.loads((root/'plan.json').read_bytes())
    launch = json.loads((root/'study-launch.json').read_bytes())
    revision = launch['checkout_revision']
    committed = subprocess.check_output(['git', 'show', revision+':'+args.plan_repository_path], cwd=repo)
    require(hashlib.sha256(committed).hexdigest() == sha(root/'plan.json') == launch['plan_sha256'],
            'Plan differs from frozen commit or launch')
    require(subprocess.check_output(['git', '-C', str(args.upstream), 'rev-parse', 'HEAD'], text=True).strip()
            == plan['upstream_commit'], 'Upstream revision differs')
    require(not subprocess.check_output(['git', '-C', str(args.upstream), 'diff', 'HEAD']), 'Upstream source changed')
    require(sha(args.initial_weights) == plan['init']['sha256'], 'Initializer differs')
    for name, digest in plan['feature_index_sha256'].items():
        require(sha(args.features/name) == digest, 'Feature index differs: '+name)
    if plan.get('feature_data_sha256'):
        require(sha(args.features/'features.bin') == plan['feature_data_sha256'],
                'Full scalar feature cache differs')
    if plan.get('background_inputs_sha256'):
        from background_inputs import inventory
        require(sha(root/'background-inputs.json') == plan['background_inputs_sha256'],
                'Frozen background inventory differs')
        require(inventory(args.features) == json.loads((root/'background-inputs.json').read_bytes()),
                'Actual training/validation background bytes differ')
    require(sha(args.bank/'bank.json') == plan['phone_bank_sha256'], 'Bank metadata differs')
    require(sha(args.bank/'bank.npz') == plan['phone_bank_data_sha256'], 'Bank array differs')
    if plan.get('bank_plan_sha256'):
        bank = json.loads((args.bank/'bank.json').read_bytes())
        require(bank['plan_sha256'] == plan['bank_plan_sha256']
                and bank['feature_data_sha256'] == plan['feature_data_sha256']
                and bank['coverage'] == plan['bank_coverage'], 'Broader bank provenance differs')
    common = plan['common_train_args']
    steps = int(common[common.index('--steps')+1])
    shared, arms = {}, {}
    for arm, gain in plan['arms'].items():
        directory = root/f'model-{arm}'
        execution = json.loads((directory/'launch.json').read_bytes())
        train = json.loads((directory/'training.json').read_bytes())
        command = execution['command']
        start = command.index('--architecture')
        background_gain = plan.get('background_gains', {}).get(arm)
        positive_gain = plan.get('positive_gains', {}).get(arm)
        options = ['--output', '--upstream', '--features', '--initial-weights',
            '--phone-bank', '--phone-gain', '--compact-phrase-plan']
        if background_gain is not None:
            options.append('--phone-background-gain')
        if positive_gain is not None:
            options.append('--phone-positive-gain')
        require(start == 2+2*len(options) and command[start:] == common, 'Training options differ: '+arm)
        require(command[2:start:2] == options, 'Input options differ: '+arm)
        for option, expected in (('--output', directory), ('--upstream', args.upstream),
                ('--features', args.features), ('--initial-weights', args.initial_weights),
                ('--phone-bank', args.bank), ('--compact-phrase-plan', root/'plan.json')):
            require((repo/command[command.index(option)+1]).resolve() == expected.resolve(),
                    'Input path differs: '+arm+' '+option)
        require(float(command[command.index('--phone-gain')+1]) == gain, 'Launch gain differs: '+arm)
        require(Path(command[1]).resolve() == directory/'source/train_stream.py', 'Wrong executed source: '+arm)
        require(execution['checkout_revision'] == revision, 'Source revision differs: '+arm)
        for name, digest in plan['source_sha256'].items():
            require(execution['source_files_sha256'][name] == digest
                and train['source_files_sha256'][name] == digest
                and sha(directory/'source'/name) == digest, 'Executed source differs: '+arm+' '+name)
        for key, expected in {'steps': steps, 'initial_learning_rate': .0002, 'input_frames': 335,
                'architecture': 'transfer60', 'batch_size': 128, 'negative_weight': 5,
                'source_sampling': 'uniform', 'specaugment': True, 'align_stream': True,
                'initial_weights_sha256': plan['init']['sha256'], 'parameters': 25831,
                'training_prefix_batches': 16, 'phrase_output': False, 'paired_near_words': None}.items():
            require(train[key] == expected, 'Training metadata differs: '+arm+' '+key)
        require([row['step'] for row in train['history']] == [steps], 'Unplanned checkpoint selection: '+arm)
        require(train['model_sha256'] == sha(directory/'hey_nyx.tflite'), 'Export identity differs: '+arm)
        phone = train['phone_distillation']
        require(phone['gain'] == gain and phone['objective'] == 'ctc' and phone['vocabulary'] == 6,
                'Compact objective differs: '+arm)
        require(phone['bank_sha256'] == plan['phone_bank_sha256']
                and phone['pairs'] == plan['compact_bank']['rows']//2
                and phone['draws'] == steps*8 and phone['compact_phrase_plan_sha256'] == sha(root/'plan.json'),
                'Compact input/draw identity differs: '+arm)
        require(phone['sequence']['tokens'] == ['blank', 'hey', 'nyx', 'mix', 'next', 'nick'],
                'Compact vocabulary differs: '+arm)
        require(phone['encoder_ratio_cap'] is None and not phone['encoder_balance']['enabled'],
                'Unplanned gradient cap: '+arm)
        if background_gain is not None:
            background = phone['background']
            require(float(command[command.index('--phone-background-gain')+1]) == background_gain
                    and background['gain'] == background_gain and background['steps'] == steps
                    and background['examples'] == steps*48 and background['frames_per_example'] == 67,
                    'Background objective/cohort count differs: '+arm)
        else:
            require(phone.get('background') is None, 'Unplanned background supervision: '+arm)
        if positive_gain is not None:
            from positive_exposure import audit as audit_positive
            require(float(command[command.index('--phone-positive-gain')+1]) == positive_gain
                    and phone['ordinary_positive']['gain'] == positive_gain,
                    'Positive objective gain differs: '+arm)
            positive_checked = audit_positive(directory, args.features, steps)
        else:
            require(phone.get('ordinary_positive') is None, 'Unplanned positive supervision: '+arm)
            positive_checked = None
        warmup = phone['warmup']
        require(all(warmup[k] == v for k, v in plan['warmup'].items()), 'Warmup recipe differs: '+arm)
        for group in ('weights', 'optimizer'):
            require(warmup[f'binary_{group}_before_sha256'] == warmup[f'binary_{group}_after_sha256'],
                    'Warmup changes binary '+group+': '+arm)
        if arm == 'control':
            for key, digest in plan.get('required_control_replay', {}).items():
                require(train[key] == digest, 'Original control replay differs: '+key)
            if plan.get('required_control_warmup'):
                require(warmup == plan['required_control_warmup'], 'Original control warmup replay differs')
                require(phone['schedule_sha256'] == plan['required_control_schedule_sha256'],
                        'Original control pair schedule replay differs')
        shared[arm] = {'warmup': warmup, 'prefix': train['training_prefix_sha256'],
            'draw_schedule': phone['schedule_sha256'], 'sequence': phone['sequence'],
            'sources': train['sources'], 'ptq_inputs': train['representative_training_inputs_sha256']}
        if positive_checked is not None:
            shared[arm]['positive_exposure'] = positive_checked
        arms[arm] = {'launch_sha256': sha(directory/'launch.json'),
            'training_sha256': sha(directory/'training.json'), 'model_sha256': train['model_sha256'],
            'phone_fit': train['history'][0]['phone_fit'],
            'checkpoint_sha256': {path.name: sha(path) for path in sorted(directory.glob('*.weights.h5'))}}
    require(shared['control'] == shared['phonetic'], 'Arms differ in warmup, batches, pairs, targets or PTQ')
    return {'source_sha256': sha(Path(__file__)), 'plan_sha256': sha(root/'plan.json'),
        'checkout_revision': revision, 'frozen_source_plan_and_launch_match': True,
        'bank_initializer_and_index_identities_match': True, 'warmup_preserves_binary_state': True,
        'original_binary_control_replayed': bool(plan.get('required_control_replay')),
        'shared_training_inputs': shared['control'],
        'arms': arms,
        'full_scalar_feature_cache_hashed': bool(plan.get('feature_data_sha256')),
        'training_and_validation_background_bytes_hashed': bool(plan.get('background_inputs_sha256')),
        'limits': 'Read-only inputs/replay audit. Full scalar feature bytes are verified only when the plan explicitly supplies feature_data_sha256; compact sampler also checks exact bank crops during training. No wake-quality, pronunciation, P4 resource or call-stability acceptance.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('artifacts', 'repository', 'upstream', 'features', 'bank', 'initial-weights', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--plan-repository-path', default='research/wakeword_p4/training/plans/microwake-compact-phrase-ctc-v1.json')
    args = parser.parse_args()
    result = audit(args)
    with args.output.open('x') as stream:
        stream.write(json.dumps(result, indent=2)+'\n')
    print(json.dumps({key: value for key, value in result.items() if key not in ('shared_training_inputs', 'arms')}))


if __name__ == '__main__':
    main()
