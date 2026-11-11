"""Verify frozen phone-study inputs and replays before scoring wake validation.

This reads existing research artifacts only. It does not train, infer, access
audio devices, change a recipe or authorize a firmware candidate.
"""
import argparse
import hashlib
import json
import math
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
    revision = subprocess.check_output(['git', 'rev-parse', args.revision], cwd=repo, text=True).strip()
    committed = subprocess.check_output(['git', 'show', revision+':'+args.plan_repository_path], cwd=repo)
    require(hashlib.sha256(committed).hexdigest() == sha(root/'plan.json'), 'Plan differs from frozen commit')
    upstream = subprocess.check_output(['git', '-C', str(args.upstream), 'rev-parse', 'HEAD'], text=True).strip()
    require(upstream == plan['upstream_commit'], 'Upstream revision differs')
    require(not subprocess.check_output(['git', '-C', str(args.upstream), 'diff', 'HEAD'], text=True),
            'Upstream has tracked source changes')
    for name, digest in plan['prior_failed_studies'].items():
        require(sha(repo/'research/wakeword_p4'/name) == digest, 'Prior result differs: '+name)
    require(sha(args.phone_bank/'bank.json') == plan['phone_bank_sha256'], 'Bank metadata differs')
    require(sha(args.phone_bank/'bank.npz') == plan['phone_bank_data_sha256'], 'Bank arrays differ')
    require(sha(args.initial_weights) == plan['init']['sha256'], 'Initializer differs')
    for name, digest in plan['feature_index_sha256'].items():
        require(sha(args.features/name) == digest, 'Feature index differs: '+name)
    prior = json.loads(args.prior_warmup.read_bytes())
    common = plan['common_train_args']
    steps = int(common[common.index('--steps')+1])
    shared, arms = {}, {}
    for arm, gain in plan['arms'].items():
        directory = root/f'model-{arm}'
        launch = json.loads((directory/'launch.json').read_bytes())
        train = json.loads((directory/'training.json').read_bytes())
        command = launch['command']
        start = command.index('--architecture')
        require(start == 14 and command[start:] == common, 'Training options differ: '+arm)
        require(command[2:start:2] == ['--output', '--upstream', '--features', '--initial-weights',
                                     '--phone-bank', '--phone-gain'], 'Unexpected input options: '+arm)
        for option, expected in (('--output', directory), ('--upstream', args.upstream),
                                 ('--features', args.features), ('--initial-weights', args.initial_weights),
                                 ('--phone-bank', args.phone_bank)):
            require((repo/command[command.index(option)+1]).resolve() == expected.resolve(),
                    'Input path differs: '+arm+' '+option)
        require(float(command[command.index('--phone-gain')+1]) == gain, 'Launch gain differs: '+arm)
        require(Path(command[1]).resolve() == directory/'source/train_stream.py', 'Wrong executed source: '+arm)
        require(launch['checkout_revision'] == revision, 'Launch revision differs: '+arm)
        for name, digest in plan['source_sha256'].items():
            require(launch['source_files_sha256'][name] == digest
                    and train['source_files_sha256'][name] == digest
                    and sha(directory/'source'/name) == digest, 'Source differs: '+arm+' '+name)
        for key, expected in {'steps': steps, 'initial_learning_rate': float(common[common.index('--learning-rate')+1]),
                              'input_frames': 335, 'batch_size': 128, 'negative_weight': 5,
                              'source_sampling': 'uniform', 'specaugment': True, 'align_stream': True,
                              'initial_weights_sha256': plan['init']['sha256']}.items():
            require(train[key] == expected, 'Training metadata differs: '+arm+' '+key)
        require([row['step'] for row in train['history']] == [steps], 'Unplanned checkpoint selection: '+arm)
        require(train['model_sha256'] == sha(directory/'hey_nyx.tflite'), 'Export hash differs: '+arm)
        phone = train['phone_distillation']
        require(phone['gain'] == gain and phone['objective'] == plan['phone_objective'], 'Phone objective differs: '+arm)
        require(phone['bank_sha256'] == plan['phone_bank_sha256']
                and phone['draws'] == steps*8 and phone['pairs'] == 118, 'Phone input/draw identity differs: '+arm)
        for name in ('sequence_sha256', 'teacher_posteriors_sha256'):
            require(phone['sequence'][name] == plan[name], 'Teacher sequence differs: '+arm+' '+name)
        warmup = phone['warmup']
        require(all(warmup[key] == value for key, value in plan['warmup'].items()), 'Warmup recipe differs: '+arm)
        required = plan['required_warmup_replay']
        require(warmup['head_weights_sha256'] == required['head_weights_sha256'], 'Head replay differs: '+arm)
        if 'weights_file_sha256' in required:
            require(sha(directory/'phone-warmup.weights.h5') == required['weights_file_sha256'],
                    'Warmup checkpoint replay differs: '+arm)
        for key in ('history', 'schedule_sha256', 'encoder_cache_sha256', 'initial_fit',
                    'binary_weights_before_sha256', 'binary_weights_after_sha256'):
            require(warmup[key] == prior[key], 'Prior warmup replay differs: '+arm+' '+key)
        for group in ('weights', 'optimizer'):
            require(warmup[f'binary_{group}_before_sha256'] == warmup[f'binary_{group}_after_sha256'],
                    'Warmup changed binary '+group+': '+arm)
        require(train['training_prefix_batches'] == 16, 'Incomplete prefix: '+arm)
        if plan.get('encoder_ratio_cap') is not None:
            cap, balance = plan['encoder_ratio_cap'], phone['encoder_balance']
            require(phone['encoder_ratio_cap'] == cap, 'Encoder cap differs: '+arm)
            if gain:
                require(balance['enabled'] and balance['steps'] == steps
                        and 0 < balance['balanced_steps'] <= steps
                        and math.isfinite(balance['maximum_scaled_encoder_ratio'])
                        and 0 <= balance['maximum_scaled_encoder_ratio'] <= cap+1e-6,
                        'Encoder gradient bound failed: '+arm)
            else:
                require(not balance['enabled'] and balance['steps'] == 0, 'Control applied balancing')
        if arm == 'control':
            for key, digest in plan.get('required_control_replay', {}).items():
                if key != 'scope':
                    require(train[key] == digest, 'Prior control replay differs: '+key)
        shared[arm] = {'warmup': warmup, 'prefix': train['training_prefix_sha256'],
                       'draw_schedule': phone['schedule_sha256'], 'sources': train['sources'],
                       'ptq_inputs': train['representative_training_inputs_sha256']}
        arms[arm] = {'launch_sha256': sha(directory/'launch.json'), 'training_sha256': sha(directory/'training.json'),
                     'model_sha256': train['model_sha256'], 'phone_fit': train['history'][0]['phone_fit'],
                     'encoder_balance': phone.get('encoder_balance')}
    require(shared['control'] == shared['phonetic'], 'Arms differ in warmup, source/pair draws, prefix or PTQ inputs')
    return {'source_sha256': sha(Path(__file__)), 'plan_sha256': sha(root/'plan.json'), 'checkout_revision': revision,
            'prior_reports_unchanged': True, 'frozen_plan_source_and_launch_match': True,
            'bank_initializer_and_index_identities_match': True, 'prior_warmup_replayed': True,
            'binary_state_preserved_by_warmup': True, 'shared_training_inputs': shared['control'], 'arms': arms,
            'limits': 'Input and replay audit only. Feature-index identities do not hash the full external scalar feature cache. Wake quality, inference resources, real-speaker performance and call stability require separate evidence.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('artifacts', 'repository', 'upstream', 'features', 'phone-bank', 'initial-weights', 'prior-warmup', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--revision', required=True)
    parser.add_argument('--plan-repository-path', required=True)
    args = parser.parse_args()
    result = audit(args)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps({key: value for key, value in result.items() if key not in ('shared_training_inputs', 'arms')}, indent=2))


if __name__ == '__main__':
    main()
