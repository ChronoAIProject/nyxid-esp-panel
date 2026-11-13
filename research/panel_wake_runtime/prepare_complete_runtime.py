"""Prepare a separate 225-frame development build with a complete NYXKWS02 test.

The caller supplies a frozen profile after completed-model/numerical admission
and non-null new-development selection. This preparer authenticates those
artifact bindings, not the numerical admission calculation itself. It grants
no independent acceptance or production permission.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import struct


SCHEMA = 'complete-development-p4-profile-v1'
TAIL = 'whole-features-final-zero-pad-stride3-v1'
ARITHMETIC = 'uint8-sum-binary64-v1'


def digest(data):
    return hashlib.sha256(data).hexdigest()


def prepare(profile_path, model_path, calibration_path, selftest_path, output):
    profile_path = Path(profile_path)
    profile_bytes = profile_path.read_bytes()
    profile = json.loads(profile_bytes)
    if (profile['schema'] != SCHEMA or profile['accepted'] is not False
            or profile['target_id'] != 'hey_nyx' or profile['input_frames'] != 225
            or profile['event_arithmetic'] != ARITHMETIC or profile['tail_policy'] != TAIL
            or profile['smoothing_steps'] != 5 or profile['refractory_steps'] != 67):
        raise ValueError('Expected explicit225-frame Nyx research profile and complete event policy')
    artifacts = {}
    for name in ('model', 'training', 'completion', 'calibration', 'execution', 'selftest'):
        item = profile['artifacts'][name]
        path = profile_path.parent/item['path']
        data = path.read_bytes()
        if digest(data) != item['sha256']:
            raise ValueError('Changed profile artifact: '+name)
        artifacts[name] = data
    for name,path in (('model',model_path), ('calibration',calibration_path), ('selftest',selftest_path)):
        if path is None or Path(path).read_bytes() != artifacts[name]:
            raise ValueError('Build argument differs from frozen profile: '+name)
    model = artifacts['model']; model_sha = digest(model)
    if not 1024 <= len(model) <= 512*1024 or model[4:8] != b'TFL3':
        raise ValueError('Expected bounded TFLite model')
    training = json.loads(artifacts['training'])
    completion = json.loads(artifacts['completion'])
    calibration = json.loads(artifacts['calibration'])
    execution = json.loads(artifacts['execution'])
    reviewed = json.loads((Path(__file__).parent/'plans/gpu-v30-runtime-v1.json').read_bytes())
    if (training['architecture'] != 'teacher60' or training['architecture_args'] != reviewed['architecture_args']
            or training['context_frames'] != 225 or training['parameters'] != 24301
            or training['steps'] != 120000 or training['selected_step'] != 120000
            or training['checkpoint_policy'] != 'fixed-final' or training['classifier_reset'] is not False
            or training['batchnorm_initial'] != training['batchnorm_selected']
            or training['model_sha256'] != model_sha or training['model_bytes'] != len(model)):
        raise ValueError('Training report differs from reviewed225 fixed-final family')
    batch = training['batch_schedule']
    if (batch['train_batches'] != 120000 or batch['ptq_batches'] != 100
            or batch['verified_batches'] != 120100 or batch['all_actual_batch_tensors_match_precomputed'] is not True
            or completion['target'] != 'hey_nyx' or completion['steps'] != 120000
            or completion['training_report_sha256'] != digest(artifacts['training'])
            or completion['model_sha256'] != model_sha
            or any(completion[k] is not True for k in ('finite_weights','finite_logged_loss','frozen_bn_exact'))):
        raise ValueError('Original training completion evidence differs')
    model_id = profile['model_id']; selected = calibration.get('selected')
    if (execution['schema'] != 'complete-development-stream-execution-v1'
            or execution['tail_policy'] != TAIL or execution['authorizes'] != 'development-raw-collection-only'
            or calibration['schema'] != 'complete-development-calibration-v1'
            or calibration['model_id'] != model_id or calibration['target_id'] != 'hey_nyx'
            or calibration['execution_sha256'] != digest(artifacts['execution'])
            or calibration['model'] != execution['models'][model_id]
            or calibration['materials'] != execution['materials']
            or calibration['model']['sha256'] != model_sha or calibration['model']['bytes'] != len(model)
            or calibration['model']['target_id'] != 'hey_nyx'
            or any(execution['materials'][phase]['role'] != 'new-development' for phase in ('synthetic','background'))
            or calibration['candidate_promoted'] is not False or calibration['acceptance_authorized'] is not False
            or not isinstance(selected,dict) or selected.get('eligible') is not True
            or selected not in calibration['grid'] or selected.get('registered_grid') is not True):
        raise ValueError('Need authenticated non-null new-development operating point')
    threshold = profile['threshold']
    if (type(threshold) not in (int,float) or not math.isfinite(threshold) or not 0 < threshold < 1
            or threshold != calibration['threshold'] or threshold != selected['threshold']):
        raise ValueError('Profile must preserve the frozen selected threshold')
    fixture = artifacts['selftest']
    if len(fixture) < 48 or fixture[:8] != b'NYXKWS02' or fixture[8:40] != bytes.fromhex(model_sha):
        raise ValueError('Need model-bound complete NYXKWS02 test')
    samples,outputs = struct.unpack_from('<II',fixture,40)
    frames = max(0,(samples+159)//160-2)
    if (not 960 <= samples <= 16000*16 or outputs != (frames+2)//3
            or len(fixture) != 48+samples*2+outputs or outputs < (225+2)//3+5):
        raise ValueError('Incomplete or insufficient225-frame self-test coverage')
    expected = fixture[48+samples*2:]
    if min(expected) == max(expected):
        raise ValueError('Self-test must contain varied raw outputs')
    manifest = dict(schema='complete-development-p4-prepared-v1',
        scope='Opt-in225-frame development firmware only; no independent recognition/voice acceptance or production release',
        model_id=model_id,target_id='hey_nyx',model_sha256=model_sha,model_bytes=len(model),profile_sha256=digest(profile_bytes),
        artifact_sha256={key:digest(value) for key,value in artifacts.items()},
        input_frames=225,sample_rate=16000,frontend_hop_samples=160,model_stride_frames=3,
        threshold=threshold,smoothing_steps=5,refractory_steps=67,event_arithmetic=ARITHMETIC,tail_policy=TAIL,
        selftest_version=2,selftest_samples=samples,selftest_outputs=outputs,
        final_wave_padding=(-samples)%160,final_feature_padding=(-frames)%3,
        prepare_source_sha256=digest(Path(__file__).read_bytes()),accepted=False)
    config = ('#pragma once\n'+f'#define PANEL_WAKE_MODEL_SHA256 "{model_sha}"\n'
        +f'#define PANEL_WAKE_THRESHOLD {threshold:.17g}\n'
        +'#define PANEL_WAKE_SMOOTHING_STEPS 5\n#define PANEL_WAKE_PRIME_SAMPLES 68000\n'
        +'#define PANEL_WAKE_COMPLETE_INPUT 1\n#define PANEL_WAKE_SELFTEST_VERSION 2\n').encode()
    files = {'hey_nyx_model.bin':model,'wake_selftest.bin':fixture,'wake_config.h':config,
             'manifest.json':(json.dumps(manifest,indent=2)+'\n').encode()}
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        if {p.name for p in output.iterdir()} != set(files) or any((output/name).read_bytes()!=data for name,data in files.items()):
            raise FileExistsError('Preserve different or partial complete preparation; use a new build directory')
        return manifest
    output.mkdir(parents=True,exist_ok=True)
    for name,data in files.items():
        with (output/name).open('xb') as target:
            target.write(data)
    return manifest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('profile','model','calibration','selftest','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    result=prepare(args.profile,args.model,args.calibration,args.selftest,args.output)
    print(json.dumps(dict(model_sha256=result['model_sha256'],selftest_outputs=result['selftest_outputs'],accepted=False)))


if __name__=='__main__':main()
