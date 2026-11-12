"""Authenticate completed matched run, then use unchanged historical-only scoring/gates."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--arm', choices=['independent','common'], required=True)
a=p.parse_args()
tag=f'matched-noise-{a.arm}-formal-v1'

def run(script, *args, gpu=False):
    command=[sys.executable]
    if gpu: command.append('research/wakeword_p4/training/gpu_python.py')
    subprocess.run(command+['research/wakeword_p4/training/'+script]+list(args), check=True)

def historical(root, tag):
    validation=f'.local/wakeword/eval-{tag}-ami-extra-validation'
    assert not Path(validation).exists()
    run('evaluate_stream.py','--model',str(root/'hey_nyx.tflite'),
        '--features','.local/wakeword/features-gpu-v6-qwen','--output',validation,
        '--validation-only','--threshold-step','0.00390625')
    for corpus,features in [('mandarin','aishell'),('dipco','dipco'),('libri-dev-other','libri-dev-other')]:
        run('evaluate_frozen_human.py','--model',str(root/'hey_nyx.tflite'),
            '--frozen-report',validation+'/report.json','--features',f'.local/wakeword/{features}-human-features-v1' if corpus!='libri-dev-other' else '.local/wakeword/libri-dev-other-features-v1',
            '--corpus-name',corpus,'--output',f'.local/wakeword/eval-{tag}-{corpus}-validation')
    args=['--english-validation',validation]
    for corpus in ['mandarin','dipco','libri-dev-other']:
        args+=['--negative-validation',f'.local/wakeword/eval-{tag}-{corpus}-validation']
    run('calibrate_multicorpus.py',*args,'--output',f'.local/wakeword/{tag}-expanded-validation-calibration.json')
    threshold=json.loads(Path(validation+'/report.json').read_text())['threshold']
    run('compare_streaming.py','--model-dir',str(root),'--features','.local/wakeword/features-gpu-v6-qwen',
        '--threshold',str(threshold),'--upstream','.local/wakeword/microWakeWord',
        '--output',f'.local/wakeword/{tag}-streaming-probe.json',gpu=True)

root=Path(f'.local/wakeword/model-{tag}')
training=json.loads((root/'training.json').read_text())
if (training['steps']!=120000 or training['effective_training_batch_size']!=128
        or hashlib.sha256((root/'hey_nyx.tflite').read_bytes()).hexdigest()!=training['model_sha256']):
    raise RuntimeError(f'Training process ended without complete authenticated {tag} output')
if (training.get('balanced_coverage',{}).get('seed')!=207726
        or training['balanced_coverage']['temporal']['seed']!=217726
        or training['balanced_coverage']['positive_slots']!=list(range(8,16))
        or training['balanced_coverage']['negative_slots']!=list(range(44,56))
        or training['balanced_coverage']['temporal']['complete_negative_slots']!=list(range(44,50))
        or training['balanced_coverage']['temporal']['timeline_negative_slots']!=list(range(50,56))
        or training['balanced_coverage']['pack_sha256']!='1a6cbf491c624c58cf96cfdac882966a7863927f546ac553075248caacfaaf5f'
        or training['balanced_coverage']['report_sha256']!='7b798653f73e2dcd3b01fd5b7092c947a26b9c8ec2525ff295de2e6478042e44'
        or len(training['balanced_coverage']['groups'])!=2688
        or set(training['balanced_coverage']['groups'].values())!={15}
        or training.get('us_coverage',{}).get('temporal',{}).get('seed')!=177726
        or training['us_coverage']['temporal']['complete_negative_slots']!=list(range(32,38))
        or training['us_coverage']['temporal']['timeline_negative_slots']!=list(range(38,44))
        or training.get('us_coverage',{}).get('seed')!=167726
        or training['us_coverage']['pack_sha256']!='4a0d6febb8c0cd72b6bc58d26123d71b3cf3aa06736e013fd6ccae27dc58e6da'
        or training['us_coverage']['positive_slots']!=list(range(8))
        or training['us_coverage']['negative_slots']!=list(range(32,44))
        or training['complete_coverage']['seed']!=127726
        or training['complete_coverage']['input_sha256']['training.npz']!='bb01f3219db10f31e1095af4912c1807ec38b0b3c640c4676186c1b77a620aa1'
        or training['complete_coverage'].get('negative_timeline',{}).get('seed')!=137726
        or training['complete_coverage'].get('positive_timeline',{}).get('seed')!=147726
        or training['complete_coverage'].get('negative_container_mixture',{}).get('seed')!=157726
        or training['complete_coverage']['negative_container_mixture']['batch_slots']!=list(range(32,44))
        or training['five_score'] is not None or 'temporal' in training['full_source']
        or training['phone_binary'] is not None or training['full_source']['seed']!=97726
        or training['prefix_negatives']['count_per_batch']!=0
        or training['full_source']['pack_sha256']!='c0c389fc4a22e1fb9807d99e8a7a98e0e028696d2047dd27dce81f1de8fef096'
        or training['batchnorm_initial']!=training['batchnorm_selected']):
    raise RuntimeError(f'{tag} recipe or frozen BN mismatch')
paired=training['paired_rendering']
assert paired['objective']=='phrase-ranking' and paired['phrase_groups']==10
assert paired['coefficient'] == 1 and paired['noise_condition']==a.arm
assert paired['optimizer_updates'] == 120000
assert paired['numerical_policy']['op_determinism'] and not paired['numerical_policy']['tf32']
assert len((root/'paired-sampling-trace.jsonl').read_text().splitlines()) == 120000
subprocess.run([sys.executable,'research/wakeword_p4/training/audit_stream_export.py',
    '--model-dir',str(root),'--expected-steps','120000',
    '--output',f'.local/wakeword/{tag}-export-audit.json'],check=True)
print(f'Authenticated complete {tag}; starting exploratory historical regression',flush=True)
historical(root, tag)
subprocess.run([sys.executable,'research/wakeword_p4/training/score_phone_development.py',
    '--manifest','.local/wakeword/synthetic-phone-development-replay-v1/manifest.json',
    '--model',str(root/'hey_nyx.tflite'),
    '--calibration',f'.local/wakeword/eval-{tag}-ami-extra-validation/report.json',
    '--official-reports','research/wakeword_p4/results/4090-phone-development-official-replay-reports-v1.json',
    '--output',f'.local/wakeword/{tag}-phone-development-historical-threshold-v1'],check=True)
subprocess.run([sys.executable,'research/wakeword_p4/training/score_phone_development.py',
    '--manifest','.local/wakeword/vctk-development-replay-v1/manifest.json',
    '--model',str(root/'hey_nyx.tflite'),
    '--calibration',f'.local/wakeword/eval-{tag}-ami-extra-validation/report.json',
    '--official-reports','research/wakeword_p4/results/4090-vctk-development-official-replay-reports-v1.json',
    '--output',f'.local/wakeword/{tag}-vctk-development-historical-threshold-v1'],check=True)
subprocess.run([sys.executable,'research/wakeword_p4/training/evaluate_frozen_human.py',
    '--model',str(root/'hey_nyx.tflite'),
    '--frozen-report',f'.local/wakeword/eval-{tag}-ami-extra-validation/report.json',
    '--features','.local/wakeword/voxconverse-development-features-v1',
    '--output',f'.local/wakeword/{tag}-voxconverse-development-v1',
    '--corpus-name','VoxConverse unscreened conversational background','--identity-unit','recording'],check=True)
subprocess.run([sys.executable,'research/wakeword_p4/training/calibrate_phone_development.py',
    '--synthetic',f'.local/wakeword/{tag}-phone-development-historical-threshold-v1',
    f'.local/wakeword/{tag}-vctk-development-historical-threshold-v1',
    '--background',f'.local/wakeword/{tag}-voxconverse-development-v1',
    '--output',f'.local/wakeword/{tag}-reused-development-exploratory-calibration-v1.json'],check=True)
print('Exploratory diagnostics finished; reused development only, no promotion/final-data access/publication',flush=True)
