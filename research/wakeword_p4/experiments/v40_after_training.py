"""Wait for the identified existing v40 process, then evaluate completed output."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--training-pid',type=int,required=True)
a=p.parse_args()
proc=Path(f'/proc/{a.training_pid}/stat')

def identity():
    try:
        # Process name may contain spaces; fields after its closing parenthesis.
        return proc.read_text().rsplit(')',1)[1].split()[19]
    except FileNotFoundError:
        return None

original=identity()
if original is None:
    raise RuntimeError('Specified live training PID missing; inspect before scheduling')
print(json.dumps(dict(waiting_pid=a.training_pid,starttime=original)),flush=True)
while identity()==original:
    time.sleep(10)
root=Path('.local/wakeword/model-gpu-v40-ami-extra')
training=json.loads((root/'training.json').read_text())
if (training['steps']!=120000 or training['effective_training_batch_size']!=128
        or hashlib.sha256((root/'hey_nyx.tflite').read_bytes()).hexdigest()!=training['model_sha256']):
    raise RuntimeError('Training process ended without complete authenticated v40 output')
if (training['complete_coverage']['seed']!=127726
        or training['complete_coverage']['input_sha256']['training.npz']!='bb01f3219db10f31e1095af4912c1807ec38b0b3c640c4676186c1b77a620aa1'
        or training['five_score'] is not None or 'temporal' in training['full_source']
        or training['phone_binary'] is not None or training['full_source']['seed']!=97726
        or training['prefix_negatives']['count_per_batch']!=0
        or training['full_source']['pack_sha256']!='c0c389fc4a22e1fb9807d99e8a7a98e0e028696d2047dd27dce81f1de8fef096'
        or training['batchnorm_initial']!=training['batchnorm_selected']):
    raise RuntimeError('v40 recipe or frozen BN mismatch')
subprocess.run([sys.executable,'research/wakeword_p4/training/audit_stream_export.py',
    '--model-dir',str(root),'--expected-steps','120000',
    '--output','.local/wakeword/v40-export-audit.json'],check=True)
print('Authenticated complete v40; starting exploratory historical regression',flush=True)
subprocess.run(['bash','research/wakeword_p4/experiments/v40_historical_regression.sh'],check=True)
subprocess.run([sys.executable,'research/wakeword_p4/training/score_phone_development.py',
    '--manifest','.local/wakeword/synthetic-phone-development-replay-v1/manifest.json',
    '--model',str(root/'hey_nyx.tflite'),
    '--calibration','.local/wakeword/eval-v40-ami-extra-validation/report.json',
    '--official-reports','research/wakeword_p4/results/4090-phone-development-official-replay-reports-v1.json',
    '--output','.local/wakeword/v40-phone-development-historical-threshold-v1'],check=True)
subprocess.run([sys.executable,'research/wakeword_p4/training/score_phone_development.py',
    '--manifest','.local/wakeword/vctk-development-replay-v1/manifest.json',
    '--model',str(root/'hey_nyx.tflite'),
    '--calibration','.local/wakeword/eval-v40-ami-extra-validation/report.json',
    '--official-reports','research/wakeword_p4/results/4090-vctk-development-official-replay-reports-v1.json',
    '--output','.local/wakeword/v40-vctk-development-historical-threshold-v1'],check=True)
subprocess.run([sys.executable,'research/wakeword_p4/training/evaluate_frozen_human.py',
    '--model',str(root/'hey_nyx.tflite'),
    '--frozen-report','.local/wakeword/eval-v40-ami-extra-validation/report.json',
    '--features','.local/wakeword/voxconverse-development-features-v1',
    '--output','.local/wakeword/v40-voxconverse-development-v1',
    '--corpus-name','VoxConverse unscreened conversational background','--identity-unit','recording'],check=True)
subprocess.run([sys.executable,'research/wakeword_p4/training/calibrate_phone_development.py',
    '--synthetic','.local/wakeword/v40-phone-development-historical-threshold-v1',
    '.local/wakeword/v40-vctk-development-historical-threshold-v1',
    '--background','.local/wakeword/v40-voxconverse-development-v1',
    '--output','.local/wakeword/v40-reused-development-exploratory-calibration-v1.json'],check=True)
print('Exploratory diagnostics finished; reused development only, no promotion/final-data access/publication',flush=True)
