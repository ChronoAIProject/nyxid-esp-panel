"""Wait for the identified existing v36 process, then evaluate completed output."""
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
root=Path('.local/wakeword/model-gpu-v36-ami-extra')
training=json.loads((root/'training.json').read_text())
if (training['steps']!=120000 or training['effective_training_batch_size']!=144
        or hashlib.sha256((root/'hey_nyx.tflite').read_bytes()).hexdigest()!=training['model_sha256']):
    raise RuntimeError('Training process ended without complete authenticated v36 output')
if (not training['phone_binary']['source_balanced']
        or training['phone_binary']['pack_sha256']!='c0c389fc4a22e1fb9807d99e8a7a98e0e028696d2047dd27dce81f1de8fef096'
        or training['batchnorm_initial']!=training['batchnorm_selected']):
    raise RuntimeError('v36 recipe or frozen BN mismatch')
subprocess.run([sys.executable,'research/wakeword_p4/training/audit_stream_export.py',
    '--model-dir',str(root),'--expected-steps','120000',
    '--output','.local/wakeword/v36-export-audit.json'],check=True)
print('Authenticated complete v36; starting exploratory historical regression',flush=True)
subprocess.run(['bash','research/wakeword_p4/experiments/v36_historical_regression.sh'],check=True)
subprocess.run([sys.executable,'research/wakeword_p4/training/score_phone_development.py',
    '--manifest','.local/wakeword/synthetic-phone-development-replay-v1/manifest.json',
    '--model',str(root/'hey_nyx.tflite'),
    '--calibration','.local/wakeword/eval-v36-ami-extra-validation/report.json',
    '--official-reports','research/wakeword_p4/results/4090-phone-development-official-replay-reports-v1.json',
    '--output','.local/wakeword/v36-phone-development-historical-threshold-v1'],check=True)
subprocess.run([sys.executable,'research/wakeword_p4/training/score_phone_development.py',
    '--manifest','.local/wakeword/vctk-development-replay-v1/manifest.json',
    '--model',str(root/'hey_nyx.tflite'),
    '--calibration','.local/wakeword/eval-v36-ami-extra-validation/report.json',
    '--official-reports','research/wakeword_p4/results/4090-vctk-development-official-replay-reports-v1.json',
    '--output','.local/wakeword/v36-vctk-development-historical-threshold-v1'],check=True)
subprocess.run([sys.executable,'research/wakeword_p4/training/evaluate_frozen_human.py',
    '--model',str(root/'hey_nyx.tflite'),
    '--frozen-report','.local/wakeword/eval-v36-ami-extra-validation/report.json',
    '--features','.local/wakeword/voxconverse-development-features-v1',
    '--output','.local/wakeword/v36-voxconverse-development-v1',
    '--corpus-name','VoxConverse unscreened conversational background','--identity-unit','recording'],check=True)
subprocess.run([sys.executable,'research/wakeword_p4/training/calibrate_phone_development.py',
    '--synthetic','.local/wakeword/v36-phone-development-historical-threshold-v1',
    '.local/wakeword/v36-vctk-development-historical-threshold-v1',
    '--background','.local/wakeword/v36-voxconverse-development-v1',
    '--output','.local/wakeword/v36-reused-development-exploratory-calibration-v1.json'],check=True)
print('Exploratory diagnostics finished; reused development only, no promotion/final-data access/publication',flush=True)
