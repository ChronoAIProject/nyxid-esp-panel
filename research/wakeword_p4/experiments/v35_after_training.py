"""Wait for the identified existing v35 process, then evaluate completed output."""
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
root=Path('.local/wakeword/model-gpu-v35-ami-extra')
training=json.loads((root/'training.json').read_text())
if (training['steps']!=120000 or training['effective_training_batch_size']!=144
        or hashlib.sha256((root/'hey_nyx.tflite').read_bytes()).hexdigest()!=training['model_sha256']):
    raise RuntimeError('Training process ended without complete authenticated v35 output')
print('Authenticated complete v35; starting historical regression',flush=True)
subprocess.run(['bash','research/wakeword_p4/experiments/v35_historical_regression.sh'],check=True)
subprocess.run([sys.executable,'research/wakeword_p4/training/score_phone_development.py',
    '--manifest','.local/wakeword/synthetic-phone-development-replay-v1/manifest.json',
    '--model',str(root/'hey_nyx.tflite'),
    '--calibration','.local/wakeword/eval-v35-ami-extra-validation/report.json',
    '--official-reports','research/wakeword_p4/results/4090-phone-development-official-replay-reports-v1.json',
    '--output','.local/wakeword/v35-phone-development-historical-threshold-v1'],check=True)
print('Post-training diagnostics finished; review results before publication or model selection',flush=True)
