"""After existing historical evaluator exits, run the preregistered new dev gates."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

p=argparse.ArgumentParser(description=__doc__);p.add_argument('--prior-evaluator-pid',type=int,required=True)
a=p.parse_args();proc=Path(f'/proc/{a.prior_evaluator_pid}/stat')

def identity():
    try: return proc.read_text().rsplit(')',1)[1].split()[19]
    except FileNotFoundError: return None

original=identity()
print(json.dumps({'waiting_prior_evaluator_pid':a.prior_evaluator_pid,'starttime':original}),flush=True)
while original is not None and identity()==original: time.sleep(10)
root=Path('.local/wakeword/model-gpu-v35-ami-extra')
training=json.loads((root/'training.json').read_text())
model=root/'hey_nyx.tflite';model_sha=hashlib.sha256(model.read_bytes()).hexdigest()
assert training['steps']==120000 and training['model_sha256']==model_sha
kok=Path('.local/wakeword/v35-phone-development-historical-threshold-v1')
assert json.loads((kok/'report.json').read_text())['model_sha256']==model_sha
cal=Path('.local/wakeword/eval-v35-ami-extra-validation/report.json')
assert json.loads(cal.read_text())['model_sha256']==model_sha
vctk=Path('.local/wakeword/v35-vctk-development-historical-threshold-v1')
vox=Path('.local/wakeword/v35-voxconverse-development-v1')
commands=[
 [sys.executable,'research/wakeword_p4/training/score_phone_development.py',
  '--manifest','.local/wakeword/vctk-development-replay-v1/manifest.json','--model',str(model),
  '--calibration',str(cal),'--official-reports','research/wakeword_p4/results/4090-vctk-development-official-replay-reports-v1.json','--output',str(vctk)],
 [sys.executable,'research/wakeword_p4/training/evaluate_frozen_human.py','--model',str(model),
  '--frozen-report',str(cal),'--features','.local/wakeword/voxconverse-development-features-v1',
  '--output',str(vox),'--corpus-name','VoxConverse unscreened conversational background','--identity-unit','recording']]
with ThreadPoolExecutor(2) as pool:
    jobs=[pool.submit(subprocess.run,command,check=True) for command in commands]
    for job in jobs: job.result()
subprocess.run([sys.executable,'research/wakeword_p4/training/calibrate_phone_development.py',
 '--synthetic',str(kok),str(vctk),'--background',str(vox),
 '--output','.local/wakeword/v35-new-development-calibration-v1.json'],check=True)
print('New-development diagnostics completed; review before final data access or candidate promotion',flush=True)
