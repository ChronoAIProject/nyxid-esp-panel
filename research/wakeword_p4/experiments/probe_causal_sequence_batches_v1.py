"""Preserve first two actual whole-sequence CPU batches, no model calls."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import numpy as np
from causal_sequence_batch_v1 import Sampler,digest
R=Path('research/wakeword_p4/results');LOCK=R/'4090-causal-sequence-batch-plan-v1.json';OUT=Path('.local/wakeword/causal-sequence-batch-v1')
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def plan():
    source=R/'4090-causal-onset-material-original-v1.json';original=json.loads(source.read_text())
    parent=R/'4090-causal-onset-material-plan-v1.json';p=json.loads(parent.read_text())
    assert original['plan_sha256']==sha(parent) and original['quality_checks_passed'] and original['presentations']==51840
    files=dict(p['files'])
    paths=[Path(__file__),Path(__file__).with_name('causal_sequence_batch_v1.py'),Path('research/wakeword_p4/training/ambient_sampling.py'),source,parent,Path(original['records_path']),Path(original['feature_binary_path'])]
    assert sha(original['records_path'])==original['records_sha256'] and sha(original['feature_binary_path'])==original['feature_binary_sha256']
    files.update({str(path):sha(path) for path in paths})
    with LOCK.open('x') as f:f.write(json.dumps(dict(schema='causal-sequence-batch-plan-v1',files=files,versions={n:importlib.metadata.version(n) for n in ['numpy']},seed=726,batches=2,expected_new_gib=.1,policy='Two first actualCPU128x750x40 whole-sequence batches, regionnormalized positive/negative masks, rightfuturepadding unsupervised. All training sources, no model/audio inference, fit or threshold selection.'),indent=2)+'\n')
    print(json.dumps(dict(plan_sha256=sha(LOCK))))
def run():
    assert subprocess.check_output(['git','show','origin/research-hye-nyx-voice-gpu:'+str(LOCK)])==LOCK.read_bytes()
    p=json.loads(LOCK.read_text())
    for path,h in p['files'].items():assert sha(path)==h,path
    for package,v in p['versions'].items():assert importlib.metadata.version(package)==v
    subprocess.run(['/usr/bin/python3','research/wakeword_p4/tools/check_disk_budget.py','--expected-gib','0.1'],check=True)
    OUT.mkdir(exist_ok=False);sampler=Sampler(p['seed']);rows=[]
    for step in [1,2]:
        (x,pw,nw),descriptors=sampler.batch();path=OUT/f'batch-{step}.npz'
        np.savez_compressed(path,x=x,positive_weights=pw,negative_weights=nw)
        rows.append(dict(step=step,path=str(path),sha256=sha(path),x_sha256=digest(x),positive_weights_sha256=digest(pw),negative_weights_sha256=digest(nw),shape=list(x.shape),positive_mass=float(pw.sum()),negative_mass=float(nw.sum()),descriptors=descriptors))
    result=dict(schema='causal-sequence-batch-original-v1',plan_sha256=sha(LOCK),batches=rows,model_calls=0,optimizer_updates=0)
    with (R/'4090-causal-sequence-batch-original-v1.json').open('x') as f:f.write(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(batches=2,shape=[128,750,40],positive_mass=32,negative_mass=128,model_calls=0)))
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('action',choices=['plan','run']);args=a.parse_args();plan() if args.action=='plan' else run()
