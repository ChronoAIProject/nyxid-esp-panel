"""Authenticate original120100 batch replay and save original PTQ first rows."""
import argparse
import json
from pathlib import Path
import numpy as np
from schedule_hey_broad_v1 import sampler, sha
from hey_batch_schedule_v1 import batch_digest
from run_hey_broad_trained_quantization_v1 import ROOT, published

PLAN=ROOT/'results/4090-hey-broad-ptq-material-plan-v1.json'
RESULT=ROOT/'results/4090-hey-broad-ptq-material-result-v1.json'
OUT=Path('.local/wakeword/hey-broad-ptq-material-v1')

def plan():
    source=ROOT/'results/4090-hey-broad-formal-training-lock-v1.json'
    lock=json.loads(source.read_text())
    files=dict(lock['files'])
    for p in [source,Path(__file__),Path(__file__).with_name('schedule_hey_broad_v1.py'),Path(__file__).with_name('hey_batch_schedule_v1.py'),Path(__file__).with_name('run_hey_broad_trained_quantization_v1.py')]:files[str(p)]=sha(p)
    record=dict(schema='hey-broad-ptq-material-plan-v1',files=files,arms=lock['arms'],policy='CPU only, no TensorFlow or model. Replay original sampler seeds and all120100 full x/y/weight tensor hashes per arm against original schedule, stop on mismatch. Save100 actual final PTQ x[:1] arrays in original order. These are precisely the original calibration sources, not selected witness substitutes. Original zero/full26 prefixes and contiguous75slices/source, no per-source state reset, will remain in a future separately registered controlled export. No export authorized here.',limits='Material authentication only. No new training, candidate export, inference, development, acceptance or threshold changes.')
    with PLAN.open('x') as f:f.write(json.dumps(record,indent=2)+'\n')

def run():
    published(PLAN);lock=json.loads(PLAN.read_text())
    for p,h in lock['files'].items():assert sha(p)==h,p
    OUT.mkdir(exist_ok=False)
    arms=[]
    for target,arm in lock['arms'].items():
        env=sampler(target);saved=[];count=0
        with Path(arm['schedule']).open() as schedule:
            for index,line in enumerate(schedule,1):
                expected=json.loads(line);values=env['batch'](mine=False,augment=False)
                assert expected['index']==index and expected['kind']==('train' if index<=120000 else 'ptq')
                assert batch_digest(*values)==expected['tensor_sha256'],(target,index)
                if index>120000:saved.append(values[0][:1].copy())
                count=index
                if index%1000==0:print(json.dumps(dict(target=target,verified=index)),flush=True)
        assert count==120100 and len(saved)==100
        path=OUT/f'{target}.npy';np.save(path,np.stack(saved),allow_pickle=False)
        assert np.load(path,mmap_mode='r').shape==(100,1,225,40)
        row=dict(target=target,verified_batches=count,path=str(path),sha256=sha(path),shape=[100,1,225,40],schedule_sha256=sha(arm['schedule']))
        arms.append(row)
        with (OUT/'arms.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
    with RESULT.open('x') as f:f.write(json.dumps(dict(schema='hey-broad-ptq-material-result-v1',plan_sha256=sha(PLAN),arms=arms,limits=lock['limits']),indent=2)+'\n')

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=['plan','run']);args=parser.parse_args()
    plan() if args.action=='plan' else run()
