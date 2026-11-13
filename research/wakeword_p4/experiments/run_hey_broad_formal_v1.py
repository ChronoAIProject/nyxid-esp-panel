"""Locked broad225 fixed-final120k execution; needs completed pushed lock."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
from train_hey_broad_adapter_v1 import TRAINER, adapted_tree, admit_args, load_initial
from hey_batch_schedule_v1 import BatchGuard


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8<<20), b''):h.update(b)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--training-lock',type=Path,required=True)
    p.add_argument('--target',choices=['hey_willow','hey_nyx'],required=True)
    a=p.parse_args()
    relative=a.training_lock.resolve().relative_to(Path.cwd())
    remote=subprocess.check_output(['git','show','origin/research-hye-nyx-voice-gpu:'+str(relative)])
    assert remote==a.training_lock.read_bytes();lock=json.loads(remote)
    assert lock['schema']=='hey-broad-formal-training-lock-v1' and lock['steps']==120000
    assert lock['checkpoint_policy']=='fixed-final120000'
    for path,h in lock['files'].items():assert sha(path)==h,path
    for name,v in lock['versions'].items():assert importlib.metadata.version(name)==v,name
    # Every original training helper, upstream source and adapter must be bound.
    code=list(TRAINER.parent.glob('*.py'))+list(Path(lock['upstream']).rglob('*.py'))
    code += [Path(__file__),Path(__file__).with_name('train_hey_broad_adapter_v1.py'),
             Path(__file__).with_name('hey_batch_schedule_v1.py'),Path(__file__).with_name('audit_hey_training_batch_v1.py')]
    for path in code:assert lock['files'].get(str(path))==sha(path),str(path)
    arm=lock['arms'][a.target];root=Path(arm['features']);output=Path(arm['output'])
    prov=json.loads((root/'target-provenance.json').read_text())
    assert prov['schema']=='hey-broad-context225-target-pack-v1' and prov['target_id']==a.target and prov['context_frames']==225
    for name,h in prov['files'].items():assert sha(root/name)==h
    for path in [root/'target-provenance.json',Path(arm['schedule']),Path(lock['initial_weights'])]:
        assert lock['files'].get(str(path))==sha(path)
    # All actual ambient arrays, not just their row index, must be bound.
    ambient=[]
    for name in ('human.json','ambient.json'):
        ambient+=json.loads((root/name).read_text())
    for row in ambient:
        assert row['split']=='training'
        path=root/row['path'];assert lock['files'].get(str(path))==sha(path)
    rows=json.loads((root/'features.json').read_text())
    assert all(r['split']=='training' and r['label']==int(r['target_id']==a.target) for r in rows)
    guard=BatchGuard(Path(arm['schedule']),output,train_steps=120000,ptq_batches=100)
    captured=[]
    def initial(model):
        loaded=load_initial(model)
        assert loaded==lock['initial_weights_sha256']
        model.save_weights(output/'actual-initial.weights.h5')
        inventory=[dict(shape=list(v.shape),dtype=str(v.numpy().dtype),sha256=hashlib.sha256(v.numpy().tobytes()).hexdigest()) for v in model.variables]
        (output/'actual-initial-inventory.json').write_text(json.dumps(inventory,indent=2)+'\n')
        captured.append(model)
    def admit(args):
        admit_args(args)
        assert args.steps==120000 and args.eval_interval==5000 and args.device=='cuda'
    sys.path.insert(0,str(TRAINER.parent))
    sys.argv=[str(TRAINER),'--upstream',lock['upstream'],'--features',str(root),'--output',str(output),
        '--steps','120000','--eval-interval','5000','--architecture','teacher60','--context-frames','225',
        '--batchnorm-policy','frozen','--device','cuda','--prefix-negatives','16',
        '--source-exponent','0','--ambient-sampling','stratified']
    exec(compile(adapted_tree(),str(TRAINER),'exec'),dict(__name__='__main__',__file__=str(TRAINER),
        HEY_ADMIT_ARGS=admit,HEY_LOAD_INITIAL=initial,HEY_INITIAL_SHA=lock['initial_weights_sha256'],HEY_BATCH_GUARD=guard))
    assert len(captured)==1 and all(np.isfinite(v.numpy()).all() for v in captured[0].variables)
    assert int(captured[0].optimizer.iterations)==120000
    report=json.loads((output/'training.json').read_text())
    assert report['selected_step']==120000 and report['parameters']==24301
    assert report['batchnorm_initial']==report['batchnorm_selected']
    assert all(np.isfinite(item['loss']) for item in report['history'])
    receipt=dict(target=a.target,training_lock_sha256=sha(a.training_lock),training_report_sha256=sha(output/'training.json'),
        model_sha256=sha(output/'hey_nyx.tflite'),actual_initial_inventory_sha256=sha(output/'actual-initial-inventory.json'),
        finite_weights=True,finite_logged_loss=True,frozen_bn_exact=True,steps=120000,
        limits='Training/export completion only. Independent synthetic development and acceptance gates required before promotion.')
    with (output/'formal-execution-receipt.json').open('x') as f:f.write(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt),flush=True)


if __name__=='__main__':main()
