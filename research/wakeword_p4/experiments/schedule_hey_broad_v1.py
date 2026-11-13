"""Precompute120000 training +100PTQ actual tensor batches per arm, no TF."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import numpy as np

BASE=Path('.local/wakeword/hey-broad-schedules-v1')
RESULTS=Path('research/wakeword_p4/results')
PLAN=RESULTS/'4090-hey-broad-schedule-plan-v1.json'
AUDITOR=Path(__file__).with_name('audit_hey_broad_batches_v1.py')


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()


def sampler(target):
    # Reuse the exact admitted original-function extraction, data validation and
    # policy; stop before its50-batch test, keeping every original RNG seed.
    tree=ast.parse(AUDITOR.read_text())
    audit=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='audit')
    cut=next(i for i,n in enumerate(audit.body) if isinstance(n,ast.Assign) and ast.unparse(n)=='first = run()')
    audit.body[cut:]=ast.parse('''
for name, seed in [('rng',726), ('mining_rng',4726), ('augmentation_rng',10726), ('prefix_rng',20726)]:
    env[name] = np.random.default_rng(seed)
return env
''').body
    tree.body=[n for n in tree.body if not isinstance(n,ast.If)]
    env=dict(__file__=str(AUDITOR),__name__='broad_sampler_extraction')
    exec(compile(ast.fix_missing_locations(tree),str(AUDITOR),'exec'),env)
    return env['audit'](target)


def plan():
    paths=[Path(__file__),AUDITOR,Path('research/wakeword_p4/training/train_stream.py')]
    paths += [Path('research/wakeword_p4/training')/n for n in ['negative_sampling.py','spec_augment.py','ambient_sampling.py']]
    paths += [RESULTS/'4090-hey-broad-cpu-batch-audit-v1.json']
    d=dict(schema='hey-broad-schedule-plan-v1',files={str(p):sha(p) for p in paths},
        training_batches=120000,ptq_batches=100,targets=['hey_willow','hey_nyx'],
        policy='Exact admitted CPU sampler; preserve all RNGs and original draws, no augmentation/mining,16 eligible Piper/Kokoro prefix negatives, stratified ambient. SHA256 each full x/y/weights batch, first50 chain must equal original CPU admission. No model, training or heldout scoring.')
    with PLAN.open('x') as f:f.write(json.dumps(d,indent=2)+'\n')


def run():
    remote=subprocess.check_output(['git','show','origin/research-hye-nyx-voice-gpu:'+str(PLAN)])
    assert remote==PLAN.read_bytes();d=json.loads(remote)
    for p,h in d['files'].items():assert sha(p)==h
    assert not BASE.exists();BASE.mkdir()
    cpu=json.loads((RESULTS/'4090-hey-broad-cpu-batch-audit-v1.json').read_text())
    arms=[]
    for target in d['targets']:
        env=sampler(target);first50=hashlib.sha256();path=BASE/f'{target}.jsonl'
        expected=next(r['tensor_sha256'] for r in cpu['results'] if r['target']==target)
        with path.open('x') as f:
            for index in range(1,120101):
                values=env['batch'](mine=False,augment=False);h=hashlib.sha256()
                for a in values:
                    h.update(a.tobytes())
                    if index<=50:first50.update(a.tobytes())
                if index==50:assert first50.hexdigest()==expected
                f.write(json.dumps(dict(index=index,kind='train' if index<=120000 else 'ptq',tensor_sha256=h.hexdigest()),sort_keys=True)+'\n')
                if index%1000==0:
                    f.flush();print(json.dumps(dict(target=target,batches=index,total=120100)),flush=True)
        arms.append(dict(target=target,batches=120100,path=str(path),sha256=sha(path),first50_cpu_match=True))
    report=dict(schema='hey-broad-schedules-v1',plan_sha256=sha(PLAN),arms=arms,
        limits='Full tensor schedules only, not independent cross-arm source/window identity audit, model execution or recognition.')
    with (RESULTS/'4090-hey-broad-schedules-v1.json').open('x') as f:f.write(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    assert len(sys.argv)==2 and sys.argv[1] in ('plan','run')
    plan() if sys.argv[1]=='plan' else run()
