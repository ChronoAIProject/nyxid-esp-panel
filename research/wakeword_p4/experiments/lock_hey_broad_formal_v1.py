"""Build formal training lock only from complete admitted original artifacts."""
import argparse
import importlib.metadata
import json
from pathlib import Path
from run_hey_broad_formal_v1 import sha, TRAINER

RESULTS=Path('research/wakeword_p4/results')
DEST=RESULTS/'4090-hey-broad-formal-training-lock-v1.json'


def build():
    schedule_path=RESULTS/'4090-hey-broad-schedules-v1.json'
    if not schedule_path.exists():
        raise RuntimeError('Original full schedule task not complete; do not launch training')
    schedule=json.loads(schedule_path.read_text())
    assert schedule['schema']=='hey-broad-schedules-v1'
    plan=RESULTS/'4090-hey-broad-schedule-plan-v1.json'
    assert sha(plan)==schedule['plan_sha256']
    for path,h in json.loads(plan.read_text())['files'].items():assert sha(path)==h
    files={}
    def bind(path):
        path=Path(path);files[str(path)]=sha(path)
    sources=list(TRAINER.parent.glob('*.py'))
    upstream=Path('.local/wakeword/microWakeWord')
    sources+=list(upstream.rglob('*.py'))
    # Includes transitive experiment imports; original records preserve old pins.
    sources+=list(Path(__file__).parent.glob('*.py'))
    for path in sources:bind(path)
    reports=['4090-hey-broad-context225-initialization-v1.json','4090-hey-broad-cpu-batch-audit-v1.json',
        '4090-hey-broad-gpu-smokes-v1.json','4090-hey-broad-int8-probe-plan-v1.json',
        '4090-hey-broad-int8-probe-v1.json','4090-hey-broad-source-pairing-audit-v1.json',
        '4090-hey-broad-feature-independent-audit-v1.json','4090-hey-broad-features-v1.json',
        '4090-hey-broad-render-independent-audit-v1.json','4090-hey-broad-schedule-plan-v1.json',
        '4090-hey-broad-schedules-v1.json']
    for name in reports:bind(RESULTS/name)
    init=json.loads((RESULTS/reports[0]).read_text())
    assert init['context_frames']==225 and init['parameters']==24301
    for entry in (init['snapshot'],init['variable_inventory']):
        assert sha(entry['path'])==entry['sha256'];bind(entry['path'])
    smoke=json.loads((RESULTS/'4090-hey-broad-gpu-smokes-v1.json').read_text())
    assert smoke['both_exit0'] and len(smoke['arms'])==2
    for arm in smoke['arms']:
        root=Path('.local/wakeword/hey-broad-smoke-v1')/arm['target']
        assert arm['finite_logged_loss'] and arm['frozen_bn_exact'] and arm['batches']['first50_cpu_tensor_match']
        for name,h in arm['files'].items():assert sha(root/name)==h;bind(root/name)
    probe=json.loads((RESULTS/'4090-hey-broad-int8-probe-v1.json').read_text())
    assert len(probe['cases'])==12
    for case in probe['cases']:
        assert case['twice_fresh_reset_exact'] and sha(case['arrays_path'])==case['arrays_sha256']
        bind(case['arrays_path'])
    arms={}
    for target in ('hey_willow','hey_nyx'):
        row=next(x for x in schedule['arms'] if x['target']==target)
        assert row['batches']==120100 and row['first50_cpu_match'] and sha(row['path'])==row['sha256']
        lines=[json.loads(x) for x in Path(row['path']).read_text().splitlines()]
        assert len(lines)==120100
        for i,line in enumerate(lines,1):
            assert line['index']==i and line['kind']==('train' if i<=120000 else 'ptq')
            assert len(line['tensor_sha256'])==64
        bind(row['path'])
        root=Path('.local/wakeword/hey-broad-training-features-v1')/target
        bind(root/'target-provenance.json')
        prov=json.loads((root/'target-provenance.json').read_text())
        assert prov['target_id']==target and prov['context_frames']==225 and prov['training_only']
        for name,h in prov['files'].items():assert sha(root/name)==h;bind(root/name)
        for name in ('human.json','ambient.json'):
            for r in json.loads((root/name).read_text()):
                assert r['split']=='training';bind(root/r['path'])
        arms[target]=dict(features=str(root),schedule=row['path'],output=str(Path('.local/wakeword/hey-broad-formal-v1')/target))
    return dict(schema='hey-broad-formal-training-lock-v1',steps=120000,checkpoint_policy='fixed-final120000',
        upstream=str(upstream),initial_weights=init['snapshot']['path'],initial_weights_sha256=init['snapshot']['sha256'],
        files=files,arms=arms,versions={n:importlib.metadata.version(n) for n in ['tensorflow','numpy','pymicro-features','soundfile','ai-edge-litert']},
        optimizer='Original Adam1e-3 global_clipnorm5,3e-4 at50%,1e-4 at80%; no development checkpoint selection',
        sampler='batch128,32positive/48confusing/48ambient;16prefix Piper/Kokoro; equal engines; stratified ambient; seeds726/4726/10726/20726',
        scope='Formal training/export only; no heldout access/promotion. Frozenv30 retained until independent synthetic gates pass.')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--write',action='store_true');a=p.parse_args()
    result=build()
    if a.write:
        with DEST.open('x') as f:f.write(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(admitted=True,files=len(result['files']),written=a.write)))
