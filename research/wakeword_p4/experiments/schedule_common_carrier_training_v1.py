"""Freeze actual paired tensor schedules only after complete material audit."""
import argparse,hashlib,importlib.metadata,json,subprocess
from pathlib import Path

R=Path('research/wakeword_p4/results');BASE=Path('.local/wakeword');OUT=BASE/'hey-common-carrier-training-schedules-v1'
PLAN=R/'4090-hey-common-carrier-training-schedule-execution-v1.json'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def plan(check_only=False):
    needed=[R/'4090-hey-common-carrier-feature-result-v1.json',R/'4090-hey-common-carrier-feature-independent-audit-v1.json']
    for path in needed:
        if not path.exists():print(json.dumps(dict(status='not_ready',missing=str(path))));return 2
    result,audit=[json.loads(p.read_text()) for p in needed]
    assert audit['all_passed'] and audit['all_rows_independently_recomputed'] and audit['original_result_sha256']==sha(needed[0])
    if check_only:print(json.dumps(dict(status='ready',conditions=4,batches_per_condition=10000)));return 0
    root=BASE/'hey-common-carrier-features-v1';assert sha(root/'features.bin')==result['binary_sha256'] and sha(root/'features.json')==result['metadata_sha256']
    paths=[*needed,Path(__file__),root/'features.bin',root/'features.json',R/'4090-hey-common-carrier-replacement-decisions-v1.json',R/'4090-hey-common-carrier-raw-recognition-protocol-v1.json']
    decisions=json.loads((R/'4090-hey-common-carrier-replacement-decisions-v1.json').read_text());paths += [Path(t['path']) for t in decisions['targets']]
    for name in ['common_carrier_training_batches_v1.py','common_carrier_slot_policy_v1.py','schedule_hey_broad_v1.py','audit_hey_broad_batches_v1.py','hey_batch_schedule_v1.py']:paths.append(Path(__file__).with_name(name))
    for target in ('hey_willow','hey_nyx'):paths.append(BASE/'hey-broad-schedules-v1'/(target+'.jsonl'))
    p=dict(files={str(f):sha(f) for f in paths},numpy_version=importlib.metadata.version('numpy'),targets=['hey_willow','hey_nyx'],conditions=['control','command_context'],batches=10000,
           policy='All4 full10000 x/y/weight tensor schedules. Every baseline batch SHA matches original schedule. Original nonselected rows/all labels+weights exact, every oldeligible row matches original slice, replacement map uses frozen decisions, no new RNG/mining/augmentation. Compare first2 complete batches with twofresh CPU readers per condition. Save every endpoint/slot decision and both SHA. No model/optimizer/audio/new data selection.')
    with PLAN.open('x') as f:f.write(json.dumps(p,indent=2)+'\n')
    return 0

def run():
    assert subprocess.check_output(['git','show','origin/research-hye-nyx-voice-gpu:'+str(PLAN)])==PLAN.read_bytes();p=json.loads(PLAN.read_text())
    for f,h in p['files'].items():assert sha(f)==h
    assert importlib.metadata.version('numpy')==p['numpy_version']
    from common_carrier_training_batches_v1 import Batches
    OUT.mkdir(exist_ok=False);reports=[]
    for target in p['targets']:
        for condition in p['conditions']:
            reference=Batches(target,condition);first=[]
            for _ in range(2):
                values,receipt=reference.next();first.append((tuple(v.tobytes() for v in values),receipt))
            reference.log.close();del reference
            reader=Batches(target,condition);dest=OUT/(target+'-'+condition+'.jsonl');changed=0
            with dest.open('x') as log:
                for step in range(1,10001):
                    values,receipt=reader.next();assert receipt['step']==step
                    if step<=2:assert tuple(v.tobytes() for v in values)==first[step-1][0] and receipt==first[step-1][1]
                    if condition=='control':assert receipt['tensor_sha256']==receipt['original_batch_sha256']
                    changed+=receipt['tensor_sha256']!=receipt['original_batch_sha256']
                    log.write(json.dumps(receipt,separators=(',',':'))+'\n')
                    if step%1000==0:log.flush();print(json.dumps(dict(target=target,condition=condition,step=step)),flush=True)
            final=reader.finish();reports.append(dict(target=target,condition=condition,path=str(dest),sha256=sha(dest),first2_fresh_readers_exact=True,changed_batches=changed,**final))
    result=dict(execution_sha256=sha(PLAN),conditions=reports,all_passed=True,model_invocations=0,limits=p['policy'])
    with (R/'4090-hey-common-carrier-training-schedules-v1.json').open('x') as f:f.write(json.dumps(result,indent=2)+'\n')

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['plan','run']);ap.add_argument('--check-only',action='store_true');a=ap.parse_args();raise SystemExit(plan(a.check_only) if a.action=='plan' else run())
