"""Paired raw-FP32 recognition continuation; gated, never auto-restarts."""
import argparse,hashlib,importlib.metadata,json,subprocess,time,traceback
from pathlib import Path
import numpy as np

R=Path('research/wakeword_p4/results')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def signature(variables):return hashlib.sha256(b''.join(v.numpy().tobytes() for v in variables)).hexdigest()
def inventory(variables):return [dict(shape=list(v.shape),dtype=str(v.numpy().dtype),sha256=hashlib.sha256(v.numpy().tobytes()).hexdigest()) for v in variables]
def execution(mode):return R/f'4090-hey-common-carrier-raw-{mode}-execution-v1.json'
def plan(mode,check_only=False):
    required=[R/'4090-hey-common-carrier-training-schedules-v1.json',R/'4090-hey-common-carrier-training-schedule-independent-audit-v1.json',R/'4090-learned-grid-original-terminal-receipt-v1.json']
    if mode=='formal':required += [R/'4090-hey-common-carrier-raw-smoke-result-v1.json',R/'4090-hey-common-carrier-raw-smoke-independent-audit-v1.json']
    for p in required:
        if not p.exists():print(json.dumps(dict(status='not_ready',missing=str(p))));return 2
    schedules,audit,terminal=[json.loads(p.read_text()) for p in required[:3]]
    assert schedules['all_passed'] and audit['all_passed'] and audit['original_result_sha256']==sha(required[0])
    assert terminal['pid']==1589700 and str(terminal['starttime'])=='24217011' and terminal['session']==77025 and terminal['terminal_verified'] and isinstance(terminal['exit_code'],int)
    if mode=='formal':
        smoke,smoke_audit=[json.loads(p.read_text()) for p in required[3:]]
        assert smoke['all_passed'] and smoke_audit['all_passed'] and smoke_audit['original_result_sha256']==sha(required[3])
    if check_only:print(json.dumps(dict(status='ready',mode=mode)));return 0
    protocolpath=R/'4090-hey-common-carrier-raw-recognition-protocol-v1.json';protocol=json.loads(protocolpath.read_text())
    files=[Path(__file__),protocolpath,*required]
    for name in ['hey_broad_gpu_training_factory_v1.py','probe_hey_context_transfer_shapes_v1.py','common_carrier_training_batches_v1.py','common_carrier_slot_policy_v1.py','schedule_hey_broad_v1.py','audit_hey_broad_batches_v1.py','hey_batch_schedule_v1.py']:files.append(Path(__file__).with_name(name))
    files+=sorted(Path('.local/wakeword/microWakeWord/microwakeword').rglob('*.py'))
    files += [Path('research/wakeword_p4/training')/n for n in ('train_stream.py','negative_sampling.py','ambient_sampling.py','spec_augment.py')]
    for m in protocol['initial_models']:
        assert sha(m['initial_weights_path'])==m['initial_weights_sha256'];files.append(Path(m['initial_weights_path']))
    for entry in schedules['conditions']:files.append(Path(entry['path']))
    versions={n:importlib.metadata.version(n) for n in ('tensorflow','keras','numpy','h5py')}
    p=dict(files={str(f):sha(f) for f in files},versions=versions,mode=mode,steps=2 if mode=='smoke' else 10000,repeats=2 if mode=='smoke' else 1,
           optimizer=protocol['optimizer'],arms=protocol['arms'],initial_models=protocol['initial_models'],schedules=schedules['conditions'],
           output=f'.local/wakeword/hey-common-carrier-raw-{mode}-v1',policy='Exact prospective raw recognition protocol; all15trainables/frozenBN, original validconv225/rawweightedBCE. Fixedfinal10000 only formalrepeat1; snapshots evidence only, no dev/selection/restart. Smoke2updates x2fresh exact required; not formal repeat proof. Actual x/y/weight SHA enforced. No converter/PTQ/export or new data scoring in this runner.')
    with execution(mode).open('x') as f:f.write(json.dumps(p,indent=2)+'\n')
    return 0

def run(mode):
    path=execution(mode);assert subprocess.check_output(['git','show','origin/research-hye-nyx-voice-gpu:'+str(path)])==path.read_bytes();p=json.loads(path.read_text())
    for f,h in p['files'].items():assert sha(f)==h
    for n,v in p['versions'].items():assert importlib.metadata.version(n)==v
    proc=Path('/proc/1589700/stat')
    if proc.exists():assert proc.read_text().rsplit(')',1)[1].split()[19]!='24217011','Original learned-grid job still live; do not launch another GPU job'
    from common_carrier_training_batches_v1 import Batches
    import hey_broad_gpu_training_factory_v1 as helper
    tf=helper.tf;base=Path(p['output']);base.mkdir(exist_ok=False);cases=[];current=None
    try:
        for arm in p['arms']:
            target,condition=arm['target'],arm['condition'];source=next(m for m in p['initial_models'] if m['target']==target);schedule=next(s for s in p['schedules'] if s['target']==target and s['condition']==condition)
            wanted=[json.loads(l) for l in Path(schedule['path']).read_text().splitlines()];assert len(wanted)==10000
            repeats=[];identities=[]
            for repeat in range(p['repeats']):
                out=base/f'{target}-{condition}-{repeat}';out.mkdir();model=helper.create(225);model.load_weights(source['initial_weights_path'])
                for layer in model.layers:
                    cell=getattr(layer,'cell',layer)
                    if isinstance(cell,tf.keras.layers.BatchNormalization):layer.trainable=cell.trainable=False
                variables=model.trainable_variables;assert len(variables)==15 and all('GPU:0' in v.value.device for v in variables)
                frozen=[v.numpy().copy() for v in model.non_trainable_variables];initial=[v.numpy().copy() for v in variables]
                model.save_weights(out/'initial.weights.h5');(out/'initial-inventory.json').write_text(json.dumps(inventory(model.variables),indent=2)+'\n')
                optimizer=tf.keras.optimizers.Adam(**p['optimizer']);optimizer.build(variables)
                embedding=tf.keras.Model(model.inputs,model.layers[-1].input);head=model.layers[-1]
                @tf.function
                def step(x,y,weights):
                    with tf.GradientTape() as tape:
                        raw=tf.linalg.matmul(embedding(x,training=False),head.kernel)+head.bias
                        loss=tf.reduce_mean(tf.reshape(tf.nn.sigmoid_cross_entropy_with_logits(labels=y,logits=raw),[-1])*weights)
                    gradients=tape.gradient(loss,variables);assert all(g is not None for g in gradients)
                    tf.debugging.assert_all_finite(loss,'Nonfinite raw BCE')
                    for g in gradients:tf.debugging.assert_all_finite(g,'Nonfinite gradient')
                    optimizer.apply_gradients(zip(gradients,variables));return loss,tf.linalg.global_norm(gradients)
                reader=Batches(target,condition);started=time.monotonic();history=[];current=dict(target=target,condition=condition,repeat=repeat,step=0)
                with (out/'actual-batch-schedule.jsonl').open('x') as log:
                    for index in range(1,p['steps']+1):
                        (x,y,w),receipt=reader.next();assert receipt==wanted[index-1]
                        loss,norm=step(tf.constant(x),tf.constant(y),tf.constant(w));current['step']=index
                        assert int(optimizer.iterations)==index and all(np.isfinite(v.numpy()).all() for v in variables)
                        assert all(a.tobytes()==v.numpy().tobytes() for a,v in zip(frozen,model.non_trainable_variables,strict=True))
                        row=dict(step=index,tensor_sha256=receipt['tensor_sha256'],loss=float(loss),gradient_l2=float(norm));log.write(json.dumps(row)+'\n');log.flush()
                        if index<=2 or index%1000==0:history.append(row);print(json.dumps(dict(**arm,repeat=repeat,**row)),flush=True)
                        if mode=='formal' and index%1000==0:model.save_weights(out/f'step{index}.weights.h5');tf.train.Checkpoint(model=model,optimizer=optimizer).write(str(out/f'step{index}'))
                if mode=='formal':reader.finish()
                else:reader.log.close()
                assert any(a.tobytes()!=v.numpy().tobytes() for a,v in zip(initial,variables,strict=True))
                model.save_weights(out/'final.weights.h5');np.savez_compressed(out/'final-optimizer.npz',**{f'v{j}':v.numpy() for j,v in enumerate(optimizer.variables)})
                (out/'final-inventory.json').write_text(json.dumps(inventory(model.variables),indent=2)+'\n');identity=dict(model=signature(model.variables),optimizer=signature(optimizer.variables));identities.append(identity)
                receipt=dict(**arm,repeat=repeat,steps=p['steps'],weights_path=str(out/'final.weights.h5'),weights_sha256=sha(out/'final.weights.h5'),optimizer_path=str(out/'final-optimizer.npz'),optimizer_sha256=sha(out/'final-optimizer.npz'),batch_trace_path=str(out/'actual-batch-schedule.jsonl'),batch_trace_sha256=sha(out/'actual-batch-schedule.jsonl'),identity=identity,elapsed_seconds=time.monotonic()-started,history=history,frozen_bn_exact=True)
                (out/'training.json').write_text(json.dumps(receipt,indent=2)+'\n');repeats.append(receipt)
            if mode=='smoke':assert identities[0]==identities[1]
            cases.append(dict(**arm,repeats=repeats,repeat_exact=identities[0]==identities[1] if mode=='smoke' else None))
        result=dict(execution_sha256=sha(path),cases=cases,all_passed=True,gpu=helper.identity,limits=p['policy'])
        with (R/f'4090-hey-common-carrier-raw-{mode}-result-v1.json').open('x') as f:f.write(json.dumps(result,indent=2)+'\n')
    except Exception:
        (base/'failure.log').write_text(traceback.format_exc());(base/'failure.json').write_text(json.dumps(dict(current=current,completed_cases=cases,automatic_restart=False),indent=2)+'\n');raise

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['plan','run']);ap.add_argument('mode',choices=['smoke','formal']);ap.add_argument('--check-only',action='store_true');a=ap.parse_args();raise SystemExit(plan(a.mode,a.check_only) if a.action=='plan' else run(a.mode))
