"""Two real RTX4090 causal training updates; no score-comparison experiment."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import traceback
import numpy as np
from causal_sequence_batch_v1 import digest
R=Path('research/wakeword_p4/results');LOCK=R/'4090-causal-sequence-gpu-smoke-plan-v1.json';OUT=Path('.local/wakeword/causal-sequence-gpu-smoke-v1')
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def plan():
    batches=R/'4090-causal-sequence-batch-original-v1.json';source=R/'4090-causal-sequence-batch-plan-v1.json'
    b=json.loads(batches.read_text());parent=json.loads(source.read_text());assert b['plan_sha256']==sha(source) and b['model_calls']==0 and len(b['batches'])==2
    initial_path=R/'4090-multispeaker-context309-initialization-v1.json';initial=json.loads(initial_path.read_text())
    files=dict(parent['files']);paths=[Path(__file__),batches,source,initial_path,Path(__file__).with_name('hey_broad_gpu_training_factory_v1.py'),Path(__file__).with_name('probe_hey_context_transfer_shapes_v1.py'),Path('research/wakeword_p4/training/gpu_python.py')]
    paths+=list(Path('.local/wakeword/microWakeWord/microwakeword').rglob('*.py'))
    for row in b['batches']:assert sha(row['path'])==row['sha256'];paths.append(Path(row['path']))
    files.update({str(p):sha(p) for p in paths})
    p=dict(schema='causal-sequence-gpu-smoke-plan-v1',files=files,versions={n:importlib.metadata.version(n) for n in ['tensorflow','keras','numpy']},batches=b['batches'],initial=initial,seed=726,optimizer=dict(learning_rate=1e-5,beta_1=.9,beta_2=.999,epsilon=1e-7),expected_new_gib=.05,steps=2,policy='Original firsttwoactualCPUwhole-sequence batches, same preservedinitial31variables,15trainables/16frozenBN, freshAdam32vars/30zeroslots. Causalperlayerhistories and perregion normalized BCE; GPU4090 required/FP32/deterministic/TF32off. Save initial/final states; no crossdevice gradient/logit equality or float/runtime comparison, formal training/recognition/freshdata. Disposablefixedfinal2, never formalinitializer.')
    with LOCK.open('x') as f:f.write(json.dumps(p,indent=2)+'\n')
    print(json.dumps(dict(plan_sha256=sha(LOCK))))
def run():
    assert subprocess.check_output(['git','show','origin/research-hye-nyx-voice-gpu:'+str(LOCK)])==LOCK.read_bytes();p=json.loads(LOCK.read_text())
    for path,h in p['files'].items():assert sha(path)==h,path
    for package,v in p['versions'].items():assert importlib.metadata.version(package)==v
    subprocess.run(['/usr/bin/python3','research/wakeword_p4/tools/check_disk_budget.py','--expected-gib','.05'],check=True)
    OUT.mkdir(exist_ok=False)
    launch=dict(pid=os.getpid(),proc_starttime=Path('/proc/self/stat').read_text().split()[21],plan_sha256=sha(LOCK))
    (OUT/'launch.json').write_text(json.dumps(launch,indent=2)+'\n');print(json.dumps(launch),flush=True)
    try:
        import hey_broad_gpu_training_factory_v1 as factory
        from causal_fullsequence_graph_v1 import logits,presentation_loss
        tf=factory.tf;tf.config.experimental.enable_tensor_float_32_execution(False);tf.keras.utils.set_random_seed(p['seed'])
        model=factory.create(309);model.load_weights(p['initial']['snapshot']['path'])
        for layer in model.layers:
            cell=getattr(layer,'cell',layer)
            if isinstance(cell,tf.keras.layers.BatchNormalization):layer.trainable=cell.trainable=False
        assert len(model.variables)==31 and len(model.trainable_variables)==15 and len(model.non_trainable_variables)==16
        for v,e in zip(model.variables,p['initial']['variable_inventory']['variables'],strict=True):assert digest(v.numpy())==e['sha256']
        frozen=[v.numpy().copy() for v in model.non_trainable_variables]
        optimizer=tf.keras.optimizers.Adam(**p['optimizer']);optimizer.build(model.trainable_variables)
        assert len(optimizer.variables)==32 and int(optimizer.iterations.numpy())==0
        assert all(np.all(v.numpy()==0) for v in optimizer.variables[2:])
        def save(prefix):
            for kind,variables in [('model',model.variables),('optimizer',optimizer.variables)]:
                np.savez(OUT/(prefix+'-'+kind+'-variables.npz'),**{f'v{i:03}':v.numpy() for i,v in enumerate(variables)})
        save('initial');steps=[]
        for request in p['batches']:
            with np.load(request['path'],allow_pickle=False) as b:
                x,pw,nw=[b[k].copy() for k in ['x','positive_weights','negative_weights']]
            assert digest(x)==request['x_sha256'] and digest(pw)==request['positive_weights_sha256'] and digest(nw)==request['negative_weights_sha256']
            with tf.device('/GPU:0'):
                with tf.GradientTape() as tape:
                    z=logits(model,tf.constant(x),tf);assert tuple(z.shape)==(128,250,1)
                    loss=presentation_loss(z,tf.constant(pw),tf.constant(nw),tf)
                gradients=tape.gradient(loss,model.trainable_variables)
                assert 'GPU:0' in z.device and all(g is not None for g in gradients)
                tf.debugging.assert_all_finite(loss,'loss')
                for g in gradients:tf.debugging.assert_all_finite(g,'gradient')
                optimizer.apply_gradients(zip(gradients,model.trainable_variables,strict=True))
            assert all(np.isfinite(v.numpy()).all() for v in model.variables)
            assert all(np.array_equal(a,v.numpy()) for a,v in zip(frozen,model.non_trainable_variables,strict=True))
            row=dict(step=request['step'],loss=float(loss.numpy()),logits_shape=list(z.shape),device=z.device,optimizer_iteration=int(optimizer.iterations.numpy()),all15_gradients_finite=True,frozen16_unchanged=True,x_sha256=request['x_sha256']);steps.append(row);print(json.dumps(row),flush=True)
        save('final');model.save_weights(OUT/'final.weights.h5')
        assert int(optimizer.iterations.numpy())==2
        result=dict(schema='causal-sequence-gpu-smoke-original-v1',plan_sha256=sha(LOCK),launch=launch,device=factory.identity,steps=steps,files={f.name:sha(f) for f in OUT.iterdir()},all_update_checks_passed=True,formal_training_started=False,recognition_scored=False)
        with (R/'4090-causal-sequence-gpu-smoke-original-v1.json').open('x') as f:f.write(json.dumps(result,indent=2)+'\n')
    except Exception:
        (OUT/'failure.log').write_text(traceback.format_exc());raise
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('action',choices=['plan','run']);args=a.parse_args();plan() if args.action=='plan' else run()
