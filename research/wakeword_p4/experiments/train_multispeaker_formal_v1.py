"""Single fixed-final paired 20k speaker-coverage experiment; no holdout use."""
import argparse
import gzip
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import time
import traceback
import numpy as np

R=Path('research/wakeword_p4/results')
LOCK=R/'4090-multispeaker-formal-execution-v1.json'
OUT=Path('.local/wakeword/multispeaker-formal-v1')
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def plan():
    smoke=R/'4090-multispeaker-gpu-smoke-plan-v1.json'
    export=R/'4090-multispeaker-smoke-export-independent-audit-v1.json'
    assert json.loads(export.read_text())['all_checks_passed']
    initial=json.loads((R/'4090-multispeaker-context309-initialization-v1.json').read_text())
    paths={Path(__file__),smoke,export,R/'4090-multispeaker-smoke-export-original-v1.json',
           R/'4090-multispeaker-smoke-export-runtime-original-v1.json',
           R/'4090-multispeaker-smoke-export-execution-exit-v1.json',
           R/'4090-multispeaker-features-independent-audit-v1.json',
           Path(__file__).with_name('multispeaker_batch_v1.py'),
           Path(__file__).with_name('multispeaker_feature_training_contract_v1.md'),
           Path(__file__).resolve().parents[1]/'training/ambient_sampling.py',
           Path('research/wakeword_p4/tools/check_disk_budget.py')}
    paths.update(Path(p) for p in json.loads(smoke.read_text())['files'])
    # Pin every actual feature input, including hardlinked original ambient arrays.
    for arm in ['narrow','broad']:
        root=Path('.local/wakeword/multispeaker-coverage-features-v1')/arm
        paths.update(root/n for n in ['features.bin','features.json','human.json','ambient.json'])
        for name in ['human.json','ambient.json']:
            paths.update(root/r['path'] for r in json.loads((root/name).read_text()))
    p=dict(schema='multispeaker-formal-execution-v1',files={str(p):sha(p) for p in sorted(paths)},
           versions={n:importlib.metadata.version(n) for n in ['tensorflow','keras','numpy','ai-edge-litert']},
           initial=initial,arms=['narrow','broad'],candidate='broad',control='narrow',steps=20000,seed=726,
           context=309,expected_new_gib=1,optimizer=dict(learning_rate=1e-5,beta_1=.9,beta_2=.999,epsilon=1e-7),
           recipe='Batch128:32wholepositive+48syntheticnegative+48trainingambient; raw-logit sigmoid crossentropy weighted mean1/20. All15trainables;16frozenBN. Same preserved CPU initialization/freshAdam32vars/30zero slots eacharm. FP32 deterministic RTX4090, TF32disabled, no JIT. No masks/mining/prefix negatives/domain loss. Canonical paired sampler seed726.',
           retention='gzip ordered descriptors and x/y/weights SHA every update; initial/final model+optimizer arrays, final H5, progress every100. No intermediate checkpoint selection; no restart/resume/retry.',
           export_policy='After independent final-state audit: one unchanged raw-sigmoid FP32 and builtinINT8 export per fixedfinal20000 model; same fixedfirst16actualtrainingwindows/arm and zero26 sentinels as smoke; six real-zero states, no intermediate calibration resets. Whole-wave streaming checks and consumed regression separate before fresh development. No calibration/seed/LR/checkpoint rescues.',
           limits='Broad sole candidate, narrow diagnostic only. Formal training completion is not recognition or promotion. Freshdev/acceptance, v30/officialHiESP samePCM and FARupper95<=.5/h gates remain required. No protected data consumed.')
    with LOCK.open('x') as f:f.write(json.dumps(p,indent=2)+'\n')
    print(json.dumps(dict(plan_sha256=sha(LOCK),files=len(paths),steps_per_arm=20000)))

def run():
    assert subprocess.check_output(['git','show','origin/research-hye-nyx-voice-gpu:'+str(LOCK)])==LOCK.read_bytes()
    p=json.loads(LOCK.read_text())
    for path,h in p['files'].items():assert sha(path)==h,path
    for package,v in p['versions'].items():assert importlib.metadata.version(package)==v
    subprocess.run(['/usr/bin/python3','research/wakeword_p4/tools/check_disk_budget.py','--expected-gib',str(p['expected_new_gib'])],check=True)
    import hey_broad_gpu_training_factory_v1 as factory
    from multispeaker_batch_v1 import Sampler,digest
    tf=factory.tf;tf.config.experimental.enable_tensor_float_32_execution(False)
    OUT.mkdir(exist_ok=False)
    live=dict(pid=__import__('os').getpid(),proc_starttime=Path('/proc/self/stat').read_text().split()[21],plan_sha256=sha(LOCK),device=factory.identity)
    (OUT/'launch.json').write_text(json.dumps(live,indent=2)+'\n');print(json.dumps(live),flush=True)
    completed=[]
    try:
        for arm in p['arms']:
            tf.keras.backend.clear_session();tf.keras.utils.set_random_seed(p['seed'])
            root=OUT/arm;root.mkdir()
            model=factory.create(p['context']);model.load_weights(p['initial']['snapshot']['path'])
            for layer in model.layers:
                cell=getattr(layer,'cell',layer)
                if isinstance(cell,tf.keras.layers.BatchNormalization):layer.trainable=cell.trainable=False
            assert len(model.trainable_variables)==15 and len(model.non_trainable_variables)==16
            for v,e in zip(model.variables,p['initial']['variable_inventory']['variables'],strict=True):assert digest(v.numpy())==e['sha256']
            frozen=[v.numpy().copy() for v in model.non_trainable_variables]
            optimizer=tf.keras.optimizers.Adam(**p['optimizer']);optimizer.build(model.trainable_variables)
            assert len(optimizer.variables)==32 and int(optimizer.iterations.numpy())==0
            assert all(np.all(v.numpy()==0) for v in optimizer.variables[2:])
            def save_arrays(prefix):
                for label,vs in [('model',model.variables),('optimizer',optimizer.variables)]:
                    np.savez(root/(prefix+'-'+label+'-variables.npz'),**{f'v{i:03}':v.numpy() for i,v in enumerate(vs)})
            save_arrays('initial')
            head=model.layers[-1];assert head.activation.__name__=='sigmoid'
            embedding=tf.keras.Model(model.inputs,head.input)
            @tf.function(jit_compile=False)
            def update(x,y,w):
                with tf.device('/GPU:0'):
                    with tf.GradientTape() as tape:
                        logits=tf.linalg.matmul(embedding(x,training=False),head.kernel)+head.bias
                        loss=tf.reduce_mean(tf.reshape(tf.nn.sigmoid_cross_entropy_with_logits(labels=y,logits=logits),[-1])*w)
                    gs=tape.gradient(loss,model.trainable_variables)
                    assert all(g is not None for g in gs)
                    tf.debugging.assert_all_finite(loss,'loss')
                    for g in gs:tf.debugging.assert_all_finite(g,'gradient')
                    optimizer.apply_gradients(zip(gs,model.trainable_variables,strict=True))
                    return loss,tf.constant(logits.device)
            sampler=Sampler(arm,p['seed']);start=time.monotonic()
            with gzip.open(root/'ordered-batches.jsonl.gz','xt',compresslevel=3) as schedule:
                for step in range(1,p['steps']+1):
                    (x,y,w),descriptors=sampler.batch()
                    loss,device=update(x,y,w);value=float(loss.numpy())
                    assert np.isfinite(value) and 'GPU:0' in device.numpy().decode()
                    schedule.write(json.dumps(dict(step=step,descriptors=descriptors,x_sha256=digest(x),y_sha256=digest(y),weights_sha256=digest(w),loss=value),separators=(',',':'))+'\n')
                    if step==1 or step%100==0:
                        schedule.flush()
                        assert int(optimizer.iterations.numpy())==step
                        assert all(np.isfinite(v.numpy()).all() for v in model.variables)
                        assert all(np.array_equal(a,v.numpy()) for a,v in zip(frozen,model.non_trainable_variables,strict=True))
                        print(json.dumps(dict(arm=arm,step=step,loss=value,elapsed_seconds=time.monotonic()-start)),flush=True)
                    if step%1000==0:
                        subprocess.run(['/usr/bin/python3','research/wakeword_p4/tools/check_disk_budget.py','--expected-gib','0.1'],check=True)
            assert int(optimizer.iterations.numpy())==p['steps']
            save_arrays('final');model.save_weights(root/'final.weights.h5')
            row=dict(arm=arm,steps=p['steps'],optimizer_iterations=int(optimizer.iterations.numpy()),files={f.name:sha(f) for f in root.iterdir()},elapsed_seconds=time.monotonic()-start)
            (root/'report.json').write_text(json.dumps(row,indent=2)+'\n');completed.append(row)
        report=dict(schema='multispeaker-formal-original-v1',execution_sha256=sha(LOCK),device=factory.identity,arms=completed,formal_training_completed=True,development_scored=False,acceptance_scored=False)
        with (R/'4090-multispeaker-formal-original-v1.json').open('x') as f:f.write(json.dumps(report,indent=2)+'\n')
    except Exception:
        (OUT/'failure.log').write_text(traceback.format_exc());raise

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['plan','run']);args=parser.parse_args()
    plan() if args.action=='plan' else run()
