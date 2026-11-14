"""One fixed-budget RTX4090 causal sequence/onset candidate, no rescue sweep."""
import argparse
import gzip
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import time
import traceback
import numpy as np
from causal_sequence_batch_v1 import Sampler,digest
R=Path('research/wakeword_p4/results');LOCK=R/'4090-causal-sequence-formal-plan-v1.json';OUT=Path('.local/wakeword/causal-sequence-formal-v1')
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def plan():
    parent=R/'4090-causal-sequence-gpu-smoke-plan-v1.json';p=json.loads(parent.read_text())
    abi=R/'4090-causal-sequence-smoke-int8-original-v1.json';a=json.loads(abi.read_text());assert a['all_static_and_runtime_abi_passed'] and not a['recognition_scored']
    priming=R/'4090-multispeaker-priming-findings-v1.json';diagnosis=json.loads(priming.read_text());assert all(x['joint_negative_qualified_points']==0 for x in diagnosis['arms'])
    files=dict(p['files']);paths=[Path(__file__),parent,abi,priming,Path(__file__).with_name('causal_onset_training_contract_v1.md'),Path(__file__).with_name('causal_startup_coverage_hypothesis_v1.md'),R/'4090-causal-sequence-smoke-int8-plan-v1.json',Path(__file__).with_name('export_causal_sequence_smoke_v1.py')]
    files.update({str(x):sha(x) for x in paths})
    initial=p['initial'];assert 'smoke' not in initial['snapshot']['path']
    lock=dict(schema='causal-sequence-formal-plan-v1',files=files,versions=p['versions'],initial=initial,seed=726,optimizer=p['optimizer'],steps=20000,batch_size=128,padded_frames=750,positive_presentations=32,synthetic_negative_presentations=48,ambient_presentations=48,expected_new_gib=1.,progress_every=200,disk_check_every=1000,export_representative=dict(batch=p['batches'][0]['path'],lengths=json.loads((R/'4090-causal-sequence-smoke-int8-plan-v1.json').read_text())['lengths'],sentinels=[0,26],intermediate_reset=False),policy='One original preserved initialization/freshAdam, one seed726/fixed20000updates,15trainables/16frozenBN. Causalperlayer zero activation histories, variable source onset, full sequence regionnormalized positive BCE+20negativeBCE; source interval masked, synthetic startup negatives supervised. ActualRTX4090 deterministicFP32/TF32off/jit_compileFalse; no crossdevice equality or newfloatcomparison. Save initial/final model+Adam/finalH5 and compressed allbatchdescriptors/x/maskhashes. Progress/diskchecks only, no checkpoints or earlystop/modelselection. Final one builtinINT8 export with fixed first16actualbatch1 representatives/actual lengths/sentinels. Then realwholewave INT8 recognition. Consumed diagnostics never thresholdselection; no newdev/acceptance access in training. V30 and all failures retained. No LR/seed/checkpoint/calibration rescue.')
    with LOCK.open('x') as f:f.write(json.dumps(lock,indent=2)+'\n')
    print(json.dumps(dict(plan_sha256=sha(LOCK),steps=20000,initial=initial['snapshot']['path'])))
def run():
    assert subprocess.check_output(['git','show','origin/research-hye-nyx-voice-gpu:'+str(LOCK)])==LOCK.read_bytes();p=json.loads(LOCK.read_text())
    for path,h in p['files'].items():assert sha(path)==h,path
    for package,v in p['versions'].items():assert importlib.metadata.version(package)==v
    def disk():subprocess.run(['/usr/bin/python3','research/wakeword_p4/tools/check_disk_budget.py','--expected-gib',str(p['expected_new_gib'])],check=True)
    disk();OUT.mkdir(exist_ok=False);launch=dict(pid=os.getpid(),proc_starttime=Path('/proc/self/stat').read_text().split()[21],plan_sha256=sha(LOCK));(OUT/'launch.json').write_text(json.dumps(launch,indent=2)+'\n');print(json.dumps(launch),flush=True)
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
        assert len(optimizer.variables)==32 and int(optimizer.iterations.numpy())==0 and all(np.all(v.numpy()==0) for v in optimizer.variables[2:])
        def save(prefix):
            for kind,variables in [('model',model.variables),('optimizer',optimizer.variables)]:np.savez(OUT/(prefix+'-'+kind+'-variables.npz'),**{f'v{i:03}':v.numpy() for i,v in enumerate(variables)})
        save('initial')
        @tf.function(jit_compile=False,input_signature=[tf.TensorSpec([128,750,40],tf.float32),tf.TensorSpec([128,250],tf.float32),tf.TensorSpec([128,250],tf.float32)])
        def update(x,pw,nw):
            with tf.device('/GPU:0'):
                with tf.GradientTape() as tape:
                    z=logits(model,x,tf);loss=presentation_loss(z,pw,nw,tf)
                gradients=tape.gradient(loss,model.trainable_variables);assert len(gradients)==15 and all(g is not None for g in gradients)
                tf.debugging.assert_all_finite(loss,'loss')
                for g in gradients:tf.debugging.assert_all_finite(g,'gradient')
                optimizer.apply_gradients(zip(gradients,model.trainable_variables,strict=True))
                return tf.identity(loss)
        sampler=Sampler(p['seed']);start=time.monotonic();progress=[]
        original_batches=json.loads((R/'4090-causal-sequence-batch-original-v1.json').read_text())['batches']
        with gzip.open(OUT/'batches.jsonl.gz','wt') as log:
            for step in range(1,p['steps']+1):
                (x,pw,nw),descriptors=sampler.batch();hashes=dict(x_sha256=digest(x),positive_weights_sha256=digest(pw),negative_weights_sha256=digest(nw))
                if step<=2:
                    prior=original_batches[step-1];assert descriptors==prior['descriptors']
                    for key,h in hashes.items():assert h==prior[key]
                loss=update(tf.constant(x),tf.constant(pw),tf.constant(nw));value=float(loss.numpy());assert np.isfinite(value) and 'GPU:0' in loss.device
                log.write(json.dumps(dict(step=step,loss=value,**hashes,descriptors=descriptors))+'\n')
                if step==1 or step%p['progress_every']==0 or step==p['steps']:
                    assert all(np.isfinite(v.numpy()).all() for v in model.variables)
                    assert all(np.array_equal(v.numpy(),a) for v,a in zip(model.non_trainable_variables,frozen,strict=True))
                    assert int(optimizer.iterations.numpy())==step
                    row=dict(step=step,loss=value,elapsed_seconds=time.monotonic()-start,device=loss.device,optimizer_iterations=step,frozen16_unchanged=True,all15_gradients_finite=True);progress.append(row);log.flush();(OUT/'progress.json').write_text(json.dumps(dict(launch=launch,latest=row),indent=2)+'\n');print(json.dumps(row),flush=True)
                if step%p['disk_check_every']==0:disk()
        save('final');model.save_weights(OUT/'final.weights.h5');assert int(optimizer.iterations.numpy())==p['steps']
        assert all(np.isfinite(v.numpy()).all() for v in optimizer.variables)
        result=dict(schema='causal-sequence-formal-original-v1',plan_sha256=sha(LOCK),launch=launch,device=factory.identity,steps=p['steps'],optimizer_iterations=int(optimizer.iterations.numpy()),elapsed_seconds=time.monotonic()-start,progress=progress,files={f.name:sha(f) for f in OUT.iterdir()},all_training_checks_passed=True,recognition_scored=False,threshold_selected=False,acceptance_scored=False,policy=p['policy'])
        with (R/'4090-causal-sequence-formal-original-v1.json').open('x') as f:f.write(json.dumps(result,indent=2)+'\n')
        print(json.dumps(dict(completed=True,steps=p['steps'],elapsed_seconds=result['elapsed_seconds'])),flush=True)
    except Exception:
        (OUT/'failure.log').write_text(traceback.format_exc());raise
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['plan','run']);a=parser.parse_args();plan() if a.action=='plan' else run()
