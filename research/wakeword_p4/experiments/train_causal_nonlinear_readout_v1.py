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
from causal_source_balanced_batch_v1 import Sampler,digest
R=Path('research/wakeword_p4/results');LOCK=R/'4090-causal-nonlinear-readout-plan-v1.json';OUT=Path('.local/wakeword/causal-nonlinear-readout-v1')
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def plan():
    parent=R/'4090-causal-source-balanced-plan-v1.json';p=json.loads(parent.read_text())
    p['schema']='causal-nonlinear-readout-plan-v1';p['padded_frames']=846
    paths=[parent,Path(__file__),Path(__file__).with_name('causal_nonlinear_readout_v1.py'),Path(__file__).with_name('nonlinear_binary_readout_proposal_v1.md'),Path(__file__).with_name('hey_broad_gpu_training_factory_v1.py'),Path(__file__).with_name('probe_hey_context_transfer_shapes_v1.py'),Path('research/wakeword_p4/training/gpu_python.py')]
    p['files'].update({str(f):sha(f) for f in paths})
    p['architecture']=dict(context=309,stride=3,readout_frames=58,hidden_units=16,hidden_activation='relu',head_seeds=[1726,1727],parameters=78213,trainable_variables=17,frozen_variables=16,optimizer_variables=36,initial_encoder='first29 variables from original preserved initializer; old binary head discarded',auxiliary_losses=False)
    p['policy']='One fixed nonlinear deployed binary readout:3480to16ReLUto1sigmoid, original encoder initializer/frozenBN/sourcebalanced128/seed726/lexical727/freshAdam1e-5/20000RTX4090updates/final-only. Same regionnormalized binaryBCE+20negativeBCE, no auxiliary. Newheadseeds1726/1727; no failedfinal encoder. Save actual initial/final33model/36optimizer inventories. No widthgrid/LRseedthresholdcheckpointcalibrationrescue/newTTS/human/FP32comparison/numericalequalityadmission. Final one builtinINT8 export and actualcompletewave behavior required; size/arena/latency not inferred. V30 retained; no independentqualification claim.'
    p['hypothesis']='A fixed nonlinear binary readout may combine temporal encoder patterns that one affine projection cannot separate. Three auxiliary failures motivate testing the deployed function, not a proven capacity cause; overfit and unseenvoice failure remain possible.'
    with LOCK.open('x') as f:f.write(json.dumps(p,indent=2)+'\n')
    print(json.dumps(dict(plan_sha256=sha(LOCK),steps=p['steps'],parameters=78213)))
def run():
    assert subprocess.check_output(['git','show','origin/research-hye-nyx-voice-gpu:'+str(LOCK)])==LOCK.read_bytes();p=json.loads(LOCK.read_text())
    for path,h in p['files'].items():assert sha(path)==h,path
    for package,v in p['versions'].items():assert importlib.metadata.version(package)==v
    def disk():subprocess.run(['/usr/bin/python3','research/wakeword_p4/tools/check_disk_budget.py','--expected-gib',str(p['expected_new_gib'])],check=True)
    disk();OUT.mkdir(exist_ok=False);launch=dict(pid=os.getpid(),proc_starttime=Path('/proc/self/stat').read_text().split()[21],plan_sha256=sha(LOCK));(OUT/'launch.json').write_text(json.dumps(launch,indent=2)+'\n');print(json.dumps(launch),flush=True)
    try:
        import hey_broad_gpu_training_factory_v1 as factory
        from causal_nonlinear_readout_v1 import create,loss as binary_loss
        tf=factory.tf;tf.config.experimental.enable_tensor_float_32_execution(False);tf.keras.utils.set_random_seed(p['seed'])
        model=factory.create(309);model.load_weights(p['initial']['snapshot']['path'])
        for layer in model.layers:
            cell=getattr(layer,'cell',layer)
            if isinstance(cell,tf.keras.layers.BatchNormalization):layer.trainable=cell.trainable=False
        assert len(model.variables)==31 and len(model.trainable_variables)==15 and len(model.non_trainable_variables)==16
        for v,e in zip(model.variables,p['initial']['variable_inventory']['variables'],strict=True):assert digest(v.numpy())==e['sha256']
        with tf.device('/GPU:0'):model=create(model,tf)
        assert model.count_params()==78213 and len(model.trainable_variables)==17 and len(model.non_trainable_variables)==16
        inventory=dict(parameters=model.count_params(),variables=[dict(shape=list(v.shape),dtype=str(v.dtype),sha256=digest(v.numpy())) for v in model.variables],trainables=len(model.trainable_variables),frozen=len(model.non_trainable_variables))
        (OUT/'initial-model-inventory.json').write_text(json.dumps(inventory,indent=2)+'\n')
        frozen=[v.numpy().copy() for v in model.non_trainable_variables]
        optimizer=tf.keras.optimizers.Adam(**p['optimizer']);optimizer.build(model.trainable_variables)
        assert len(optimizer.variables)==36 and int(optimizer.iterations.numpy())==0 and all(np.all(v.numpy()==0) for v in optimizer.variables[2:])
        def save(prefix):
            for kind,variables in [('model',model.variables),('optimizer',optimizer.variables)]:np.savez(OUT/(prefix+'-'+kind+'-variables.npz'),**{f'v{i:03}':v.numpy() for i,v in enumerate(variables)})
        save('initial')
        @tf.function(jit_compile=False,input_signature=[tf.TensorSpec([128,846,40],tf.float32),tf.TensorSpec([128,282],tf.float32),tf.TensorSpec([128,282],tf.float32)])
        def update(x,pw,nw):
            with tf.device('/GPU:0'):
                with tf.GradientTape() as tape:
                    loss=binary_loss(model,x,pw,nw,tf)
                gradients=tape.gradient(loss,model.trainable_variables);assert len(gradients)==17 and all(g is not None for g in gradients)
                tf.debugging.assert_all_finite(loss,'loss')
                for g in gradients:tf.debugging.assert_all_finite(g,'gradient')
                optimizer.apply_gradients(zip(gradients,model.trainable_variables,strict=True))
                return tf.identity(loss)
        sampler=Sampler(p['seed']);start=time.monotonic();progress=[]
        with gzip.open(OUT/'batches.jsonl.gz','wt') as log:
            for step in range(1,p['steps']+1):
                (x,pw,nw),descriptors=sampler.batch();hashes=dict(x_sha256=digest(x),positive_weights_sha256=digest(pw),negative_weights_sha256=digest(nw))
                loss=update(tf.constant(x),tf.constant(pw),tf.constant(nw));value=float(loss.numpy());assert np.isfinite(value) and 'GPU:0' in loss.device
                log.write(json.dumps(dict(step=step,loss=value,**hashes,descriptors=descriptors))+'\n')
                if step==1 or step%p['progress_every']==0 or step==p['steps']:
                    assert all(np.isfinite(v.numpy()).all() for v in model.variables)
                    assert all(np.array_equal(v.numpy(),a) for v,a in zip(model.non_trainable_variables,frozen,strict=True))
                    assert int(optimizer.iterations.numpy())==step
                    row=dict(step=step,loss=value,elapsed_seconds=time.monotonic()-start,device=loss.device,optimizer_iterations=step,frozen16_unchanged=True,all17_gradients_finite=True);progress.append(row);log.flush();(OUT/'progress.json').write_text(json.dumps(dict(launch=launch,latest=row),indent=2)+'\n');print(json.dumps(row),flush=True)
                if step%p['disk_check_every']==0:disk()
        save('final');model.save_weights(OUT/'final.weights.h5');assert int(optimizer.iterations.numpy())==p['steps']
        assert all(np.isfinite(v.numpy()).all() for v in optimizer.variables)
        result=dict(schema='causal-nonlinear-readout-original-v1',plan_sha256=sha(LOCK),launch=launch,device=factory.identity,steps=p['steps'],optimizer_iterations=int(optimizer.iterations.numpy()),elapsed_seconds=time.monotonic()-start,progress=progress,files={f.name:sha(f) for f in OUT.iterdir()},all_training_checks_passed=True,recognition_scored=False,threshold_selected=False,acceptance_scored=False,policy=p['policy'])
        with (R/'4090-causal-nonlinear-readout-original-v1.json').open('x') as f:f.write(json.dumps(result,indent=2)+'\n')
        print(json.dumps(dict(completed=True,steps=p['steps'],elapsed_seconds=result['elapsed_seconds'])),flush=True)
    except Exception:
        (OUT/'failure.log').write_text(traceback.format_exc());raise
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['plan','run']);a=parser.parse_args();plan() if a.action=='plan' else run()
