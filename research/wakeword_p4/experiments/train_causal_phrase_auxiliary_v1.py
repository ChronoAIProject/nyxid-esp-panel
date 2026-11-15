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
from causal_phrase_supervision_v1 import Sampler
from causal_sequence_batch_v1 import digest
R=Path('research/wakeword_p4/results');LOCK=R/'4090-causal-phrase-auxiliary-plan-v1.json';OUT=Path('.local/wakeword/causal-phrase-auxiliary-v1')
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def plan():
    from causal_phrase_supervision_v1 import BASES,SUFFIXES
    parent=R/'4090-causal-source-balanced-plan-v1.json'
    p=json.loads(parent.read_text())
    p['schema']='causal-phrase-auxiliary-plan-v1'
    p['padded_frames']=846  # actual source-balanced sampler shape, not historical750 metadata
    paths=[parent,Path(__file__),Path(__file__).with_name('causal_phrase_supervision_v1.py'),Path(__file__).with_name('causal_character_supervision_v1.py'),Path(__file__).with_name('causal_character_graph_v1.py'),Path(__file__).with_name('compact_phrase_auxiliary_proposal_v1.md'),R/'4090-causal-phrase-supervision-v1.json']
    p['files'].update({str(f):sha(f) for f in paths})
    p['phrase_auxiliary']=dict(classes=list(BASES),command_suffixes=SUFFIXES,head_seed=1726,head_parameters=1159,coefficient=1,pooling='mean logits over true complete causal container; exclude batch padding only',unknown_ambient='excluded auxiliary CE; original binary BCE retained',deployment='original binary model only; no posthoc auxiliary head selection')
    p['policy']='Original initializer, never failed final checkpoint; original source-balanced128 inputs/seed726/lexical727/frozenBN/freshAdam1e-5/20000 RTX4090updates/final-only. Training-only19class whole-container mean-logit CE head seed1726 coefficient1 plus originalbinaryBCE including48unknownambient. No newTTS/human/phonetictruth/admissionaudit/thresholdLRcheckpointcalibrationrescue. Primary deployment binary only, no head selection. All prior failures retained; training loss is not recognition qualification.'
    p['hypothesis']='Complete-container lexical intent supervision removes per-token CTC alignment and transcript-length normalization, preserving known generator intent distinctions. It may still dilute lexical evidence or fail transfer/generalization; no causal mechanism claimed from prior CTC failures.'
    with LOCK.open('x') as f:f.write(json.dumps(p,indent=2)+'\n')
    print(json.dumps(dict(plan_sha256=sha(LOCK),steps=p['steps'],classes=len(BASES))))
def run():
    assert subprocess.check_output(['git','show','origin/research-hye-nyx-voice-gpu:'+str(LOCK)])==LOCK.read_bytes();p=json.loads(LOCK.read_text())
    for path,h in p['files'].items():assert sha(path)==h,path
    for package,v in p['versions'].items():assert importlib.metadata.version(package)==v
    def disk():subprocess.run(['/usr/bin/python3','research/wakeword_p4/tools/check_disk_budget.py','--expected-gib',str(p['expected_new_gib'])],check=True)
    disk();OUT.mkdir(exist_ok=False);launch=dict(pid=os.getpid(),proc_starttime=Path('/proc/self/stat').read_text().split()[21],plan_sha256=sha(LOCK));(OUT/'launch.json').write_text(json.dumps(launch,indent=2)+'\n');print(json.dumps(launch),flush=True)
    try:
        import hey_broad_gpu_training_factory_v1 as factory
        from causal_phrase_supervision_v1 import joint_loss
        tf=factory.tf;tf.config.experimental.enable_tensor_float_32_execution(False);tf.keras.utils.set_random_seed(p['seed'])
        model=factory.create(309);model.load_weights(p['initial']['snapshot']['path'])
        for layer in model.layers:
            cell=getattr(layer,'cell',layer)
            if isinstance(cell,tf.keras.layers.BatchNormalization):layer.trainable=cell.trainable=False
        assert len(model.variables)==31 and len(model.trainable_variables)==15 and len(model.non_trainable_variables)==16
        for v,e in zip(model.variables,p['initial']['variable_inventory']['variables'],strict=True):assert digest(v.numpy())==e['sha256']
        with tf.device('/GPU:0'):
            head=tf.keras.layers.Dense(19,kernel_initializer=tf.keras.initializers.GlorotUniform(seed=1726),bias_initializer='zeros',name='training_phrase_head');head.build((None,60))
        assert head.count_params()==1159
        trainable=list(model.trainable_variables)+list(head.trainable_variables)
        frozen=[v.numpy().copy() for v in model.non_trainable_variables]
        optimizer=tf.keras.optimizers.Adam(**p['optimizer']);optimizer.build(trainable)
        assert len(optimizer.variables)==36 and int(optimizer.iterations.numpy())==0 and all(np.all(v.numpy()==0) for v in optimizer.variables[2:])
        def save(prefix):
            for kind,variables in [('model',model.variables),('phrase-head',head.variables),('optimizer',optimizer.variables)]:np.savez(OUT/(prefix+'-'+kind+'-variables.npz'),**{f'v{i:03}':v.numpy() for i,v in enumerate(variables)})
        save('initial')
        @tf.function(jit_compile=False,input_signature=[tf.TensorSpec([128,846,40],tf.float32),tf.TensorSpec([128,282],tf.float32),tf.TensorSpec([128,282],tf.float32),tf.TensorSpec([80],tf.int32),tf.TensorSpec([80],tf.int32),tf.TensorSpec([80],tf.int32)])
        def update(x,pw,nw,indices,labels,logit_lengths):
            with tf.device('/GPU:0'):
                with tf.GradientTape() as tape:
                    loss,binary_loss,phrase_loss=joint_loss(model,head,x,pw,nw,indices,labels,logit_lengths,tf)
                gradients=tape.gradient(loss,trainable);assert len(gradients)==17 and all(g is not None for g in gradients)
                tf.debugging.assert_all_finite(loss,'loss')
                for g in gradients:tf.debugging.assert_all_finite(g,'gradient')
                optimizer.apply_gradients(zip(gradients,trainable,strict=True))
                return tf.identity(loss),binary_loss,phrase_loss
        sampler=Sampler(p['seed']);start=time.monotonic();progress=[]
        with gzip.open(OUT/'batches.jsonl.gz','wt') as log:
            for step in range(1,p['steps']+1):
                (x,pw,nw),descriptors,transcripts,targets=sampler.phrase_batch();hashes=dict(x_sha256=digest(x),positive_weights_sha256=digest(pw),negative_weights_sha256=digest(nw))
                loss,binary_loss,phrase_loss=update(tf.constant(x),tf.constant(pw),tf.constant(nw),*[tf.constant(targets[k]) for k in ['batch_indices','labels','logit_lengths']]);value=float(loss.numpy());assert np.isfinite(value) and 'GPU:0' in loss.device
                log.write(json.dumps(dict(step=step,loss=value,**hashes,descriptors=descriptors))+'\n')
                if step==1 or step%p['progress_every']==0 or step==p['steps']:
                    assert all(np.isfinite(v.numpy()).all() for v in model.variables)
                    assert all(np.array_equal(v.numpy(),a) for v,a in zip(model.non_trainable_variables,frozen,strict=True))
                    assert int(optimizer.iterations.numpy())==step
                    row=dict(step=step,loss=value,binary_loss=float(binary_loss.numpy()),phrase_loss=float(phrase_loss.numpy()),elapsed_seconds=time.monotonic()-start,device=loss.device,optimizer_iterations=step,frozen16_unchanged=True,all17_gradients_finite=True);progress.append(row);log.flush();(OUT/'progress.json').write_text(json.dumps(dict(launch=launch,latest=row),indent=2)+'\n');print(json.dumps(row),flush=True)
                if step%p['disk_check_every']==0:disk()
        save('final');model.save_weights(OUT/'final.weights.h5');assert int(optimizer.iterations.numpy())==p['steps']
        assert all(np.isfinite(v.numpy()).all() for v in optimizer.variables)
        result=dict(schema='causal-phrase-auxiliary-original-v1',plan_sha256=sha(LOCK),launch=launch,device=factory.identity,steps=p['steps'],optimizer_iterations=int(optimizer.iterations.numpy()),elapsed_seconds=time.monotonic()-start,progress=progress,files={f.name:sha(f) for f in OUT.iterdir()},all_training_checks_passed=True,recognition_scored=False,threshold_selected=False,acceptance_scored=False,policy=p['policy'])
        with (R/'4090-causal-phrase-auxiliary-original-v1.json').open('x') as f:f.write(json.dumps(result,indent=2)+'\n')
        print(json.dumps(dict(completed=True,steps=p['steps'],elapsed_seconds=result['elapsed_seconds'])),flush=True)
    except Exception:
        (OUT/'failure.log').write_text(traceback.format_exc());raise
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['plan','run']);a=parser.parse_args();plan() if a.action=='plan' else run()
