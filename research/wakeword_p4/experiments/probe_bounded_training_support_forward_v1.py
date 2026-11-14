"""Frozen final models on prospectively fixed authenticated training batches."""
import json
from pathlib import Path
import sys
import numpy as np
from schedule_hey_broad_v1 import sha
R=Path('research/wakeword_p4/results');E=Path(__file__).parent
P=R/'4090-bounded-training-support-forward-plan-v1.json'
OUT=Path('.local/wakeword/bounded-training-support-forward-v1')

def plan():
    priorp=R/'4090-bounded-training-forward-witness-plan-v1.json';prior=json.loads(priorp.read_text())
    paths=[Path(__file__),priorp]
    names=['schedule_hey_broad_v1.py','audit_hey_broad_batches_v1.py','hey_batch_schedule_v1.py','hey_broad_folded_operators_v1.py','hey_broad_scalar_fc_training_observer_v1.py','hey_broad_scalar_fc_ste_v1.py','hey_broad_fixedpoint_ste_v2.py','hey_broad_fake_quant_primitives_v1.py','hey_broad_gpu_training_factory_v1.py','probe_hey_context_transfer_shapes_v1.py']
    paths += [E/n for n in names]
    paths += [Path(prior['compile_plan'])]
    cases=[]
    for c in prior['cases']:
        paths += [Path(c['weights']),Path(c['arm']['path'])]
        target=c['target'];condition=c['condition']
        trace=Path('.local/wakeword/hey-broad-bounded-consistency-formal-v1')/f'{target}-{condition}-0'/'actual-batch-schedule.jsonl'
        paths.append(trace);cases.append(dict(target=target,condition=condition,weights=c['weights'],arm=c['arm'],trace=str(trace)))
    for t in ('hey_willow','hey_nyx'):
        paths += [Path('.local/wakeword/hey-broad-training-features-v1')/t/n for n in ('features.json','target-provenance.json')]
    d=dict(files={str(p):sha(p) for p in paths},cases=cases,compile_plan=prior['compile_plan'],versions=prior['versions'],batch_steps=list(range(1,9002,1000)),
           hypothesis='Measure whether final float versus exact-scalar-FC bounded discrepancy persists on actual supervised training support, rather than assume only heldout support causes failures. Original nominal225 versus derived shift-left224+zero are paired descriptive contexts, not a runner or training change.',
           policy='Authenticate every replayed batch1..9001 against both original formal traces; retain fixed10 whole128example batches per target with original labels/weights. All4 final models, frozenBN, original nofold/noarithmetic policy. Two repeated forward calls per context mode, save all logits/probabilities/exactFC and inputs. No gradients, optimizer, runtime activation invoke, export, new PCM, heldout, selection or promotion. Phase is a local context perturbation, not complete waveform streaming or a substitute numerical gate. Stop structural/hash/nonfinite/repeat/model-change errors, no retry.')
    with P.open('x') as f:f.write(json.dumps(d,indent=2)+'\n')

def run():
    import subprocess,importlib.metadata
    assert subprocess.check_output(['git','show','origin/research-hye-nyx-voice-gpu:'+str(P)])==P.read_bytes()
    lock=json.loads(P.read_text())
    for p,h in lock['files'].items():assert sha(p)==h,p
    for n,v in lock['versions'].items():assert importlib.metadata.version(n)==v,n
    from schedule_hey_broad_v1 import sampler
    from hey_batch_schedule_v1 import batch_digest
    OUT.mkdir(exist_ok=False);data={}
    for target in ('hey_willow','hey_nyx'):
        cases=[c for c in lock['cases'] if c['target']==target]
        traces=[[json.loads(l) for l in Path(c['trace']).read_text().splitlines()] for c in cases]
        env=sampler(target);selected=[];digests=[]
        for step in range(1,max(lock['batch_steps'])+1):
            values=env['batch'](mine=False,augment=False);digest=batch_digest(*values)
            assert all(t[step-1]['step']==step and t[step-1]['batch_sha256']==digest for t in traces)
            if step in lock['batch_steps']:selected.append(values);digests.append(digest)
            if step%1000==0:print(target,'authenticated',step,flush=True)
        x,y,w=[np.concatenate([v[i] for v in selected]) for i in range(3)]
        dest=OUT/f'{target}-inputs.npz';np.savez_compressed(dest,x=x,y=y,weights=w)
        data[target]=(x,y,w,dict(path=str(dest),sha256=sha(dest),batch_steps=lock['batch_steps'],batch_digests=digests))
    import hey_broad_gpu_training_factory_v1 as helper
    from hey_broad_folded_operators_v1 import operators as floating
    from hey_broad_scalar_fc_training_observer_v1 import operators as quantized,configuration
    tf=helper.tf;tf.config.optimizer.set_experimental_options({'constant_folding':False,'arithmetic_optimization':False})
    templates=json.loads(Path(lock['compile_plan']).read_text())['arms'];reports=[]
    for c in lock['cases']:
        target=c['target'];condition=c['condition'];x,y,w,inputs=data[target]
        arm=dict(c['arm']);arm['ops']=next(a['template']['ops'] for a in templates if (a['target'],a['condition'])==(target,condition))
        model=helper.create(225);model.load_weights(c['weights'])
        for layer in model.layers:
            cell=getattr(layer,'cell',layer)
            if isinstance(cell,tf.keras.layers.BatchNormalization):layer.trainable=cell.trainable=False
        before=[v.numpy().copy() for v in model.variables]
        finfer=floating(model,tf);qinfer=quantized(model,configuration(arm),tf)
        @tf.function
        def infer(v):
            zf=finfer(v);zq,p,o=qinfer(v)
            return zf,zq,p,o['scalar_fc_exact']
        arrays={};metrics={}
        for mode in ('nominal','phase'):
            values=x if mode=='nominal' else np.concatenate([x[:,1:,:],np.zeros((len(x),1,40),np.float32)],axis=1)
            runs=[]
            for repeat in range(2):
                chunks=[infer(tf.constant(values[i:i+128])) for i in range(0,len(values),128)]
                v=[np.concatenate([a[j].numpy() for a in chunks]) for j in range(4)]
                assert all(np.isfinite(a).all() for a in v);runs.append(v)
                for name,a in zip(('float_logit','raw_qat_logit','qat_probability','exact_fc'),v):arrays[f'{mode}_r{repeat}_{name}']=a
            assert all(np.array_equal(a,b) for a,b in zip(*runs))
            zf,zq,p,fc=runs[0];bf=np.clip(zf,-8,8);bq=np.clip((fc.astype(np.float32)-3)*np.float32(8/3),-8,8)
            difference=np.abs(bf.astype(float)-bq.astype(float))
            metrics[mode]=dict(examples=len(values),bounded_logit_abs_mean=float(difference.mean()),bounded_logit_abs_max=float(difference.max()),same_side_saturated=int(np.sum(((zf>=8)&(zq>=8))|((zf<=-8)&(zq<=-8)))),saved_repeats_exact=True)
        assert all(np.array_equal(a,v.numpy()) for a,v in zip(before,model.variables,strict=True))
        dest=OUT/f'{target}-{condition}.npz';np.savez_compressed(dest,**arrays)
        reports.append(dict(target=target,condition=condition,inputs=inputs,arrays_path=str(dest),arrays_sha256=sha(dest),model_unchanged=True,metrics=metrics));print(json.dumps(reports[-1]),flush=True)
    with (R/'4090-bounded-training-support-forward-result-v1.json').open('x') as f:f.write(json.dumps(dict(plan_sha256=sha(P),cases=reports,gpu=helper.identity,limits=lock['policy']),indent=2)+'\n')
if __name__=='__main__':
    assert sys.argv[1] in ('plan','run')
    plan() if sys.argv[1]=='plan' else run()
