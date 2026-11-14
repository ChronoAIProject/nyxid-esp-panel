"""Derived no-update one-way wiring verification; original S/C/F/Q retained."""
import json
from pathlib import Path
import sys
import numpy as np
from schedule_hey_broad_v1 import sha
R=Path('research/wakeword_p4/results');E=Path(__file__).parent
P=R/'4090-bounded-support-oneway-wiring-plan-v1.json'
OUT=Path('.local/wakeword/bounded-support-oneway-wiring-v1')

def inventory(variables):
    import hashlib
    return [dict(index=i,shape=list(v.shape),dtype=str(v.numpy().dtype),sha256=hashlib.sha256(v.numpy().tobytes()).hexdigest()) for i,v in enumerate(variables)]

def plan():
    parentp=R/'4090-bounded-training-support-forward-plan-v1.json'
    resultp=R/'4090-bounded-training-support-forward-result-v1.json'
    auditp=R/'4090-bounded-training-support-forward-independent-audit-v1.json'
    parent=json.loads(parentp.read_text());result=json.loads(resultp.read_text())
    assert json.loads(auditp.read_text())['result_sha256']==sha(resultp)
    files=dict(parent['files'])
    for p in [parentp,resultp,auditp,Path(__file__),E/'bounded_support_gradient_diagnostic_v1.md',E/'hey_broad_bounded_consistency_loss_v1.py',E/'hey_broad_oneway_consistency_loss_v1.py',E/'hey_broad_oneway_consistency_proposal_v1.md',E/'probe_bounded_support_gradients_v1.py',R/'4090-bounded-support-gradient-result-v1.json',R/'4090-bounded-support-gradient-independent-audit-v1.json']:
        files[str(p)]=sha(p)
    cases=[]
    for c in result['cases']:
        pc,=[v for v in parent['cases'] if (v['target'],v['condition'])==(c['target'],c['condition'])]
        for p in [c['inputs']['path'],c['arrays_path']]:files[p]=sha(p)
        cases.append(dict(**pc,inputs=c['inputs'],prior_arrays=c['arrays_path']))
    lock=dict(files=files,cases=cases,versions=parent['versions'],compile_plan=parent['compile_plan'],batch_steps=parent['batch_steps'],
              policy='No-update one-way wiring validation, same original4models/10batches/2modes/tworepeats. Require exact prior forward/S gradients and exact same-run W=Q gradient. Save original C/F/Q/W gradients; retained old gradient comparisons descriptive except S exact. All original failures retained, no retry/tolerance adaptation/optimizer/heldout/selection. Shared weights imply this is not fixed teacher. '+(E/'hey_broad_oneway_consistency_proposal_v1.md').read_text())
    with P.open('x') as f:f.write(json.dumps(lock,indent=2)+'\n')

def run():
    import subprocess,importlib.metadata
    assert subprocess.check_output(['git','show','origin/research-hye-nyx-voice-gpu:'+str(P)])==P.read_bytes()
    lock=json.loads(P.read_text())
    for p,h in lock['files'].items():assert sha(p)==h,p
    for n,v in lock['versions'].items():assert importlib.metadata.version(n)==v,n
    import hey_broad_gpu_training_factory_v1 as helper
    from hey_broad_folded_operators_v1 import operators as floating
    from hey_broad_scalar_fc_training_observer_v1 import operators as quantized,configuration
    from hey_broad_bounded_consistency_loss_v1 import components
    from hey_broad_oneway_consistency_loss_v1 import components as oneway
    tf=helper.tf;tf.config.optimizer.set_experimental_options({'constant_folding':False,'arithmetic_optimization':False})
    OUT.mkdir(exist_ok=False);reports=[]
    prior_gradient=json.loads((R/'4090-bounded-support-gradient-result-v1.json').read_text())
    templates=json.loads(Path(lock['compile_plan']).read_text())['arms']
    for case in lock['cases']:
        target=case['target'];condition=case['condition'];dest=OUT/f'{target}-{condition}';dest.mkdir()
        with np.load(case['inputs']['path']) as z:x=z['x'];y=z['y'];w=z['weights']
        arm=dict(case['arm']);arm['ops']=next(v['template']['ops'] for v in templates if (v['target'],v['condition'])==(target,condition))
        model=helper.create(225);model.load_weights(case['weights'])
        for layer in model.layers:
            cell=getattr(layer,'cell',layer)
            if isinstance(cell,tf.keras.layers.BatchNormalization):layer.trainable=cell.trainable=False
        variables=model.trainable_variables;assert len(variables)==15
        before=inventory(model.variables);trainable=inventory(variables)
        fwd=floating(model,tf);config=configuration(arm);qwd=quantized(model,config,tf)
        @tf.function
        def observe(v,labels,weight):
            with tf.GradientTape(persistent=True) as tape:
                zf=fwd(v);zq,prob,obs=qwd(v);fc=obs['scalar_fc_exact'];acc=obs['scalar_fc_accumulator']
                bf=tf.clip_by_value(zf,-8.,8.);bq=tf.clip_by_value((tf.cast(fc,tf.float32)-3)*tf.constant(8/3,tf.float32),-8.,8.)
                total,s,c=components(zf,zq,bf,bq,labels,weight,1,tf)
                ot,os,oc=oneway(zf,zq,bf,bq,labels,weight,1,tf)
                tf.debugging.assert_equal(s,os);tf.debugging.assert_equal(c,oc);tf.debugging.assert_equal(total,ot)
            seedf=tape.gradient(c,zf);seedq=tape.gradient(c,fc);seeda=tape.gradient(c,acc)
            assert seedf is not None and seedq is not None and seeda is not None
            grads=dict(S=tape.gradient(s,variables),C=tape.gradient(c,variables),F=tape.gradient(zf,variables,output_gradients=seedf),Q=tape.gradient(fc,variables,output_gradients=seedq),W=tape.gradient(oc,variables))
            outputs=dict(oneway_consistency=oc,oneway_float_seed_is_none=tf.constant(tape.gradient(oc,zf) is None),float_logit=zf,raw_qat_logit=zq,qat_probability=prob,exact_fc=fc,accumulator=acc,seed_float=seedf,seed_fc=seedq,seed_accumulator=seeda,supervised=s,consistency=c,total=total)
            for name,values in grads.items():
                outputs[name+'_none']=tf.constant([v is None for v in values])
                for i,(g,var) in enumerate(zip(values,variables,strict=True)):
                    outputs[f'{name}_{i:02}']=tf.zeros_like(var) if g is None else g
            return outputs
        records=[]
        previous,=[v for v in prior_gradient['cases'] if (v['target'],v['condition'])==(target,condition)]
        with np.load(case['prior_arrays']) as prior:
            for mode in ('nominal','phase'):
                values=x if mode=='nominal' else np.concatenate([x[:,1:,:],np.zeros((len(x),1,40),np.float32)],axis=1)
                for b,step in enumerate(lock['batch_steps']):
                    sl=slice(b*128,(b+1)*128);saved=[]
                    for repeat in range(2):
                        result={k:v.numpy() for k,v in observe(tf.constant(values[sl]),tf.constant(y[sl]),tf.constant(w[sl])).items()}
                        assert all(np.isfinite(v).all() for v in result.values())
                        for name in ('float_logit','raw_qat_logit','qat_probability','exact_fc'):
                            assert np.array_equal(result[name],prior[f'{mode}_r0_{name}'][sl]),(target,condition,mode,step,name,'prior-forward-mismatch')
                        assert bool(result['oneway_float_seed_is_none'])
                        assert np.array_equal(result['W_none'],result['Q_none'])
                        assert all(np.array_equal(result[f'W_{i:02}'],result[f'Q_{i:02}']) for i in range(15))
                        priorrec,=[v for v in previous['records'] if (v['mode'],v['step'],v['repeat'])==(mode,step,repeat)]
                        with np.load(priorrec['path']) as pg:
                            assert np.array_equal(result['S_none'],pg['S_none'])
                            assert all(np.array_equal(result[f'S_{i:02}'],pg[f'S_{i:02}']) for i in range(15)),(target,condition,mode,step,'prior-S-gradient-mismatch')
                        path=dest/f'{mode}-step{step}-r{repeat}.npz';np.savez_compressed(path,**result);saved.append(result)
                        records.append(dict(mode=mode,step=step,repeat=repeat,path=str(path),sha256=sha(path)))
                    assert saved[0].keys()==saved[1].keys() and all(np.array_equal(saved[0][k],saved[1][k]) for k in saved[0])
                    assert inventory(model.variables)==before
                    print(target,condition,mode,step,'exact-repeat/forward/unchanged',flush=True)
        reports.append(dict(target=target,condition=condition,inventory=before,trainable_inventory=trainable,configuration=config,records=records,model_unchanged=True))
    with (R/'4090-bounded-support-oneway-wiring-result-v1.json').open('x') as f:f.write(json.dumps(dict(plan_sha256=sha(P),cases=reports,gpu=helper.identity,optimizer_updates=0,limits=lock['policy']),indent=2)+'\n')
if __name__=='__main__':
    assert sys.argv[1] in ('plan','run')
    if sys.argv[1]=='plan':plan()
    else:
        try:run()
        except Exception as e:
            import traceback
            path=R/'4090-bounded-support-oneway-wiring-failure-v1.json'
            with path.open('x') as f:f.write(json.dumps(dict(error=repr(e),traceback=traceback.format_exc()),indent=2)+'\n')
            raise
