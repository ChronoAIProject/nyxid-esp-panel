"""Register and test actual training-batch float graph and gradient parity."""
import argparse,json
from pathlib import Path
import numpy as np
from plan_hey_broad_trained_quantization_v1 import sha
from run_hey_broad_trained_quantization_v1 import ROOT,published
PLAN=ROOT/'results/4090-hey-broad-folded-operator-plan-v1.json'
RESULT=ROOT/'results/4090-hey-broad-folded-operator-result-v1.json'
OUT=Path('.local/wakeword/hey-broad-folded-operator-v1')
def plan():
    parent=ROOT/'results/4090-hey-broad-equalized-export-execution-v1.json'
    old=json.loads(parent.read_text());files=dict(old['files'])
    for p in [parent,Path(__file__),Path(__file__).with_name('hey_broad_folded_operators_v1.py'),Path(__file__).with_name('schedule_hey_broad_v1.py'),Path(__file__).with_name('hey_batch_schedule_v1.py'),Path(__file__).with_name('hey_broad_qat_preflight_v1.md')]:files[str(p)]=sha(p)
    record=dict(schema='hey-broad-folded-operator-plan-v1',files=files,versions=old['versions'],arms=old['equalized_weights'],policy='CPU2threads nooptimizer. Originalsampler first2full batches perarm, verify complete x/y/weight SHA against original formal schedule. Load compensated immutableH5, freezeBN. Compare originalsharedencoder+Dense prelogit versus explicitConv/Depthwise/foldedBNpointwise/ReLU/flatten/Dense operator graph, same variables. Weighted BCE fromrawlogits. Probability maxabs<=1e-4; relativeBCEdiff<=1e-4*max(1,abs(referenceBCE)); wholetrainable-gradient relativeL2<=1e-3. Allloss/gradients finite, noNonegradient; variables/statistics exactunchanged. Save actual x/y/weights/logits/allgradients andhashes. No fakequant orformaltraining yet.',limits='Floatgraph andgradient preflight only. Does not validate straight-through fakequant gradients, integer kernel/state equivalence, candidate numerical/recognition quality, newdevelopment oracceptance.')
    with PLAN.open('x') as f:f.write(json.dumps(record,indent=2)+'\n')
def run():
    published(PLAN);lock=json.loads(PLAN.read_text())
    for p,h in lock['files'].items():assert sha(p)==h,p
    import importlib.metadata
    for n,v in lock['versions'].items():assert importlib.metadata.version(n)==v
    import probe_hey_context_transfer_shapes_v1 as helper
    tf=helper.tf
    from hey_broad_folded_operators_v1 import operators
    from schedule_hey_broad_v1 import sampler
    from hey_batch_schedule_v1 import batch_digest
    formal=json.loads((ROOT/'results/4090-hey-broad-formal-training-lock-v1.json').read_text())
    OUT.mkdir(exist_ok=False);cases=[]
    for arm in lock['arms']:
        target=arm['target'];assert sha(arm['path'])==arm['sha256']
        model=helper.create(225);model.load_weights(arm['path'])
        for l in model.layers:
            c=getattr(l,'cell',l)
            if isinstance(c,tf.keras.layers.BatchNormalization):l.trainable=c.trainable=False
        before=[v.numpy().copy() for v in model.variables];variables=model.trainable_variables
        head,=[l for l in model.layers if isinstance(l,tf.keras.layers.Dense)]
        encoder=tf.keras.Model(model.inputs,head.input)
        def reference(x):return tf.linalg.matmul(encoder(x,training=False),head.kernel)+head.bias
        functions=[tf.function(reference),tf.function(operators(model,tf))]
        env=sampler(target)
        with Path(formal['arms'][target]['schedule']).open() as f:expected=[json.loads(next(f)) for _ in range(2)]
        for order in range(2):
            x,y,weight=env['batch'](mine=False,augment=False);digest=batch_digest(x,y,weight)
            assert expected[order]['index']==order+1 and expected[order]['tensor_sha256']==digest
            logits=[];losses=[];gradients=[]
            for function in functions:
                with tf.GradientTape() as tape:
                    logit=function(tf.convert_to_tensor(x))
                    loss=tf.reduce_mean(tf.reshape(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.convert_to_tensor(y),logits=logit),[-1])*tf.convert_to_tensor(weight))
                gradient=tape.gradient(loss,variables);assert all(g is not None for g in gradient)
                gradient=[g.numpy() for g in gradient]
                assert all(np.isfinite(g).all() for g in gradient) and np.isfinite(float(loss))
                logits.append(logit.numpy());losses.append(float(loss));gradients.append(gradient)
            probability=[np.exp(-np.logaddexp(0,-z.astype(np.float64))) for z in logits]
            pe=float(np.max(np.abs(probability[0]-probability[1])))
            delta=np.concatenate([(b.astype(np.float64)-a).reshape(-1) for a,b in zip(*gradients,strict=True)])
            reference_gradient=np.concatenate([g.astype(np.float64).reshape(-1) for g in gradients[0]])
            rel=float(np.linalg.norm(delta)/max(np.linalg.norm(reference_gradient),1e-30))
            le=abs(losses[1]-losses[0]);passed=bool(pe<=1e-4 and le<=1e-4*max(1,abs(losses[0])) and rel<=1e-3)
            assert all(np.array_equal(a,v.numpy()) for a,v in zip(before,model.variables,strict=True))
            path=OUT/f'{target}-{order+1:02}.npz'
            arrays=dict(x=x,y=y,weight=weight,reference_logit=logits[0],folded_logit=logits[1])
            for i,(a,b) in enumerate(zip(*gradients,strict=True)):arrays[f'reference_gradient_{i}']=a;arrays[f'folded_gradient_{i}']=b
            np.savez_compressed(path,**arrays)
            case=dict(target=target,batch=order+1,actual_input_label_weight_sha256=digest,reference_bce=losses[0],folded_bce=losses[1],probability_max_abs_error=pe,logit_max_abs_error=float(np.max(np.abs(logits[0]-logits[1]))),gradient_relative_l2=rel,gradient_max_abs_error=float(np.max(np.abs(delta))),variables_exact_unchanged=True,optimizer_updates=0,passed=passed,arrays_path=str(path),arrays_sha256=sha(path),variable_shapes=[list(v.shape) for v in variables]);cases.append(case)
            print(json.dumps(case),flush=True)
    with RESULT.open('x') as f:f.write(json.dumps(dict(schema='hey-broad-folded-operator-result-v1',plan_sha256=sha(PLAN),cases=cases,all_passed=all(r['passed'] for r in cases),limits=lock['limits']),indent=2)+'\n')
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['plan','run']);a=p.parse_args();plan() if a.action=='plan' else run()
