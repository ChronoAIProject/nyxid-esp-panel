"""Registered real-batch fixed-grid fake-quant surrogate gradient preflight."""
import argparse,json
from pathlib import Path
import numpy as np
from plan_hey_broad_trained_quantization_v1 import sha
from run_hey_broad_trained_quantization_v1 import ROOT,published
PLAN=ROOT/'results/4090-hey-broad-integer-encoder-gpu-plan-v1.json'
RESULT=ROOT/'results/4090-hey-broad-integer-encoder-gpu-result-v1.json'
OUT=Path('.local/wakeword/hey-broad-integer-encoder-gpu-v1')
def plan():
    parent=ROOT/'results/4090-hey-broad-folded-operator-plan-v1.json'
    old=json.loads(parent.read_text());files=dict(old['files'])
    source=ROOT/'results/4090-hey-broad-folded-operator-result-v1.json'
    export=ROOT/'results/4090-hey-broad-equalized-export-result-v1.json'
    for p in [source,export,Path(__file__),Path(__file__).with_name('hey_broad_gpu_training_factory_v1.py'),ROOT/'results/4090-hey-broad-integer-training-parity-result-v1.json',Path(__file__).with_name('hey_broad_fixedpoint_ste_v2.py'),ROOT/'results/4090-hey-broad-recursive-integer-result-v2.json',ROOT/'results/4090-hey-broad-fixedpoint-boundaries-result-v2.json',Path(__file__).with_name('hey_broad_integer_encoder_training_v1.py'),Path(__file__).with_name('hey_broad_fake_quant_primitives_v1.py'),ROOT/'results/4090-hey-broad-fake-quant-primitives-result-v1.json']:files[str(p)]=sha(p)
    failure=ROOT/'results/4090-hey-broad-quantized-operator-initial-failure-v1.log'
    files[str(failure)]=sha(failure)
    for c in json.loads(source.read_text())['cases']:files[c['arrays_path']]=c['arrays_sha256']
    r=dict(schema='hey-broad-integer-encoder-gpu-plan-v1',files=files,versions=old['versions'],arms=old['arms'],batches=str(source),export=str(export),policy='Integer-valued float64 encoder accumulators through shared original variables, guarded source-double-rounding encoder with declaredclippedSTE. Head rawlogit retains original continuousquantized readout and BCE; posthead diagnostic remains original surrogate, not integerkernel claim. Actual RTX4090 visible, deterministic ops; check shared variables GPU placement. same4actual128x225trainingbatches/labels/weights. Fixed actualequalized TFLite input/output/weight/bias grids, frozenBNfoldedpointwise, depthwiseperchannel axis and int32biaswide. Static input/state/concat/flatten quantization grids mustmatch. Shared15trainablevariables; weighted BCE from quantizedhead precliprawlogits, diagnostic fullbounded/requantized probability separately. Repeatforward/loss/gradients exact, allfinite/nonNone and nonzerooverallgradient. Allmodelvariables exactunchanged. Nooptimizer. Saveinputs/rawlogits/probability/allgradients plusquantizerconfig.',limits='Quantization-aware surrogate feasibility only. Finalclippedhead quantizers are diagnostic and outsideBCEgradient to avoid zeroing confident-wrong gradient. Encoder integerforward mechanism integrated but actualfullbatch/integer-reference matching remains separate from gradient checks. No claim of exact sigmoid or completeinference-loss QAT. No training/export/dev/acceptance or hyperparameter selection.')
    with PLAN.open('x') as f:f.write(json.dumps(r,indent=2)+'\n')
def run():
    published(PLAN);lock=json.loads(PLAN.read_text())
    for p,h in lock['files'].items():assert sha(p)==h,p
    import importlib.metadata
    for n,v in lock['versions'].items():assert importlib.metadata.version(n)==v
    import hey_broad_gpu_training_factory_v1 as helper
    tf=helper.tf
    from hey_broad_integer_encoder_training_v1 import configuration,operators
    from hey_batch_schedule_v1 import batch_digest
    batches=json.loads(Path(lock['batches']).read_text())['cases'];exports=json.loads(Path(lock['export']).read_text())['arms']
    OUT.mkdir(exist_ok=False);cases=[];configs={}
    for arm in lock['arms']:
        target=arm['target'];model=helper.create(225);model.load_weights(arm['path'])
        for l in model.layers:
            c=getattr(l,'cell',l)
            if isinstance(c,tf.keras.layers.BatchNormalization):l.trainable=c.trainable=False
        before=[v.numpy().copy() for v in model.variables];variables=model.trainable_variables
        assert all('GPU:0' in v.value.device for v in variables),[(v.name,v.value.device) for v in variables]
        config=configuration(next(r for r in exports if r['target']==target));configs[target]=config
        infer=tf.function(operators(model,config,tf))
        for case in (r for r in batches if r['target']==target):
            with np.load(case['arrays_path'],allow_pickle=False) as saved:x=saved['x'];y=saved['y'];weight=saved['weight']
            assert batch_digest(x,y,weight)==case['actual_input_label_weight_sha256']
            repeats=[]
            for _ in range(2):
                with tf.GradientTape() as tape:
                    logits,prob=infer(tf.convert_to_tensor(x))
                    loss=tf.reduce_mean(tf.reshape(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.convert_to_tensor(y),logits=logits),[-1])*tf.convert_to_tensor(weight))
                gradients=tape.gradient(loss,variables);assert all(g is not None for g in gradients)
                repeats.append([logits.numpy(),prob.numpy(),np.array(float(loss))]+[g.numpy() for g in gradients])
            assert all(np.array_equal(a,b) for a,b in zip(*repeats,strict=True))
            assert all(np.isfinite(v).all() for v in repeats[0])
            assert all(np.array_equal(a,v.numpy()) for a,v in zip(before,model.variables,strict=True))
            norm=float(np.sqrt(sum(np.sum(g.astype(np.float64)**2) for g in repeats[0][3:])));assert norm>0
            data=dict(x=x,y=y,weight=weight,logits=repeats[0][0],probability=repeats[0][1])
            for i,g in enumerate(repeats[0][3:]):data[f'gradient_{i}']=g
            path=OUT/f'{target}-{case["batch"]:02}.npz';np.savez_compressed(path,**data)
            report=dict(target=target,batch=case['batch'],actual_input_label_weight_sha256=case['actual_input_label_weight_sha256'],weighted_bce=float(repeats[0][2]),gradient_l2=norm,twice_exact=True,finite_loss_gradients=True,variables_exact_unchanged=True,optimizer_updates=0,arrays_path=str(path),arrays_sha256=sha(path));cases.append(report);print(json.dumps(report),flush=True)
    with RESULT.open('x') as f:f.write(json.dumps(dict(schema='hey-broad-integer-encoder-gpu-result-v1',plan_sha256=sha(PLAN),cases=cases,gpu_identity=helper.identity,variable_devices=[v.value.device for v in variables],quantizer_configs=configs,all_passed=len(cases)==4,limits=lock['limits']),indent=2)+'\n')
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['plan','run']);a=p.parse_args();plan() if a.action=='plan' else run()
