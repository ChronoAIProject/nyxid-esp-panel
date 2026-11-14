"""Registered same-step encoder float/INT8 boundary observations, no export."""
import argparse
import json
from pathlib import Path
import numpy as np
from plan_hey_broad_trained_quantization_v1 import sha
from run_hey_broad_trained_quantization_v1 import ROOT,published
PLAN=ROOT/'results/4090-hey-broad-encoder-layers-plan-v1.json'
RESULT=ROOT/'results/4090-hey-broad-encoder-layers-result-v1.json'
OUT=Path('.local/wakeword/hey-broad-encoder-layers-v1')
# Converter-fused BN/ReLU must be compared at post-activation, not raw Conv.
SITES=[(50,'activation',0),(52,'depthwise',0),(53,'activation',1),(55,'depthwise',1),(56,'activation',2),(58,'depthwise',2),(59,'activation',3),(61,'depthwise',3),(62,'activation',4),(51,'stream',1),(54,'stream',2),(57,'stream',3),(60,'stream',4),(63,'stream',5),(64,'flatten',0)]
def plan():
    parent=ROOT/'results/4090-hey-broad-bounded-head-decomposition-plan-v1.json'
    old=json.loads(parent.read_text());files=dict(old['files'])
    for p in [parent,Path(__file__)]:files[str(p)]=sha(p)
    r=dict(schema='hey-broad-encoder-layers-plan-v1',files=files,versions=old['versions'],cases=old['cases'],sites=SITES,policy='Same2fixed worst-training-witnesses. Originalfinal H5 CPU2threads frozenBN, original six-state stream. Single shared observational graph returns originalprobability plus15matching encoder sites (post-fused BN/ReLU, depthwise, concatenatedstate/context, flatten). Natural savedraw features and actualsavedQ dequantized input, eachtwicefreshreset, allsteps exactrepeat. Naturalprobability vsoriginalsavednatural <=1e-4; shape and op producer alignments mandatory. Compare saved dequantINT8 to floatnatural and floatroundedinput per site. Save everyarray and per-site absolute/RMS/quantization-LSB error; differences include propagated upstream error, not isolated local operator error. Same steps no shifting/trimming.',limits='Two training failure witnesses only, no whole270 gate or per-layer causal proof. Observational Keras graph must match original probability and preserve state repeats. No newexport/PTQ/training/newdev/acceptance/threshold or tolerance changes.')
    with PLAN.open('x') as f:f.write(json.dumps(r,indent=2)+'\n')
def metrics(delta,scale):
    flat=delta.reshape(len(delta),-1)
    return dict(max_abs=float(np.max(np.abs(delta))),mean_abs=float(np.mean(np.abs(delta))),rms=float(np.sqrt(np.mean(delta**2))),max_abs_in_lsb=float(np.max(np.abs(delta))/scale),per_step_max_abs=np.max(np.abs(flat),axis=1).tolist())
def run():
    published(PLAN);lock=json.loads(PLAN.read_text())
    for p,h in lock['files'].items():assert sha(p)==h,p
    import importlib.metadata
    for n,v in lock['versions'].items():assert importlib.metadata.version(n)==v
    import probe_hey_context_transfer_shapes_v1 as helper
    tf=helper.tf
    from microwakeword import utils
    from microwakeword.layers import modes
    from microwakeword.layers.stream import Stream
    execution=json.loads((ROOT/'results/4090-hey-broad-bounded-screen-execution-v1.json').read_text())
    OUT.mkdir(exist_ok=False);cases=[]
    for row in lock['cases']:
        trace=row['tensor_trace'];target=trace['target'];m=helper.create(225);m.load_weights(Path(execution['models'][target])/'best.weights.h5')
        for layer in m.layers:
            c=getattr(layer,'cell',layer)
            if isinstance(c,tf.keras.layers.BatchNormalization):layer.trainable=c.trainable=False
        s=utils.to_streaming_inference(m,{'stride':3,'spectrogram_length':225},modes.Modes.STREAM_INTERNAL_STATE_INFERENCE)
        s(np.zeros((1,3,40),np.float32),training=False)
        groups=dict(activation=[l for l in s.layers if isinstance(l,tf.keras.layers.Activation)],depthwise=[l for l in s.layers if isinstance(l,tf.keras.layers.DepthwiseConv2D)],stream=[l for l in s.layers if isinstance(l,Stream)],flatten=[l for l in s.layers if isinstance(l,tf.keras.layers.Flatten)])
        assert [len(groups[k]) for k in ['activation','depthwise','stream','flatten']]==[5,4,6,1]
        observer=tf.keras.Model(s.inputs,[s.outputs[0]]+[groups[k][i].output for _,k,i in SITES])
        infer=tf.function(lambda x:observer(x,training=False),input_signature=[tf.TensorSpec([1,3,40],tf.float32)])
        states=[v for v in s.variables if 'states' in v.name];assert len(states)==6
        with np.load(row['scored']['arrays_path'],allow_pickle=False) as saved:raw=saved['raw'];q=saved['quantized_input']
        scale,zero=row['scored']['input_quantization']
        natural=np.clip(np.pad(raw,((0,(-len(raw))%3),(0,0))).astype(np.float32)/25.6,0,26)
        rounded=((q.astype(np.float64)-zero)*scale).astype(np.float32)
        outputs={}
        for arm,values in [('natural',natural),('rounded_input',rounded)]:
            repeats=[]
            for _ in range(2):
                for state in states:state.assign(tf.zeros_like(state))
                streams=[[] for _ in range(1+len(SITES))]
                for at in range(0,len(values),3):
                    for bucket,value in zip(streams,infer(values[None,at:at+3]),strict=True):bucket.append(value.numpy().copy())
                repeats.append([np.stack(v) for v in streams])
            assert all(np.array_equal(a,b) for a,b in zip(*repeats,strict=True))
            outputs[arm]=repeats[0]
        with np.load(row['original_logit_trace']['arrays_path'],allow_pickle=False) as saved:baseline=saved['probability']
        error=float(np.max(np.abs(outputs['natural'][0].reshape(-1)-baseline)));assert error<=1e-4
        metadata={t['index']:t for t in trace['tensors']};arrays={};sites=[]
        with np.load(trace['arrays_path'],allow_pickle=False) as saved:
            for j,(index,kind,ordinal) in enumerate(SITES,1):
                meta=metadata[index]
                expected={'activation':'CONV_2D','depthwise':'DEPTHWISE_CONV_2D','stream':'CONCATENATION','flatten':'RESHAPE'}[kind];assert meta['producer']==expected
                integer=(saved[f'tensor_{index}'].astype(np.float64)-meta['zero'])*meta['scale']
                a,b=outputs['natural'][j],outputs['rounded_input'][j];assert a.shape==b.shape==integer.shape
                arrays.update({f'{index}_natural':a,f'{index}_rounded':b,f'{index}_int_dequant':integer})
                sites.append(dict(index=index,kind=kind,ordinal=ordinal,float_layer=groups[kind][ordinal].name,int_tensor=meta['name'],shape=list(a.shape),quantization_scale=meta['scale'],int_vs_natural=metrics(integer-a,meta['scale']),int_vs_rounded=metrics(integer-b,meta['scale']),input_rounding_only=metrics(b.astype(np.float64)-a,meta['scale'])))
        arrays.update(probability_natural=outputs['natural'][0],probability_rounded=outputs['rounded_input'][0])
        path=OUT/f'{target}.npz';np.savez_compressed(path,**arrays)
        case=dict(target=target,request_order=trace['request_order'],original_probability_max_error=error,twice_reset_exact=True,sites=sites,arrays_path=str(path),arrays_sha256=sha(path));cases.append(case)
        print(json.dumps(dict(target=target,sites=len(sites),original_probability_max_error=error)),flush=True)
    with RESULT.open('x') as f:f.write(json.dumps(dict(schema='hey-broad-encoder-layers-result-v1',plan_sha256=sha(PLAN),cases=cases,limits=lock['limits']),indent=2)+'\n')
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['plan','run']);a=p.parse_args();plan() if a.action=='plan' else run()
