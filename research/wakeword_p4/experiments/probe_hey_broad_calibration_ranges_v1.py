"""Registered float encoder range control on original representative sequence."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from plan_hey_broad_trained_quantization_v1 import sha
from run_hey_broad_trained_quantization_v1 import ROOT,published
from probe_hey_broad_encoder_layers_v1 import SITES
PLAN=ROOT/'results/4090-hey-broad-calibration-ranges-plan-v1.json'
RESULT=ROOT/'results/4090-hey-broad-calibration-ranges-result-v1.json'
OUT=Path('.local/wakeword/hey-broad-calibration-ranges-v1')
def plan():
    parent=ROOT/'results/4090-hey-broad-bounded-export-execution-v1.json'
    old=json.loads(parent.read_text());files=dict(old['files'])
    for p in [parent,Path(__file__),Path(__file__).with_name('probe_hey_broad_encoder_layers_v1.py')]:files[str(p)]=sha(p)
    r=dict(schema='hey-broad-calibration-ranges-plan-v1',files=files,versions=old['versions'],models=old['models'],material=old['material'],arms=['zero_full26','zero_zero'],policy='Originalfinal H5 CPU2threads frozenBN, same15matchedencoder sites and originalprobability. Full original100x75 PTQ slices in exact originalorder, no source reset. Two prefix controls: original zero+full26 versus zero+zero, same7502steps. Twice freshzero six-state reset perarm; SHA all full observation tensors at every step to require exactrepeat. Save every site perstep min/max, probability and originalDense logit; independent sigmoid reconstruction<=1e-5. Separate prefixes, first75cropsteps and remaining7425steps. Compare originalactualINT8 tensor representableranges but do not equate CPUKeras ranges to originalMLIR calibration statistics. No conversion or modifiedcalibration.',limits='Training calibration mechanism observation only. Does not prove a new PTQ would pass, or isolate converter aggregation/state arithmetic. No candidate export, optimization, newdev/acceptance, tolerance/threshold change; preserve original failures.')
    with PLAN.open('x') as f:f.write(json.dumps(r,indent=2)+'\n')
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
    OUT.mkdir(exist_ok=False);cases=[]
    for target,folder in lock['models'].items():
        m=helper.create(225);m.load_weights(Path(folder)/'best.weights.h5')
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
        head,=[l for l in s.layers if isinstance(l,tf.keras.layers.Dense)];w,b=head.get_weights()
        material=next(r for r in lock['material'] if r['target']==target)
        values=np.load(material['path'],allow_pickle=False).reshape(7500,1,3,40)
        arms={}
        for arm in lock['arms']:
            repeats=[];digests=[]
            for repeat in range(2):
                for state in states:state.assign(tf.zeros_like(state))
                h=hashlib.sha256();lo=[];hi=[];prob=[];logits=[]
                def sequence():
                    yield np.zeros((1,3,40),np.float32)
                    yield np.full((1,3,40),26 if arm=='zero_full26' else 0,np.float32)
                    yield from values
                for step,value in enumerate(sequence(),1):
                    outputs=[v.numpy() for v in infer(value)]
                    for v in outputs:h.update(v.tobytes())
                    lo.append([float(v.min()) for v in outputs[1:]]);hi.append([float(v.max()) for v in outputs[1:]])
                    prob.append(float(outputs[0][0,0]));logits.append(float((outputs[-1].astype(np.float64)@w.astype(np.float64)+b)[0,0]))
                    if step%1500==0:print(json.dumps(dict(target=target,arm=arm,repeat=repeat,steps=step)),flush=True)
                repeats.append(dict(minimum=np.asarray(lo),maximum=np.asarray(hi),probability=np.asarray(prob),logit=np.asarray(logits)));digests.append(h.hexdigest())
            assert digests[0]==digests[1]
            assert all(np.array_equal(repeats[0][k],repeats[1][k]) for k in repeats[0])
            data=repeats[0];assert len(data['probability'])==7502
            error=float(np.max(np.abs(data['probability']-np.exp(-np.logaddexp(0,-data['logit'])))));assert error<=1e-5
            path=OUT/f'{target}-{arm}.npz';np.savez_compressed(path,**data)
            ranges={}
            for name,sl in [('prefix',slice(0,2)),('first_crop',slice(2,77)),('remaining_crops',slice(77,None)),('all_crops',slice(2,None)),('whole',slice(None))]:
                ranges[name]=[dict(index=index,kind=kind,ordinal=ordinal,minimum=float(data['minimum'][sl,j].min()),maximum=float(data['maximum'][sl,j].max())) for j,(index,kind,ordinal) in enumerate(SITES)]
            arms[arm]=dict(arrays_path=str(path),arrays_sha256=sha(path),full_tensor_observation_sha256=digests[0],twice_reset_exact=True,sigmoid_reconstruction_max_error=error,ranges=ranges)
        cases.append(dict(target=target,steps_per_arm=7502,arms=arms))
    with RESULT.open('x') as f:f.write(json.dumps(dict(schema='hey-broad-calibration-ranges-result-v1',plan_sha256=sha(PLAN),cases=cases,limits=lock['limits']),indent=2)+'\n')
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['plan','run']);a=p.parse_args();plan() if a.action=='plan' else run()
