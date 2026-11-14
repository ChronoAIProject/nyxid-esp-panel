"""Authenticate original representative per-channel extrema for exact equalization."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from plan_hey_broad_trained_quantization_v1 import sha
from run_hey_broad_trained_quantization_v1 import ROOT,published
from probe_hey_broad_encoder_layers_v1 import SITES
PLAN=ROOT/'results/4090-hey-broad-channel-ranges-plan-v1.json'
RESULT=ROOT/'results/4090-hey-broad-channel-ranges-result-v1.json'
OUT=Path('.local/wakeword/hey-broad-channel-ranges-v1')
def plan():
    parent=ROOT/'results/4090-hey-broad-calibration-ranges-plan-v1.json'
    previous=ROOT/'results/4090-hey-broad-calibration-ranges-result-v1.json'
    old=json.loads(parent.read_text());files=dict(old['files'])
    for p in [parent,previous,Path(__file__)]:files[str(p)]=sha(p)
    r=dict(schema='hey-broad-channel-ranges-plan-v1',files=files,versions=old['versions'],models=old['models'],material=old['material'],previous=str(previous),policy='Originalzero/full26+100x75 PTQ sequence unchanged, same original CPU2thread15site sharedobserver. Twice fresh six-state resets. SHA everyfull observation tensor must EXACTLY equal prior original zero_full26 observation; save perstep perchannel min/max and check priorglobal extrema. Derive diagnostic positive power-of-two scale s=2**(-ceil(log2(max_abs_channel))) for nonzerochannels, zerochannels1; no clamp or sweep, fail nonfinite/scales not representable. This only records prospective reparameterization factors, no model weights changed. Do not derive scales from selectedfailure/dev.',limits='Training-only calibration channel metadata. Does not guarantee exact model/state transformation or improved quantization. Actual reparameterization needs separately registered unchangedfloat/control tests before export. No candidate training/export/dev/acceptance.')
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
    previous=json.loads(Path(lock['previous']).read_text())
    OUT.mkdir(exist_ok=False);cases=[]
    for target,folder in lock['models'].items():
        m=helper.create(225);m.load_weights(Path(folder)/'best.weights.h5')
        for l in m.layers:
            c=getattr(l,'cell',l)
            if isinstance(c,tf.keras.layers.BatchNormalization):l.trainable=c.trainable=False
        s=utils.to_streaming_inference(m,{'stride':3,'spectrogram_length':225},modes.Modes.STREAM_INTERNAL_STATE_INFERENCE)
        s(np.zeros((1,3,40),np.float32),training=False)
        groups=dict(activation=[l for l in s.layers if isinstance(l,tf.keras.layers.Activation)],depthwise=[l for l in s.layers if isinstance(l,tf.keras.layers.DepthwiseConv2D)],stream=[l for l in s.layers if isinstance(l,Stream)],flatten=[l for l in s.layers if isinstance(l,tf.keras.layers.Flatten)])
        assert [len(groups[k]) for k in ['activation','depthwise','stream','flatten']]==[5,4,6,1]
        observer=tf.keras.Model(s.inputs,[s.outputs[0]]+[groups[k][i].output for _,k,i in SITES]);infer=tf.function(lambda x:observer(x,training=False),input_signature=[tf.TensorSpec([1,3,40],tf.float32)])
        states=[v for v in s.variables if 'states' in v.name];assert len(states)==6
        material=next(r for r in lock['material'] if r['target']==target);values=np.load(material['path'],allow_pickle=False).reshape(7500,1,3,40)
        prior=next(r for r in previous['cases'] if r['target']==target)['arms']['zero_full26']
        repeats=[]
        for repeat in range(2):
            for state in states:state.assign(tf.zeros_like(state))
            digest=hashlib.sha256();lower=[[] for _ in SITES];upper=[[] for _ in SITES]
            def sequence():
                yield np.zeros((1,3,40),np.float32)
                yield np.full((1,3,40),26,np.float32)
                yield from values
            for step,value in enumerate(sequence(),1):
                outputs=[v.numpy() for v in infer(value)]
                for v in outputs:digest.update(v.tobytes())
                for j,v in enumerate(outputs[1:]):
                    flat=v.reshape(-1,v.shape[-1]);lower[j].append(flat.min(axis=0));upper[j].append(flat.max(axis=0))
                if step%1500==0:print(json.dumps(dict(target=target,repeat=repeat,steps=step)),flush=True)
            assert digest.hexdigest()==prior['full_tensor_observation_sha256']
            repeats.append({f'{index}_{name}':np.stack(values) for (index,_,_),lo,hi in zip(SITES,lower,upper,strict=True) for name,values in [('min',lo),('max',hi)]})
        assert all(np.array_equal(repeats[0][k],repeats[1][k]) for k in repeats[0]);data=repeats[0];sites=[]
        for j,(index,kind,ordinal) in enumerate(SITES):
            lo=data[f'{index}_min'].min(axis=0);hi=data[f'{index}_max'].max(axis=0)
            old=prior['ranges']['whole'][j];assert float(lo.min())==old['minimum'] and float(hi.max())==old['maximum']
            peak=np.maximum(np.abs(lo),np.abs(hi)).astype(np.float64);scales=np.ones_like(peak);positive=peak>0
            scales[positive]=np.exp2(-np.ceil(np.log2(peak[positive])))
            assert np.isfinite(scales).all() and (scales>0).all() and np.isfinite(scales.astype(np.float32)).all() and (scales.astype(np.float32)>0).all()
            sites.append(dict(index=index,kind=kind,ordinal=ordinal,channel_min=lo.tolist(),channel_max=hi.tolist(),channel_peak=peak.tolist(),power_of_two_scale=scales.tolist(),nonzero_channels=int(positive.sum()),max_to_median_nonzero_peak=float(peak.max()/np.median(peak[positive])) if positive.any() else None))
        path=OUT/f'{target}.npz';np.savez_compressed(path,**data)
        cases.append(dict(target=target,steps=7502,prior_observation_sha_exact=True,twice_reset_exact=True,sites=sites,arrays_path=str(path),arrays_sha256=sha(path)))
    with RESULT.open('x') as f:f.write(json.dumps(dict(schema='hey-broad-channel-ranges-result-v1',plan_sha256=sha(PLAN),cases=cases,limits=lock['limits']),indent=2)+'\n')
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['plan','run']);a=p.parse_args();plan() if a.action=='plan' else run()
