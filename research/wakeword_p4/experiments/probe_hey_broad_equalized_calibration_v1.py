"""Full original representative float activation and state scaling control."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from plan_hey_broad_trained_quantization_v1 import sha
from run_hey_broad_trained_quantization_v1 import ROOT,published
from probe_hey_broad_encoder_layers_v1 import SITES
PLAN=ROOT/'results/4090-hey-broad-equalized-calibration-plan-v1.json'
RESULT=ROOT/'results/4090-hey-broad-equalized-calibration-result-v1.json'
OUT=Path('.local/wakeword/hey-broad-equalized-calibration-v1')
def plan():
    parent=ROOT/'results/4090-hey-broad-equalized-float-plan-v1.json'
    old=json.loads(parent.read_text());files=dict(old['files'])
    additions=[parent,Path(__file__),ROOT/'results/4090-hey-broad-equalized-weights-receipt-v1.json',ROOT/'results/4090-hey-broad-equalized-float-result-v1.json',ROOT/'results/4090-hey-broad-calibration-ranges-result-v1.json',ROOT/'results/4090-hey-broad-ptq-material-result-v1.json']
    for p in additions:files[str(p)]=sha(p)
    receipt=json.loads(additions[2].read_text())
    for a in receipt['arms']:files[a['path']]=a['sha256']
    material=json.loads(additions[-1].read_text())
    for a in material['arms']:files[a['path']]=a['sha256']
    r=dict(schema='hey-broad-equalized-calibration-plan-v1',files=files,versions=old['versions'],models=old['models'],arms=receipt['arms'],material=material['arms'],policy='Fulloriginalzero/full26+100x75sequence, same no-source-reset. Parallel original and reparameterized CPU2threads floatmodels, same15site observer. Compare every fullactivation element to original times pre-fixed factor, atol1e-4/rtol1e-5; also every element of sixactualstate buffers aftereachstep. Originalobserver SHA must equal previous7502-step trace. Original/reparameterized probability <=1e-4, all7502steps. Twofreshreset repeats for both, require fulltensor/state observation hashes exact. Save perstep maxabs/normalized errors and probabilities, no failure omission. No bounded head needed for this encoder/state control.',limits='Floatrepresentative control only, no INT8 export/admission, training/dev/acceptance or threshold change.')
    with PLAN.open('x') as f:f.write(json.dumps(r,indent=2)+'\n')
def run():
    published(PLAN);lock=json.loads(PLAN.read_text())
    for p,h in lock['files'].items():assert sha(p)==h,p
    import importlib.metadata
    for n,v in lock['versions'].items():assert importlib.metadata.version(n)==v
    admission=json.loads((ROOT/'results/4090-hey-broad-equalized-float-result-v1.json').read_text());assert admission['all_passed'] and len(admission['cases'])==270
    import probe_hey_context_transfer_shapes_v1 as helper
    tf=helper.tf
    from microwakeword import utils
    from microwakeword.layers import modes
    from microwakeword.layers.stream import Stream
    def build(weights):
        m=helper.create(225);m.load_weights(weights)
        for l in m.layers:
            c=getattr(l,'cell',l)
            if isinstance(c,tf.keras.layers.BatchNormalization):l.trainable=c.trainable=False
        s=utils.to_streaming_inference(m,{'stride':3,'spectrogram_length':225},modes.Modes.STREAM_INTERNAL_STATE_INFERENCE)
        s(np.zeros((1,3,40),np.float32),training=False)
        g=dict(activation=[l for l in s.layers if isinstance(l,tf.keras.layers.Activation)],depthwise=[l for l in s.layers if isinstance(l,tf.keras.layers.DepthwiseConv2D)],stream=[l for l in s.layers if isinstance(l,Stream)],flatten=[l for l in s.layers if isinstance(l,tf.keras.layers.Flatten)])
        assert [len(g[k]) for k in ['activation','depthwise','stream','flatten']]==[5,4,6,1]
        observer=tf.keras.Model(s.inputs,[s.outputs[0]]+[g[k][i].output for _,k,i in SITES])
        infer=tf.function(lambda x:observer(x,training=False),input_signature=[tf.TensorSpec([1,3,40],tf.float32)])
        states={tuple(v.shape):v for v in s.variables if 'states' in v.name};assert len(states)==6
        return infer,states
    prior=json.loads((ROOT/'results/4090-hey-broad-calibration-ranges-result-v1.json').read_text())
    OUT.mkdir(exist_ok=False);cases=[]
    for arm in lock['arms']:
        target=arm['target'];original,os=build(Path(lock['models'][target])/'best.weights.h5');changed,cs=build(arm['path'])
        assert os.keys()==cs.keys()=={tuple(x) for x in arm['state_shapes']}
        statekeys=[tuple(x) for x in arm['state_shapes']]
        a=[np.asarray(x,np.float32) for x in arm['activation_factors']];d=[np.asarray(x,np.float32) for x in arm['depthwise_factors']]
        factors=[a[i] if k=='activation' else d[i] if k=='depthwise' else a[i-1] if k=='stream' else np.tile(a[-1],30) for _,k,i in SITES]
        statefactors=[np.ones(40,np.float32)]+a
        material=next(r for r in lock['material'] if r['target']==target);values=np.load(material['path'],allow_pickle=False).reshape(7500,1,3,40)
        expectedsha=next(r for r in prior['cases'] if r['target']==target)['arms']['zero_full26']['full_tensor_observation_sha256']
        repeats=[];hashes=[]
        for repeat in range(2):
            for v in list(os.values())+list(cs.values()):v.assign(tf.zeros_like(v))
            oh=hashlib.sha256();ch=hashlib.sha256();sh=hashlib.sha256()
            stepmax=[];stepnorm=[];statemax=[];statenorm=[];prob=[]
            def sequence():
                yield np.zeros((1,3,40),np.float32)
                yield np.full((1,3,40),26,np.float32)
                yield from values
            for step,value in enumerate(sequence(),1):
                x=[v.numpy() for v in original(value)];y=[v.numpy() for v in changed(value)]
                for v in x:oh.update(v.tobytes())
                for v in y:ch.update(v.tobytes())
                absolute=[];normalized=[]
                for left,right,factor in zip(x[1:],y[1:],factors,strict=True):
                    assert left.shape==right.shape;expected=left*factor;diff=np.abs(right.astype(np.float64)-expected)
                    absolute.append(float(diff.max()));normalized.append(float(np.max(diff/(1e-4+1e-5*np.abs(expected)))))
                sm=[];sn=[]
                for shape,factor in zip(statekeys,statefactors,strict=True):
                    left=os[shape].numpy();right=cs[shape].numpy();sh.update(left.tobytes());sh.update(right.tobytes())
                    expected=left*factor;diff=np.abs(right.astype(np.float64)-expected)
                    sm.append(float(diff.max()));sn.append(float(np.max(diff/(1e-4+1e-5*np.abs(expected)))))
                stepmax.append(absolute);stepnorm.append(normalized);statemax.append(sm);statenorm.append(sn);prob.append([float(x[0][0,0]),float(y[0][0,0])])
                if step%1500==0:print(json.dumps(dict(target=target,repeat=repeat,steps=step)),flush=True)
            assert oh.hexdigest()==expectedsha
            hashes.append([oh.hexdigest(),ch.hexdigest(),sh.hexdigest()])
            repeats.append(dict(site_max=np.asarray(stepmax),site_normalized=np.asarray(stepnorm),state_max=np.asarray(statemax),state_normalized=np.asarray(statenorm),probability=np.asarray(prob)))
        assert hashes[0]==hashes[1] and all(np.array_equal(repeats[0][k],repeats[1][k]) for k in repeats[0])
        arrays=repeats[0];pe=float(np.max(np.abs(arrays['probability'][:,0]-arrays['probability'][:,1])))
        site=float(arrays['site_normalized'].max());state=float(arrays['state_normalized'].max())
        passed=bool(site<=1 and state<=1 and pe<=1e-4 and all(np.isfinite(v).all() for v in arrays.values()))
        path=OUT/f'{target}.npz';np.savez_compressed(path,**arrays)
        case=dict(target=target,steps=7502,all_full_tensor_repeats_exact=True,original_observation_sha_exact=True,observation_hashes=hashes[0],max_site_absolute_error=float(arrays['site_max'].max()),max_site_normalized_error=site,max_state_absolute_error=float(arrays['state_max'].max()),max_state_normalized_error=state,probability_max_error=pe,passed=passed,arrays_path=str(path),arrays_sha256=sha(path));cases.append(case);print(json.dumps(case),flush=True)
    with RESULT.open('x') as f:f.write(json.dumps(dict(schema='hey-broad-equalized-calibration-result-v1',plan_sha256=sha(PLAN),cases=cases,all_passed=all(c['passed'] for c in cases),limits=lock['limits']),indent=2)+'\n')
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['plan','run']);a=p.parse_args();plan() if a.action=='plan' else run()
