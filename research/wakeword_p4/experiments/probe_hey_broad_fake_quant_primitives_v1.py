"""Registered independent affine forward/STE tests, no model inference."""
import argparse,json
from pathlib import Path
import numpy as np
from plan_hey_broad_trained_quantization_v1 import sha
from run_hey_broad_trained_quantization_v1 import ROOT,published
PLAN=ROOT/'results/4090-hey-broad-fake-quant-primitives-plan-v1.json'
RESULT=ROOT/'results/4090-hey-broad-fake-quant-primitives-result-v1.json'
def plan():
    source=ROOT/'results/4090-hey-broad-equalized-export-result-v1.json'
    paths=[source,Path(__file__),Path(__file__).with_name('hey_broad_fake_quant_primitives_v1.py')]
    r=dict(schema='hey-broad-fake-quant-primitives-plan-v1',files={str(p):sha(p) for p in paths},source=str(source),policy='CPU2threads explicit nearest-even affine quantizer and closed-interval clipped STE, range gradients stopped. All actualequalized activation/input int8 affine scales plus output uint8; zero, every lattice level, representable halfsteps and adjacent float32 values, saturation. Independent NumPy reference forwardexact and idempotenceexact; analytic weighted STE exact; rangegradientNone. Additional last-axisperchannel symmetric[-127,127], zero/tinychannelguard and signedint32bias wideforward. Do not finite-difference the discontinuous quantizer to validateSTE. No optimizer or candidateinvocation/export; no nativefakequant tie-rule equivalence assumed.',limits='Primitive mathematical contract only, not TFLite integer-kernel oracle or completeQAT/state/deployment equivalence. Dynamicweight clipping conventions and actualgraph integration stillneed validation beforeformaltraining.')
    with PLAN.open('x') as f:f.write(json.dumps(r,indent=2)+'\n')
def run():
    published(PLAN);lock=json.loads(PLAN.read_text())
    for p,h in lock['files'].items():assert sha(p)==h,p
    import tensorflow as tf
    tf.config.set_visible_devices([],'GPU');tf.config.threading.set_intra_op_parallelism_threads(2)
    from hey_broad_fake_quant_primitives_v1 import affine_ste,weight_scale
    cases=[]
    def check(name,value,scale,zero,low,high,wide=False):
        value=np.asarray(value,np.float32);scale=np.asarray(scale,np.float32);zero=np.asarray(zero,np.float32)
        dtype=np.float64 if wide else np.float32
        s=scale.astype(dtype);z=zero.astype(dtype);x=value.astype(dtype)
        code=np.clip(np.rint(x/s)+z,np.asarray(low,dtype),np.asarray(high,dtype))
        expected=((code-z)*s).astype(np.float32)
        minimum=(np.asarray(low,dtype)-z)*s;maximum=(np.asarray(high,dtype)-z)*s
        upstream=np.arange(value.size,dtype=np.float32).reshape(value.shape)%7-3
        gradient=upstream*((x>=minimum)&(x<=maximum)).astype(np.float32)
        v=tf.Variable(value);sc=tf.Variable(scale)
        with tf.GradientTape(persistent=True) as tape:
            output=affine_ste(v,sc,zero,low,high,tf,wide)
            loss=tf.reduce_sum(output*upstream)
        actual=output.numpy();g=tape.gradient(loss,v).numpy();gs=tape.gradient(loss,sc)
        assert np.array_equal(actual,expected),name+'forward'
        assert np.array_equal(g,gradient),name+'STE'
        assert gs is None,name+'range gradient'
        second=affine_ste(tf.constant(actual),scale,zero,low,high,tf,wide).numpy()
        assert np.array_equal(actual,second),name+'idempotence'
        cases.append(dict(name=name,elements=value.size,forward_exact=True,weighted_ste_exact=True,idempotent_exact=True,range_gradient_stopped=True,wide=wide))
    models=json.loads(Path(lock['source']).read_text())['arms']
    for arm in models:
        for tensor in arm['tensors']:
            if tensor['scale']<=0 or tensor['dtype'] not in ["<class 'numpy.int8'>","<class 'numpy.uint8'>"]:continue
            # Distinct actual grids only per model, no repeated tensor-grid checks.
            if tensor['index'] not in [0,50,52,53,55,56,58,59,61,62,74]:continue
            scale=np.float32(tensor['scale']);zero=np.float32(tensor['zero'])
            low,high=(0,255) if tensor['dtype']=="<class 'numpy.uint8'>" else (-128,127)
            levels=(np.arange(low,high+1,dtype=np.float32)-zero)*scale
            half=(np.arange(low,high,dtype=np.float32)-zero+np.float32(.5))*scale
            values=np.concatenate([levels,half,np.nextafter(half,np.float32(-np.inf)),np.nextafter(half,np.float32(np.inf)),np.array([0,(low-zero-2)*scale,(high-zero+2)*scale],np.float32)])
            check(arm['target']+'-tensor-'+str(tensor['index']),values,scale,zero,low,high)
    check('perchannel-weight',np.array([[-64,-1,0],[0,.25,3],[64,1,-3]],np.float32),np.array([.5,.125,.25],np.float32),0,-127,127)
    check('int32-bias-wide',np.array([-1e9,-1.25,-.75,-.25,0,.25,.75,1.25,1e9],np.float32),np.float32(.125),0,-2147483648,2147483647,True)
    weights=tf.Variable(np.array([[0,1e-20,-2],[0,-1e-20,4]],np.float32));scales=weight_scale(weights,tf)
    assert np.isfinite(scales.numpy()).all() and (scales.numpy()>0).all()
    out=affine_ste(weights,scales,0,-127,127,tf).numpy()
    assert np.array_equal(out[:,0],np.zeros(2,np.float32)) and np.isfinite(out).all()
    result=dict(schema='hey-broad-fake-quant-primitives-result-v1',plan_sha256=sha(PLAN),tensorflow=tf.__version__,cases=cases,zero_tiny_channel_guard_finite=True,all_passed=True,optimizer_updates=0,model_invocations=0,model_exports=0,limits=lock['limits'])
    with RESULT.open('x') as f:f.write(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(groups=len(cases),elements=sum(c['elements'] for c in cases),all_passed=True)))
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['plan','run']);a=p.parse_args();plan() if a.action=='plan' else run()
