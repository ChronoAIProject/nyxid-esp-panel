"""Fixed untrained Binary40 weight transfer and integer ABI feasibility."""
import hashlib,json,os,sys
from pathlib import Path
os.environ.setdefault('TF_NUM_INTRAOP_THREADS','2')
os.environ.setdefault('TF_NUM_INTEROP_THREADS','1')
import numpy as np
R=Path('research/wakeword_p4/results')
B=Path('.local/wakeword/binary40-feasibility-v1')
LOCK=R/'4090-binary40-feasibility-plan-v1.json'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def plan():
    files=[Path(__file__),Path('research/wakeword_p4/training/causal_binary40_v1.py'),R/'4090-causal-binary40-functional-v1.json',R/'4090-causal-binary40-original-batch-v1.json']
    p=dict(seed=726,parity_calls=64,calibration_calls=64,feedback_calls=256,
           files={str(f):sha(f) for f in files},
           policy='Single untrained transferred Binary40 graph. Analytic continuous uniform[0,26] feature streams, actual random-model float cache states for representative examples, not trained/audio calibration. CUDA PyTorch vs CPU Keras logits/cache parity, single builtin INT8 export, explicit cache requantization and zeropoint reset. No optimization, recognition, tuning or P4 qualification. External original40-channel frontend remains separate.')
    with LOCK.open('x') as f:json.dump(p,f,indent=2);f.write('\n')

def build(state):
    import tensorflow as tf
    L=tf.keras.layers
    chunk=L.Input(batch_shape=(1,3,1,40),name='chunk')
    histories=[2,4,8,16,32,40,1];channels=[40,64,64,64,64,64,64]
    caches=[L.Input(batch_shape=(1,h,1,c),name=f'cache{i}') for i,(h,c) in enumerate(zip(histories,channels))]
    outputs={}
    def join(i,x):
        y=L.Concatenate(axis=1)([caches[i],x]);h=histories[i]
        outputs[f'cache{i}_out']=L.Lambda(lambda a,h=h:a[:,-h:,:,:])(y)
        return y
    x=L.ReLU()(L.Conv2D(64,(3,1),strides=(3,1),name='stem')(join(0,chunk)))
    for i,d in enumerate([1,2,4,8,10]):
        y=L.ReLU()(L.DepthwiseConv2D((5,1),dilation_rate=(d,1),name=f'dw{i}')(join(i+1,x)))
        y=L.Conv2D(64,1,name=f'pw{i}')(y);x=L.ReLU()(L.Add()([x,y]))
    outputs['probability']=L.Activation('sigmoid')(L.Conv2D(1,(2,1),name='head')(join(6,x)))
    m=tf.keras.Model([chunk,*caches],outputs)
    def arr(k):return state[k].detach().cpu().numpy()
    for layer,key in [('stem','stem'),('head','head'),*[(f'pw{i}',f'pw.{i}') for i in range(5)]]:
        m.get_layer(layer).set_weights([arr(key+'.weight').transpose(2,1,0)[:,None,:,:],arr(key+'.bias')])
    for i in range(5):m.get_layer(f'dw{i}').set_weights([arr(f'dw.{i}.weight').transpose(2,0,1)[:,None,:,:],arr(f'dw.{i}.bias')])
    return m,histories,channels

def run():
    import subprocess
    assert subprocess.check_output(['git','show','origin/research-hye-nyx-voice-gpu:'+str(LOCK)])==LOCK.read_bytes()
    p=json.loads(LOCK.read_text())
    for f,h in p['files'].items():assert sha(f)==h
    subprocess.run(['/usr/bin/python3','research/wakeword_p4/tools/check_disk_budget.py','--expected-gib','.1'],check=True)
    B.mkdir(exist_ok=False)
    import torch,tensorflow as tf
    from ai_edge_litert.interpreter import Interpreter,OpResolverType
    sys.path.insert(0,str(Path('research/wakeword_p4/training').resolve()))
    from causal_binary40_v1 import Binary40
    torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.manual_seed(726)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    tf.config.set_visible_devices([],'GPU')
    assert torch.cuda.is_available() and '4090' in torch.cuda.get_device_name()
    pt=Binary40().cuda().eval();model,histories,channels=build(pt.state_dict())
    def zeros():return {f'cache{i}':np.zeros((1,h,1,c),np.float32) for i,(h,c) in enumerate(zip(histories,channels))}
    rng=np.random.default_rng(726);cache=pt.initial_state(device='cuda');ks=zeros();maximum=0.;cache_error=0.
    with torch.no_grad():
        for _ in range(p['parity_calls']):
            x=rng.uniform(0,26,(1,40,3)).astype(np.float32)
            y,cache=pt(torch.tensor(x,device='cuda'),cache)
            out=model(dict(chunk=x.transpose(0,2,1)[:,:,None,:],**ks),training=False)
            expected=torch.sigmoid(y).cpu().numpy().transpose(0,2,1)[:,:,None,:]
            maximum=max(maximum,float(np.abs(out['probability'].numpy()-expected).max()))
            assert np.allclose(out['probability'].numpy(),expected,atol=2e-4,rtol=2e-4)
            for i,v in enumerate(cache):
                ks[f'cache{i}']=out[f'cache{i}_out'].numpy();target=v.cpu().numpy().transpose(0,2,1)[:,:,None,:]
                cache_error=max(cache_error,float(np.abs(ks[f'cache{i}']-target).max()))
                assert np.allclose(ks[f'cache{i}'],target,atol=2e-4,rtol=2e-4)
    model.save(B/'random.keras')
    def representative():
        states=zeros();rng=np.random.default_rng(726)
        for _ in range(p['calibration_calls']):
            inputs=dict(chunk=rng.uniform(0,26,(1,3,1,40)).astype(np.float32),**states)
            yield inputs
            out=model(inputs,training=False)
            states={f'cache{i}':out[f'cache{i}_out'].numpy() for i in range(7)}
    c=tf.lite.TFLiteConverter.from_keras_model(model);c.optimizations=[tf.lite.Optimize.DEFAULT]
    c.representative_dataset=representative;c.target_spec.supported_ops=[tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    c.inference_input_type=tf.int8;c.inference_output_type=tf.int8
    path=B/'random.tflite';path.write_bytes(c.convert())
    def fresh():
        it=Interpreter(model_path=str(path),num_threads=1,experimental_op_resolver_type=OpResolverType.BUILTIN_REF)
        it.allocate_tensors();return it,it.get_signature_runner()
    it,runner=fresh();ins=runner.get_input_details();outs=runner.get_output_details()
    assert set(ins)=={'chunk',*[f'cache{i}' for i in range(7)]}
    assert set(outs)=={'probability',*[f'cache{i}_out' for i in range(7)]}
    assert all(v['dtype'] in [np.int8,np.int32,np.int64] for v in it.get_tensor_details())
    ops=[d['op_name'] for d in it._get_ops_details()];assert not any(x.startswith('Flex') for x in ops)
    for d in [*ins.values(),*outs.values()]:assert d['dtype']==np.int8 and d['quantization'][0]>0
    def reset():return {f'cache{i}':np.full(ins[f'cache{i}']['shape'],ins[f'cache{i}']['quantization'][1],np.int8) for i in range(7)}
    states=reset();rng=np.random.default_rng(1726);digest=hashlib.sha256();first=None
    for _ in range(p['feedback_calls']):
        s,z=ins['chunk']['quantization'];x=np.clip(np.rint(rng.uniform(0,26,ins['chunk']['shape'])/s)+z,-128,127).astype(np.int8)
        out=runner(chunk=x,**states)
        if first is None:first=(x.copy(),{k:v.copy() for k,v in out.items()})
        for k,v in sorted(out.items()):digest.update(v.tobytes())
        for i in range(7):
            oscale,ozero=outs[f'cache{i}_out']['quantization'];iscale,izero=ins[f'cache{i}']['quantization']
            states[f'cache{i}']=np.clip(np.rint((out[f'cache{i}_out'].astype(np.float64)-ozero)*oscale/iscale)+izero,-128,127).astype(np.int8)
    other,rr=fresh();again=rr(chunk=first[0],**reset());assert all(np.array_equal(v,again[k]) for k,v in first[1].items())
    def detail(v):return dict(shape=v['shape'].tolist(),dtype=str(v['dtype']),scale=v['quantization'][0],zero=v['quantization'][1])
    result=dict(schema='binary40-feasibility-original-v1',plan_sha256=sha(LOCK),model=dict(bytes=path.stat().st_size,sha256=sha(path),path=str(path)),
                torch=torch.__version__,tensorflow=tf.__version__,gpu=torch.cuda.get_device_name(),
                parity_calls=p['parity_calls'],maximum_probability_error=maximum,maximum_cache_error=cache_error,
                inputs={k:detail(v) for k,v in ins.items()},outputs={k:detail(v) for k,v in outs.items()},ops=ops,
                cache_feedback_requantization_required=[ins[f'cache{i}']['quantization']!=outs[f'cache{i}_out']['quantization'] for i in range(7)],
                cache_input_bytes=sum(int(np.prod(ins[f'cache{i}']['shape'])) for i in range(7)),
                feedback_calls=p['feedback_calls'],feedback_sha256=digest.hexdigest(),fresh_reset_exact=True,
                all_neural_tensors_integer=True,trained=False,recognition_calls=0,policy=p['policy'])
    with (R/'4090-binary40-feasibility-original-v1.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result))
if __name__=='__main__':{'plan':plan,'run':run}[sys.argv[1]]()
