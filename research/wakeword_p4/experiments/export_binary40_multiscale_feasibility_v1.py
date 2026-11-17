"""Untrained shorter-span integer feasibility with original-training stream states."""
import hashlib,json,os,subprocess,sys
from pathlib import Path
os.environ.setdefault('TF_NUM_INTRAOP_THREADS','2')
os.environ.setdefault('TF_NUM_INTEROP_THREADS','1')
import numpy as np
R=Path('research/wakeword_p4/results');B=Path('.local/wakeword/binary40-multiscale-feasibility-v1')
LOCK=R/'4090-binary40-multiscale-feasibility-plan-v1.json'
def sha_tensor(v):return hashlib.sha256(v.detach().cpu().numpy().tobytes()).hexdigest()
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def plan():
    files=dict(json.loads((R/'4090-multiscale-batch-plan-v1.json').read_text())['files'])
    for f in [Path(__file__),Path(__file__).with_name('build_binary40_multiscale_keras_v1.py'),Path(__file__).with_name('review_binary40_multiscale_feasibility_v1.py'),R/'4090-multiscale-batch-original-v1.json']:
        files[str(f)]=sha(f)
    p=dict(files=files,seed=726,training_rows=list(range(16)),calibration_call_indices=list(range(0,282,40)),parity_calls=64,feedback_calls=256,policy='Untrained fixed multiscale one-logit graph only. Original first16trainingrows/128actual training-state representatives, CUDA/Keras parity, builtinINT8, all256calls/all8outputs/cachefeedback/reset. No optimizer, recognition, calibration search or P4 claim.')
    with LOCK.open('x') as f:json.dump(p,f,indent=2);f.write('\n')

def run():
    assert subprocess.check_output(['git','show','origin/research-hye-nyx-voice-gpu:'+str(LOCK)])==LOCK.read_bytes()
    p=json.loads(LOCK.read_text())
    for path,h in p['files'].items():assert sha(path)==h
    subprocess.run(['/usr/bin/python3','research/wakeword_p4/tools/check_disk_budget.py','--expected-gib','.1'],check=True)
    B.mkdir(exist_ok=False)
    try:
        import torch,tensorflow as tf
        from ai_edge_litert.interpreter import Interpreter,OpResolverType
        from build_binary40_multiscale_keras_v1 import build
        from causal_source_balanced_batch_v1 import Sampler,digest
        sys.path.insert(0,str(Path('research/wakeword_p4/training').resolve()))
        from causal_binary40_multiscale_v1 import Binary40Multiscale
        from causal_binary40_span_v1 import CONTROL
        torch.set_num_threads(2);torch.set_num_interop_threads(1)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        tf.config.set_visible_devices([],'GPU')
        assert torch.cuda.is_available() and '4090' in torch.cuda.get_device_name()
        torch.manual_seed(726);model=Binary40Multiscale().cuda().eval();keras,histories,channels=build(model.state_dict(),CONTROL)
        original_weights={k:sha_tensor(v) for k,v in model.state_dict().items()}
        (features,pw,nw),descriptors=Sampler(726).batch();x=features[:16].transpose(0,2,1).copy()
        reference=json.loads((R/'4090-causal-binary40-original-batch-v1.json').read_text());assert descriptors==reference['descriptors'] and dict(features=digest(features),positive_weights=digest(pw),negative_weights=digest(nw))==reference['batch_hashes']
        states=model.initial_state(16,'cuda');representatives=[];chunks=[]
        with torch.no_grad():
            full,_=model(torch.tensor(x,device='cuda'),model.initial_state(16,'cuda'))
            for call in range(282):
                chunk=x[:,:,3*call:3*call+3]
                if call in p['calibration_call_indices']:
                    for row in range(16):
                        item=dict(chunk=chunk[row:row+1].transpose(0,2,1)[:,:,None,:])
                        item.update({f'cache{i}':v[row:row+1].cpu().numpy().transpose(0,2,1)[:,:,None,:].copy() for i,v in enumerate(states)})
                        representatives.append(item)
                y,states=model(torch.tensor(chunk,device='cuda'),states);chunks.append(y)
            streamed=torch.cat(chunks,dim=2);error=float((streamed-full).abs().max())
            assert torch.allclose(streamed,full,atol=2e-4,rtol=2e-4)
        assert len(representatives)==128
        np.savez_compressed(B/'calibration.npz',**{f'r{j}_{k}':v for j,item in enumerate(representatives) for k,v in item.items()})
        states=model.initial_state(device='cuda');ks={f'cache{i}':np.zeros((1,h,1,c),np.float32) for i,(h,c) in enumerate(zip(histories,channels))};prob_error=0.;cache_error=0.
        with torch.no_grad():
            for call in range(p['parity_calls']):
                chunk=x[:1,:,call*3:call*3+3];y,states=model(torch.tensor(chunk,device='cuda'),states)
                result=keras(dict(chunk=chunk.transpose(0,2,1)[:,:,None,:],**ks),training=False)
                expected=torch.sigmoid(y).cpu().numpy().transpose(0,2,1)[:,:,None,:]
                actual=result['probability'].numpy();prob_error=max(prob_error,float(np.abs(actual-expected).max()));assert np.allclose(actual,expected,atol=2e-4,rtol=2e-4)
                for i,v in enumerate(states):
                    ks[f'cache{i}']=result[f'cache{i}_out'].numpy();target=v.cpu().numpy().transpose(0,2,1)[:,:,None,:]
                    cache_error=max(cache_error,float(np.abs(ks[f'cache{i}']-target).max()));assert np.allclose(ks[f'cache{i}'],target,atol=2e-4,rtol=2e-4)
        keras.save(B/'untrained.keras')
        converter=tf.lite.TFLiteConverter.from_keras_model(keras);converter.optimizations=[tf.lite.Optimize.DEFAULT]
        converter.representative_dataset=lambda:iter(representatives);converter.target_spec.supported_ops=[tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type=tf.int8;converter.inference_output_type=tf.int8
        path=B/'int8.tflite';path.write_bytes(converter.convert())
        def fresh():
            it=Interpreter(model_path=str(path),num_threads=1,experimental_op_resolver_type=OpResolverType.BUILTIN_REF);it.allocate_tensors();return it,it.get_signature_runner()
        it,runner=fresh();ins=runner.get_input_details();outs=runner.get_output_details()
        assert set(ins)=={'chunk',*[f'cache{i}' for i in range(7)]} and set(outs)=={'probability',*[f'cache{i}_out' for i in range(7)]}
        assert all(d['dtype'] in [np.int8,np.int32,np.int64] for d in it.get_tensor_details())
        for d in [*ins.values(),*outs.values()]:assert d['dtype']==np.int8 and d['quantization'][0]>0
        assert outs['probability']['quantization']==(1/256,-128)
        ops=[d['op_name'] for d in it._get_ops_details()];assert not any(v.startswith('Flex') for v in ops)
        def zeros():return {f'cache{i}':np.full(ins[f'cache{i}']['shape'],ins[f'cache{i}']['quantization'][1],np.int8) for i in range(7)}
        states=zeros();first=None;h=hashlib.sha256();output_arrays={}
        for call in range(p['feedback_calls']):
            scale,zero=ins['chunk']['quantization'];chunk=x[:1,:,call*3:call*3+3].transpose(0,2,1)[:,:,None,:]
            q=np.clip(np.rint(chunk/scale)+zero,-128,127).astype(np.int8);out=runner(chunk=q,**states)
            if first is None:first=(q.copy(),{k:v.copy() for k,v in out.items()})
            for k,v in sorted(out.items()):h.update(v.tobytes());output_arrays[f'{call}_{k}']=v.copy()
            for i in range(7):
                s,z=outs[f'cache{i}_out']['quantization'];ts,tz=ins[f'cache{i}']['quantization']
                states[f'cache{i}']=np.clip(np.rint((out[f'cache{i}_out'].astype(np.float64)-z)*s/ts)+tz,-128,127).astype(np.int8)
        np.savez_compressed(B/'feedback.npz',**output_arrays)
        assert sum(int(np.prod(ins[f'cache{i}']['shape'])) for i in range(7))==6864
        assert original_weights=={k:sha_tensor(v) for k,v in model.state_dict().items()}
        other,rr=fresh();again=rr(chunk=first[0],**zeros());assert all(np.array_equal(v,again[k]) for k,v in first[1].items())
        def detail(d):return dict(shape=d['shape'].tolist(),scale=d['quantization'][0],zero=d['quantization'][1],dtype=str(d['dtype']))
        result=dict(schema='binary40-multiscale-feasibility-original-v1',plan_sha256=sha(LOCK),trained=False,optimizer_updates=0,geometry=model.geometry(),initialization_tensor_sha256=original_weights,
                    model=dict(path=str(path),bytes=path.stat().st_size,sha256=sha(path)),files={f.name:sha(f) for f in B.iterdir()},
                    calibration_representatives=128,calibration_descriptors=descriptors[:16],original_batch_hashes=dict(features=digest(features),positive=digest(pw),negative=digest(nw)),
                    full_chunk_max_logit_error=error,parity_calls=64,max_probability_error=prob_error,max_cache_error=cache_error,
                    inputs={k:detail(v) for k,v in ins.items()},outputs={k:detail(v) for k,v in outs.items()},ops=ops,
                    feedback_calls=256,feedback_sha256=h.hexdigest(),fresh_reset_exact=True,all_neural_tensors_integer=True,
                    cache_requantization_required=[ins[f'cache{i}']['quantization']!=outs[f'cache{i}_out']['quantization'] for i in range(7)],
                    recognition_scored=False,policy=p['policy'])
        with (R/'4090-binary40-multiscale-feasibility-original-v1.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
        print(json.dumps(result))
    except Exception:
        import traceback
        (B/'failure.log').write_text(traceback.format_exc());raise
if __name__=='__main__':{'plan':plan,'run':run}[sys.argv[1]]()
