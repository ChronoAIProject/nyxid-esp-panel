"""One disposable builtin INT8 export; static ABI and fresh interpreter replay only."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import traceback
import numpy as np
R=Path('research/wakeword_p4/results');LOCK=R/'4090-causal-sequence-smoke-int8-plan-v1.json';OUT=Path('.local/wakeword/causal-sequence-smoke-int8-v1')
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def plan():
    parent=R/'4090-causal-sequence-gpu-smoke-plan-v1.json';original=R/'4090-causal-sequence-gpu-smoke-original-v1.json';p=json.loads(parent.read_text());r=json.loads(original.read_text());assert r['all_update_checks_passed'] and r['plan_sha256']==sha(parent)
    batches=R/'4090-causal-sequence-batch-original-v1.json';b=json.loads(batches.read_text())['batches'][0]
    records=Path('.local/wakeword/causal-onset-material-v1/records.jsonl');rows={x['order']:x for x in map(json.loads,records.read_text().splitlines())}
    lengths=[rows[d['index']]['frames'] if d['kind']=='synthetic' else d['frames'] for d in b['descriptors'][:16]]
    weights=Path('.local/wakeword/causal-sequence-gpu-smoke-v1/final.weights.h5');assert sha(weights)==r['files']['final.weights.h5']
    files=dict(p['files']);paths=[Path(__file__),parent,original,batches,records,weights,Path(__file__).with_name('probe_hey_context_transfer_shapes_v1.py')];paths+=list(Path('.local/wakeword/microWakeWord/microwakeword').rglob('*.py'));files.update({str(x):sha(x) for x in paths})
    lock=dict(schema='causal-sequence-smoke-int8-plan-v1',files=files,weights=str(weights),batch=b['path'],lengths=lengths,versions={n:importlib.metadata.version(n) for n in ['tensorflow','keras','numpy','ai-edge-litert']},expected_new_gib=.1,retries=0,policy='Disposable two-update weights only. Original context309 frozenBN streaming6states, reset states to realzero before export. One builtinINT8 export only, no FP32 export/comparison, calibration or weight retry. Representative zero/26sentinels then first16actualbatch1 sequences trimmed to actual lengths with final<=2featurezero pads, no calibration reset between sequences. StaticABI plus repeated fresh-interpreter replay only, no recognition or qualification claim.')
    with LOCK.open('x') as f:f.write(json.dumps(lock,indent=2)+'\n')
    print(json.dumps(dict(plan_sha256=sha(LOCK),lengths=lengths)))
def run():
    assert subprocess.check_output(['git','show','origin/research-hye-nyx-voice-gpu:'+str(LOCK)])==LOCK.read_bytes();p=json.loads(LOCK.read_text())
    for path,h in p['files'].items():assert sha(path)==h,path
    for name,v in p['versions'].items():assert importlib.metadata.version(name)==v
    subprocess.run(['/usr/bin/python3','research/wakeword_p4/tools/check_disk_budget.py','--expected-gib',str(p['expected_new_gib'])],check=True)
    OUT.mkdir(exist_ok=False);launch=dict(pid=os.getpid(),proc_starttime=Path('/proc/self/stat').read_text().split()[21],plan_sha256=sha(LOCK));(OUT/'launch.json').write_text(json.dumps(launch,indent=2)+'\n');print(json.dumps(launch),flush=True)
    try:
        import probe_hey_context_transfer_shapes_v1 as helper
        from microwakeword import utils
        from microwakeword.layers import modes
        from tensorflow.lite.python import schema_py_generated as schema
        from ai_edge_litert.interpreter import Interpreter,OpResolverType
        tf=helper.tf;model=helper.create(309);model.load_weights(p['weights'])
        for l in model.layers:
            cell=getattr(l,'cell',l)
            if isinstance(cell,tf.keras.layers.BatchNormalization):l.trainable=cell.trainable=False
        before=[v.numpy().copy() for v in model.variables]
        stream=utils.to_streaming_inference(model,{'stride':3,'spectrogram_length':309},modes.Modes.STREAM_INTERNAL_STATE_INFERENCE);stream(np.zeros((1,3,40),np.float32),training=False)
        states=[v for v in stream.variables if 'states' in v.name];assert len(states)==6
        for state in states:state.assign(tf.zeros_like(state))
        stream.export(str(OUT/'saved_model'))
        with np.load(p['batch']) as b:batch=b['x'].copy()
        sequences=[np.pad(batch[i,:length],((0,(-length)%3),(0,0))) for i,length in enumerate(p['lengths'])]
        def representative():
            yield [np.zeros((1,3,40),np.float32)];yield [np.full((1,3,40),26,np.float32)]
            for values in sequences:
                for at in range(0,len(values),3):yield [values[None,at:at+3].copy()]
        converter=tf.lite.TFLiteConverter.from_saved_model(str(OUT/'saved_model'));converter.optimizations=[tf.lite.Optimize.DEFAULT];converter._experimental_variable_quantization=True;converter.target_spec.supported_ops=[tf.lite.OpsSet.TFLITE_BUILTINS_INT8];converter.inference_input_type=tf.int8;converter.inference_output_type=tf.uint8;converter.representative_dataset=representative
        data=converter.convert();path=OUT/'int8.tflite';path.write_bytes(data);g=schema.Model.GetRootAsModel(data,0);sub=g.Subgraphs(0);names={v:k for k,v in vars(schema.BuiltinOperator).items() if isinstance(v,int)};ops=[names[g.OperatorCodes(sub.Operators(i).OpcodeIndex()).BuiltinCode()] for i in range(sub.OperatorsLength())]
        assert 'CUSTOM' not in ops and 'DEQUANTIZE' not in ops and all(sub.Tensors(i).Type()!=schema.TensorType.FLOAT32 for i in range(sub.TensorsLength()))
        assert sub.InputsLength()==sub.OutputsLength()==1
        initializers=[]
        for si in range(1,g.SubgraphsLength()):
            sg=g.Subgraphs(si)
            for ti in range(sg.TensorsLength()):
                t=sg.Tensors(ti);buf=g.Buffers(t.Buffer())
                if buf.DataLength():
                    assert t.Type()==schema.TensorType.INT8;zero=int(t.Quantization().ZeroPoint(0));assert np.all(buf.DataAsNumpy().view(np.int8)==zero);initializers.append(dict(shape=t.ShapeAsNumpy().tolist(),zero_point=zero,all_real_zero=True))
        assert len(initializers)==6
        traces=[]
        for _ in range(2):
            it=Interpreter(model_path=str(path),num_threads=1,experimental_op_resolver_type=OpResolverType.BUILTIN_REF);it.allocate_tensors();a,=it.get_input_details();b,=it.get_output_details()
            assert list(a['shape'])==[1,3,40] and a['dtype']==np.int8 and list(b['shape'])==[1,1] and b['dtype']==np.uint8 and b['quantization']==(1/256,0)
            scale,zero=a['quantization'];assert scale>0;q=np.clip(np.rint(sequences[0]/scale)+zero,-128,127).astype(np.int8);codes=[]
            for at in range(0,len(q),3):
                it.set_tensor(a['index'],q[None,at:at+3]);it.invoke();codes.append(it.get_tensor(b['index'])[0,0])
            traces.append(np.asarray(codes,np.uint8))
        assert np.array_equal(*traces) and len(traces[0])==len(sequences[0])//3
        assert all(np.array_equal(v.numpy(),old) for v,old in zip(model.variables,before,strict=True))
        replay=OUT/'abi-replay.npz';np.savez_compressed(replay,quantized_input=q,first=traces[0],second=traces[1])
        result=dict(schema='causal-sequence-smoke-int8-original-v1',plan_sha256=sha(LOCK),launch=launch,model=dict(path=str(path),bytes=len(data),sha256=sha(path)),ops=ops,initializers=initializers,input_quantization=list(a['quantization']),output_quantization=list(b['quantization']),calls_per_repeat=len(traces[0]),repeats_exact=True,replay_sha256=sha(replay),source_weights_unchanged=True,all_static_and_runtime_abi_passed=True,recognition_scored=False,policy=p['policy'])
        with (R/'4090-causal-sequence-smoke-int8-original-v1.json').open('x') as f:f.write(json.dumps(result,indent=2)+'\n')
        print(json.dumps(result),flush=True)
    except Exception:
        (OUT/'failure.log').write_text(traceback.format_exc());raise
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['plan','run']);a=parser.parse_args();plan() if a.action=='plan' else run()
