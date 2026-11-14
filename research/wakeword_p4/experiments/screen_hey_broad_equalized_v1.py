"""Unchanged full-wave numerical gates for registered bounded-head exports."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import numpy as np
from plan_hey_broad_trained_quantization_v1 import sha
from run_hey_broad_trained_quantization_v1 import ROOT,published,errors

PLAN=ROOT/'results/4090-hey-broad-equalized-screen-plan-v1.json'
LOCK=ROOT/'results/4090-hey-broad-equalized-screen-execution-v1.json'
RESULT=ROOT/'results/4090-hey-broad-equalized-screen-result-v1.json'
EXPORT=ROOT/'results/4090-hey-broad-equalized-export-result-v1.json'
OUT=Path('.local/wakeword/hey-broad-equalized-screen-v1')
ORIGINAL_PLAN=ROOT/'results/4090-hey-broad-trained-quantization-plan-v1.json'

def plan():
    parent=ROOT/'results/4090-hey-broad-trained-quantization-execution-v1.json'
    old=json.loads(parent.read_text());files=dict(old['files'])
    for p in [parent,Path(__file__),Path(__file__).with_name('probe_hey_broad_bounded_float_v1.py'),ROOT/'results/4090-hey-broad-equalized-export-plan-v1.json',Path(__file__).with_name('export_hey_broad_equalized_v1.py')]:files[str(p)]=sha(p)
    for p in [ROOT/'results/4090-hey-broad-equalized-weights-receipt-v1.json',ROOT/'results/4090-hey-broad-equalized-float-result-v1.json',ROOT/'results/4090-hey-broad-equalized-calibration-result-v1.json',ROOT/'results/4090-hey-broad-equalized-export-execution-v1.json']:files[str(p)]=sha(p)
    record=dict(schema='hey-broad-equalized-screen-plan-v1',files=files,versions=old['versions'],models=old['models'],numerical_gates=json.loads(ORIGINAL_PLAN.read_text())['numerical_gates'],policy='Exact original135 complete PCM witnesses per arm, original native frontend and final-only feature padding; fresh equalized bounded float offline/stream from immutable compensated H5 and bounded INT8 builtin reference1thread, twice fresh-reset exact. Actual bounded offline phase224+zero compared against stream after74steps, original1e-4 gate. All270 original five-mean MAE<=.02,p99<=.1,max<=.25 unchanged. No case dropped; save raw inputs, quantized inputs, all traces/repeats/offline and clocks. Lock actual exported bytes and metadata before any inference. No fresh development, recognition, acceptance or threshold access.')
    with PLAN.open('x') as f:f.write(json.dumps(record,indent=2)+'\n')

def lock():
    published(PLAN);p=json.loads(PLAN.read_text());published(EXPORT)
    files=dict(p['files']);files[str(PLAN)]=sha(PLAN);files[str(EXPORT)]=sha(EXPORT)
    export=json.loads(EXPORT.read_text());assert len(export['arms'])==2
    for r in export['arms']:
        assert r['static_gate_passed'];assert sha(r['path'])==r['sha256'];files[r['path']]=r['sha256']
    receipt=json.loads((ROOT/'results/4090-hey-broad-equalized-weights-receipt-v1.json').read_text())
    for arm in receipt['arms']:
        assert sha(arm['path'])==arm['sha256'];files[arm['path']]=arm['sha256']
    for path,h in files.items():assert sha(path)==h,path
    record=dict(schema='hey-broad-equalized-screen-execution-v1',files=files,models=p['models'],versions=p['versions'],exports={r['target']:r['path'] for r in export['arms']},equalized_weights=receipt['arms'])
    with LOCK.open('x') as f:f.write(json.dumps(record,indent=2)+'\n')

def run():
    published(LOCK);locked=json.loads(LOCK.read_text())
    for p,h in locked['files'].items():assert sha(p)==h,p
    for n,v in locked['versions'].items():assert importlib.metadata.version(n)==v
    OUT.mkdir(exist_ok=False)
    import probe_hey_context_transfer_shapes_v1 as helper
    from probe_hey_broad_bounded_float_v1 import bounded
    tf=helper.tf
    from microwakeword import utils
    from microwakeword.layers import modes
    import soundfile as sf
    from ai_edge_litert.interpreter import Interpreter,OpResolverType
    import sys
    sys.path.insert(0,str((ROOT/'training').resolve()))
    from features import features
    original=json.loads(ORIGINAL_PLAN.read_text());gates=original['numerical_gates'];cases=[]
    for target,folder in locked['models'].items():
        models=[]
        for _ in range(2):
            model=helper.create(225)
            source=next(r for r in locked['equalized_weights'] if r['target']==target)
            model.load_weights(source['path'])
            for layer in model.layers:
                cell=getattr(layer,'cell',layer)
                if isinstance(cell,tf.keras.layers.BatchNormalization):layer.trainable=cell.trainable=False
            models.append(bounded(model,tf))
        model,offline=models
        stream=utils.to_streaming_inference(model,{'stride':3,'spectrogram_length':225},modes.Modes.STREAM_INTERNAL_STATE_INFERENCE)
        stream(np.zeros((1,3,40),np.float32),training=False)
        states=[v for v in stream.variables if 'states' in v.name];assert len(states)==6
        infer=tf.function(lambda x:stream(x,training=False),input_signature=[tf.TensorSpec([1,3,40],tf.float32)])
        window=tf.function(lambda x:offline(x,training=False),input_signature=[tf.TensorSpec([1,225,40],tf.float32)])
        for index,row in enumerate(original['requests']):
            pcm,rate=sf.read(row['path'],dtype='int16');assert rate==16000 and pcm.ndim==1 and len(pcm)==row['samples']
            assert __import__('hashlib').sha256(pcm.astype('<i2').tobytes()).hexdigest()==row['pcm_sha256']
            raw=features(pcm);values=np.clip(np.pad(raw,((0,(-len(raw))%3),(0,0))).astype(np.float32)/25.6,0,26)
            floats=[];integers=[]
            for _ in range(2):
                for state in states:state.assign(tf.zeros_like(state))
                floats.append(np.asarray([float(infer(values[None,j:j+3]).numpy()[0,0]) for j in range(0,len(values),3)],np.float32))
                interpreter=Interpreter(model_path=locked['exports'][target],experimental_op_resolver_type=OpResolverType.BUILTIN_REF,num_threads=1);interpreter.allocate_tensors()
                inp=interpreter.get_input_details()[0];out=interpreter.get_output_details()[0]
                assert list(inp['shape'])==[1,3,40] and inp['dtype']==np.int8
                assert list(out['shape'])==[1,1] and out['dtype']==np.uint8
                scale,zero=inp['quantization'];oscale,ozero=out['quantization'];assert scale>0 and oscale>0
                q=np.clip(np.rint(values/scale)+zero,-128,127).astype(np.int8);scores=[]
                for j in range(0,len(q),3):
                    interpreter.set_tensor(inp['index'],q[None,j:j+3]);interpreter.invoke();scores.append(int(interpreter.get_tensor(out['index'])[0,0]))
                integers.append(np.asarray(scores,np.uint8))
            ends=range(225,len(values)+1,3)
            phase=np.asarray([float(window(np.concatenate((values[end-224:end],np.zeros((1,40),np.float32)))[None]).numpy()[0,0]) for end in ends])
            nominal=np.asarray([float(window(values[None,end-225:end]).numpy()[0,0]) for end in ends])
            actual=floats[0].astype(np.float64);signed=(integers[0].astype(np.float64)-ozero)*oscale-actual
            smooth=np.convolve(signed,np.ones(5)/5,mode='valid');e=errors(np.abs(smooth));pe=float(np.max(np.abs(phase-actual[74:])))
            exact=np.array_equal(*floats) and np.array_equal(*integers)
            passed=bool(exact and np.isfinite(actual).all() and pe<=gates['float_offline_stream_max_abs'] and e['mean']<=gates['per_input_five_step_mean_abs_error_max'] and e['p99']<=gates['per_input_five_step_p99_abs_error_max'] and e['maximum']<=gates['per_input_five_step_max_abs_error_max'])
            path=OUT/f'{target}-{index:03}.npz';clocks=(3*np.arange(1,len(actual)+1)+2)*160
            np.savez(path,raw=raw,quantized_input=q,float=actual,uint8=integers[0],float_repeat=floats[1],uint8_repeat=integers[1],offline_phase=phase,offline_nominal=nominal,support_unclamped=clocks,sample_end=np.minimum(clocks,len(pcm)))
            case=dict(target=target,request_order=row['order'],repeat_exact=bool(exact),phase_max_error=pe,smoothed_abs_error=e,raw_abs_error=errors(np.abs(signed)),passed=passed,arrays_path=str(path),arrays_sha256=sha(path),input_quantization=[scale,zero],output_quantization=[oscale,ozero])
            cases.append(case)
            with (OUT/'cases.jsonl').open('a') as f:f.write(json.dumps(case)+'\n')
            print(json.dumps(dict(target=target,completed=index+1,passed=passed)),flush=True)
    assert len(cases)==270
    with RESULT.open('x') as f:f.write(json.dumps(dict(schema='hey-broad-equalized-screen-result-v1',execution_sha256=sha(LOCK),cases=cases,all_passed=all(c['passed'] for c in cases),limits='Numerical gate only; no recognition/FAR/development/acceptance or promotion.'),indent=2)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['plan','lock','run']);a=p.parse_args();{'plan':plan,'lock':lock,'run':run}[a.action]()
