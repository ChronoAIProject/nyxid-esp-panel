"""Original wholewave numerical gates on four matched final exports."""
import argparse,importlib.metadata,json,subprocess
from pathlib import Path
import numpy as np
from plan_hey_broad_trained_quantization_v1 import sha
from run_hey_broad_trained_quantization_v1 import ROOT,published,errors
from collect_complete_development_v1 import runtime_identity_files
PLAN=ROOT/'results/4090-multispeaker-formal-screen-plan-v1.json'
LOCK=ROOT/'results/4090-multispeaker-formal-screen-execution-v1.json'
RESULT=ROOT/'results/4090-multispeaker-formal-screen-result-v1.json'
EXPORT=Path('.local/wakeword/multispeaker-formal-export-v1/result.json')
OUT=Path('.local/wakeword/multispeaker-formal-screen-v1')
ORIGINAL_PLAN=ROOT/'results/4090-hey-broad-trained-quantization-plan-v1.json'
def lock():
 if not EXPORT.exists():print(json.dumps(dict(status='not_ready',missing=str(EXPORT),model_invocations=0)));return 2
 audit=ROOT/'results/4090-multispeaker-formal-export-independent-audit-v1.json';a=json.loads(audit.read_text());assert a['all_checks_passed'] and a['static_report_sha256']==sha(EXPORT);e=json.loads(EXPORT.read_text());parent=ROOT/'results/4090-multispeaker-formal-export-plan-v1.json';published(parent);p=json.loads(parent.read_text());assert e['all_static_passed'] and e['execution_sha256']==sha(parent) and len(e['models'])==2;files=dict(p['files'])
 paths=[EXPORT,parent,audit,Path(__file__),ORIGINAL_PLAN,Path(__file__).with_name('audit_multispeaker_formal_screen_v1.py')]
 paths += runtime_identity_files()
 paths += [Path(__file__).with_name(n) for n in ['run_hey_broad_trained_quantization_v1.py','plan_hey_broad_trained_quantization_v1.py','collect_complete_development_v1.py']]
 paths += [ROOT/'training/features.py',ROOT/'tools/check_disk_budget.py']
 original=json.loads(ORIGINAL_PLAN.read_text());exports=[]
 for m in e['models']:
  assert m['static_abi_passed'] and m['parameters_unchanged']
  exports.append(dict(target=m['target'],condition=m['condition'],path=m['artifacts']['int8']['path'],fp32_path=m['artifacts']['fp32']['path']))
  for artifact in m['artifacts'].values():assert sha(artifact['path'])==artifact['sha256'];paths.append(Path(artifact['path']))
 for row in original['requests']:assert sha(row['path'])==row['sha256'];paths.append(Path(row['path']))
 for f in paths:files[str(f)]=sha(f)
 versions=dict(p['versions']);versions.update({n:importlib.metadata.version(n) for n in ['soundfile','pymicro-features','scipy']})
 with LOCK.open('x') as f:f.write(json.dumps(dict(files=files,versions=versions,weights=p['weights'],exports=exports,policy='BothoriginalNyxcontext309fixedfinal20000 rawsigmoid x originalshared135requests. Full startup, finalpad, two fresh/reset repeats, phase308+zero1e-4, FP32LiteRT-vsTFmax1e-4, originalINT8fiveMAE.02/p99.1/max.25. Consumed diagnostic135 only, not freshdev or acceptance. No new data, checkpoint/head/threshold selection or retries; all failures retained.'),indent=2)+'\n')
def run():
    published(LOCK);locked=json.loads(LOCK.read_text())
    for p,h in locked['files'].items():assert sha(p)==h,p
    for n,v in locked['versions'].items():assert importlib.metadata.version(n)==v
    subprocess.run(['/usr/bin/python3',str(ROOT/'tools/check_disk_budget.py'),'--expected-gib','0.25'],check=True)
    OUT.mkdir(exist_ok=False)
    import probe_hey_context_transfer_shapes_v1 as helper
    tf=helper.tf
    from microwakeword import utils
    from microwakeword.layers import modes
    import soundfile as sf
    from ai_edge_litert.interpreter import Interpreter,OpResolverType
    import sys
    sys.path.insert(0,str((ROOT/'training').resolve()))
    from features import features
    original=json.loads(ORIGINAL_PLAN.read_text());gates=original['numerical_gates'];cases=[]
    assert {(w['target'],w['condition']) for w in locked['weights']}=={(t,c) for t in ['hey_nyx'] for c in ['narrow','broad']}
    for source in locked['weights']:
        target=source['target'];condition=source['condition']
        models=[]
        for _ in range(2):
            
            model=helper.create(309)
            model.load_weights(source['path'])
            for layer in model.layers:
                cell=getattr(layer,'cell',layer)
                if isinstance(cell,tf.keras.layers.BatchNormalization):layer.trainable=cell.trainable=False
            models.append(model)
        model,offline=models
        stream=utils.to_streaming_inference(model,{'stride':3,'spectrogram_length':309},modes.Modes.STREAM_INTERNAL_STATE_INFERENCE)
        stream(np.zeros((1,3,40),np.float32),training=False)
        states=[v for v in stream.variables if 'states' in v.name];assert len(states)==6
        infer=tf.function(lambda x:stream(x,training=False),input_signature=[tf.TensorSpec([1,3,40],tf.float32)])
        window=tf.function(lambda x:offline(x,training=False),input_signature=[tf.TensorSpec([1,309,40],tf.float32)])
        for index,row in enumerate(original['requests']):
            pcm,rate=sf.read(row['path'],dtype='int16');assert rate==16000 and pcm.ndim==1 and len(pcm)==row['samples']
            assert __import__('hashlib').sha256(pcm.astype('<i2').tobytes()).hexdigest()==row['pcm_sha256']
            raw=features(pcm);values=np.clip(np.pad(raw,((0,(-len(raw))%3),(0,0))).astype(np.float32)/25.6,0,26)
            floats=[];integers=[];fp32=[]
            for _ in range(2):
                for state in states:state.assign(tf.zeros_like(state))
                floats.append(np.asarray([float(infer(values[None,j:j+3]).numpy()[0,0]) for j in range(0,len(values),3)],np.float32))
                interpreter=Interpreter(model_path=next(r['path'] for r in locked['exports'] if r['target']==target and r['condition']==condition),experimental_op_resolver_type=OpResolverType.BUILTIN_REF,num_threads=1);interpreter.allocate_tensors()
                inp=interpreter.get_input_details()[0];out=interpreter.get_output_details()[0]
                assert list(inp['shape'])==[1,3,40] and inp['dtype']==np.int8
                assert list(out['shape'])==[1,1] and out['dtype']==np.uint8
                scale,zero=inp['quantization'];oscale,ozero=out['quantization'];assert scale>0 and oscale>0
                q=np.clip(np.rint(values/scale)+zero,-128,127).astype(np.int8);scores=[]
                for j in range(0,len(q),3):
                    interpreter.set_tensor(inp['index'],q[None,j:j+3]);interpreter.invoke();scores.append(int(interpreter.get_tensor(out['index'])[0,0]))
                integers.append(np.asarray(scores,np.uint8))
                fi=Interpreter(model_path=next(r['fp32_path'] for r in locked['exports'] if r['target']==target and r['condition']==condition),experimental_op_resolver_type=OpResolverType.BUILTIN_REF,num_threads=1);fi.allocate_tensors();fa=fi.get_input_details()[0];fb=fi.get_output_details()[0];assert fa['dtype']==fb['dtype']==np.float32;fs=[]
                for j in range(0,len(values),3):
                    fi.set_tensor(fa['index'],values[None,j:j+3]);fi.invoke();fs.append(float(fi.get_tensor(fb['index'])[0,0]))
                fp32.append(np.asarray(fs,np.float32))
            ends=range(309,len(values)+1,3)
            phase=np.asarray([float(window(np.concatenate((values[end-308:end],np.zeros((1,40),np.float32)))[None]).numpy()[0,0]) for end in ends])
            nominal=np.asarray([float(window(values[None,end-309:end]).numpy()[0,0]) for end in ends])
            actual=floats[0].astype(np.float64);signed=(integers[0].astype(np.float64)-ozero)*oscale-actual
            smooth=np.convolve(signed,np.ones(5)/5,mode='valid');e=errors(np.abs(smooth));pe=float(np.max(np.abs(phase-actual[102:])))
            exact=np.array_equal(*floats) and np.array_equal(*integers) and np.array_equal(*fp32)
            fp32gap=float(np.max(np.abs(actual-fp32[0].astype(float))))
            passed=bool(exact and np.isfinite(actual).all() and all(np.isfinite(v).all() for v in fp32) and fp32gap<=1e-4 and pe<=gates['float_offline_stream_max_abs'] and e['mean']<=gates['per_input_five_step_mean_abs_error_max'] and e['p99']<=gates['per_input_five_step_p99_abs_error_max'] and e['maximum']<=gates['per_input_five_step_max_abs_error_max'])
            path=OUT/f'{target}-{condition}-{index:03}.npz';clocks=(3*np.arange(1,len(actual)+1)+2)*160
            np.savez(path,fp32=fp32[0],fp32_repeat=fp32[1],raw=raw,quantized_input=q,float=actual,uint8=integers[0],float_repeat=floats[1],uint8_repeat=integers[1],offline_phase=phase,offline_nominal=nominal,support_unclamped=clocks,sample_end=np.minimum(clocks,len(pcm)))
            case=dict(fp32_max_abs=fp32gap,target=target,condition=condition,request_order=row['order'],repeat_exact=bool(exact),phase_max_error=pe,smoothed_abs_error=e,raw_abs_error=errors(np.abs(signed)),passed=passed,arrays_path=str(path),arrays_sha256=sha(path),input_quantization=[scale,zero],output_quantization=[oscale,ozero])
            cases.append(case)
            with (OUT/'cases.jsonl').open('a') as f:f.write(json.dumps(case)+'\n')
            print(json.dumps(dict(target=target,condition=condition,completed=index+1,passed=passed)),flush=True)
    assert len(cases)==270
    with RESULT.open('x') as f:f.write(json.dumps(dict(schema='multispeaker-formal-screen-result-v1',execution_sha256=sha(LOCK),cases=cases,all_passed=all(c['passed'] for c in cases),limits='Numerical gate only; no recognition/FAR/development/acceptance or promotion.'),indent=2)+'\n')

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['lock','run']);a=p.parse_args();raise SystemExit({'lock':lock,'run':run}[a.action]())
