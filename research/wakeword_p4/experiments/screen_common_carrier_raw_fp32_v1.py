"""Authenticate each new raw FP32 weight/export over all135 original witnesses."""
import argparse,hashlib,importlib.metadata,json,subprocess,traceback
from pathlib import Path
import numpy as np
R=Path('research/wakeword_p4/results');PLAN=R/'4090-hey-common-carrier-raw-fp32-waveform-execution-v1.json';RESULT=R/'4090-hey-common-carrier-raw-fp32-waveform-result-v1.json';EXPORT=R/'4090-hey-common-carrier-raw-fp32-export-result-v1.json';ORIGINAL=R/'4090-hey-broad-trained-quantization-plan-v1.json';OUT=Path('.local/wakeword/hey-common-carrier-raw-fp32-waveform-v1')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def plan(check_only=False):
    required=[EXPORT,R/'4090-hey-common-carrier-raw-fp32-export-execution-v1.json']
    for p in required:
        if not p.exists():print(json.dumps(dict(status='not_ready',missing=str(p))));return 2
    result,execution=[json.loads(p.read_text()) for p in required]
    assert result['all_static_passed'] and len(result['models'])==4 and result['execution_sha256']==sha(required[1])
    if check_only:print(json.dumps(dict(status='ready',cases=540)));return 0
    files=dict(execution['files']);paths=[*required,ORIGINAL,Path(__file__),Path(__file__).with_name('audit_common_carrier_raw_fp32_waveform_v1.py'),Path('research/wakeword_p4/training/features.py')]
    for m in result['models']:
        assert m['static_abi_passed'] and m['parameters_unchanged'] and sha(m['path'])==m['sha256'];paths.append(Path(m['path']))
        a=m['artifacts']['fp32'];assert sha(a['path'])==a['sha256'];paths.append(Path(a['path']))
    original=json.loads(ORIGINAL.read_text());assert len(original['requests'])==135
    for row in original['requests']:assert sha(row['path'])==row['sha256'];paths.append(Path(row['path']))
    import pymicro_features,micro_features_cpp
    paths.extend([Path(pymicro_features.__file__),Path(micro_features_cpp.__file__)])
    for p in paths:files[str(p)]=sha(p)
    for p,h in files.items():assert sha(p)==h,p
    lock=dict(files=files,models=result['models'],versions={n:importlib.metadata.version(n) for n in ('tensorflow','keras','numpy','soundfile','ai-edge-litert','pymicro-features')},policy='All4 new rawFP32 fixedfinal10000 x135 complete original training witnesses. OwnH5-vs-ownLiteRT1e-4 and valid224+zero phase1e-4, reset/twofresh exact, complete startup/finalpad/clocks. CPU TF, no INT8 claim, oldmodel gate inheritance, recognition/freshdata/threshold selection or retry. All failures retained.')
    with PLAN.open('x') as f:f.write(json.dumps(lock,indent=2)+'\n')
    return 0

def run():
    assert subprocess.check_output(['git','show','origin/research-hye-nyx-voice-gpu:'+str(PLAN)])==PLAN.read_bytes();lock=json.loads(PLAN.read_text())
    for p,h in lock['files'].items():assert sha(p)==h,p
    for n,v in lock['versions'].items():assert importlib.metadata.version(n)==v
    OUT.mkdir(exist_ok=False)
    try:
        import probe_hey_context_transfer_shapes_v1 as helper
        from microwakeword import utils
        from microwakeword.layers import modes
        from ai_edge_litert.interpreter import Interpreter,OpResolverType
        import soundfile as sf
        import sys
        sys.path.insert(0,str(Path('research/wakeword_p4/training').resolve()));from features import features
        tf=helper.tf;tf.config.experimental.enable_op_determinism();original=json.loads(ORIGINAL.read_text());cases=[]
        for source in lock['models']:
            model,offline=helper.create(225),helper.create(225)
            for m in (model,offline):
                m.load_weights(source['path'])
                for layer in m.layers:
                    cell=getattr(layer,'cell',layer)
                    if isinstance(cell,tf.keras.layers.BatchNormalization):layer.trainable=cell.trainable=False
            stream=utils.to_streaming_inference(model,{'stride':3,'spectrogram_length':225},modes.Modes.STREAM_INTERNAL_STATE_INFERENCE);stream(np.zeros((1,3,40),np.float32),training=False)
            states=[v for v in stream.variables if 'states' in v.name];assert len(states)==6
            infer=tf.function(lambda x:stream(x,training=False),input_signature=[tf.TensorSpec([1,3,40],tf.float32)])
            window=tf.function(lambda x:offline(x,training=False),input_signature=[tf.TensorSpec([1,225,40],tf.float32)])
            for row in original['requests']:
                pcm,rate=sf.read(row['path'],dtype='int16');assert rate==16000 and pcm.ndim==1 and len(pcm)==row['samples'] and hashlib.sha256(pcm.astype('<i2').tobytes()).hexdigest()==row['pcm_sha256']
                raw=features(pcm);assert raw.dtype==np.uint16 and raw.shape==((len(pcm)+159)//160-2,40)
                values=np.clip(np.pad(raw,((0,(-len(raw))%3),(0,0))).astype(np.float32)/25.6,0,26);floats=[];outputs=[]
                for repeat in range(2):
                    for state in states:state.assign(tf.zeros_like(state))
                    floats.append(np.asarray([float(infer(values[None,j:j+3]).numpy()[0,0]) for j in range(0,len(values),3)],np.float32))
                    it=Interpreter(model_path=source['artifacts']['fp32']['path'],experimental_op_resolver_type=OpResolverType.BUILTIN_REF,num_threads=1);it.allocate_tensors();a=it.get_input_details()[0];b=it.get_output_details()[0]
                    assert a['dtype']==b['dtype']==np.float32 and a['shape'].tolist()==[1,3,40] and b['shape'].tolist()==[1,1]
                    scores=[]
                    for j in range(0,len(values),3):it.set_tensor(a['index'],values[None,j:j+3]);it.invoke();scores.append(float(it.get_tensor(b['index'])[0,0]))
                    outputs.append(np.asarray(scores,np.float32))
                phase=np.asarray([float(window(np.concatenate((values[end-224:end],np.zeros((1,40),np.float32)))[None]).numpy()[0,0]) for end in range(225,len(values)+1,3)])
                nominal=np.asarray([float(window(values[None,end-225:end]).numpy()[0,0]) for end in range(225,len(values)+1,3)])
                exact=np.array_equal(*floats) and np.array_equal(*outputs);gap=float(np.max(np.abs(outputs[0].astype(float)-floats[0].astype(float))));pe=float(np.max(np.abs(phase-floats[0][74:])))
                valid=all(np.isfinite(v).all() and np.all((v>=0)&(v<=1)) for v in [*floats,*outputs,phase,nominal])
                passed=bool(exact and valid and gap<=1e-4 and pe<=original['numerical_gates']['float_offline_stream_max_abs'])
                clocks=(3*np.arange(1,len(floats[0])+1)+2)*160;path=OUT/f"{source['target']}-{source['condition']}-{row['order']:03}.npz"
                np.savez(path,raw=raw,float=floats[0],float_repeat=floats[1],fp32=outputs[0],fp32_repeat=outputs[1],offline_phase=phase,offline_nominal=nominal,support_unclamped=clocks,sample_end=np.minimum(clocks,len(pcm)))
                case=dict(target=source['target'],condition=source['condition'],request_order=row['order'],arrays_path=str(path),arrays_sha256=sha(path),repeat_exact=bool(exact),probabilities_valid=bool(valid),fp32_max_abs=gap,phase_max_error=pe,passed=passed);cases.append(case)
                with (OUT/'cases.jsonl').open('a') as f:f.write(json.dumps(case)+'\n')
                print(json.dumps(dict(target=source['target'],condition=source['condition'],completed=len(cases),passed=passed)),flush=True)
        assert len(cases)==540
        with RESULT.open('x') as f:f.write(json.dumps(dict(execution_sha256=sha(PLAN),cases=cases,all_passed=all(c['passed'] for c in cases),limits=lock['policy']),indent=2)+'\n')
    except Exception:
        (OUT/'failure.log').write_text(traceback.format_exc());raise

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['plan','run']);ap.add_argument('--check-only',action='store_true');a=ap.parse_args();raise SystemExit(plan(a.check_only) if a.action=='plan' else run())
