"""Final folded-weight full540 waveform gate; no old raw-BN reconstruction."""
import argparse, hashlib, importlib.metadata, json, os, subprocess, sys
from pathlib import Path
import numpy as np
R=Path('research/wakeword_p4/results')
PLAN=R/'4090-learned-grid-final-waveform-execution-v1.json'
EXPORT=R/'4090-learned-grid-final-export-result-v1.json'
RESULT=R/'4090-learned-grid-final-waveform-result-v1.json'
ORIGINAL=R/'4090-hey-broad-trained-quantization-plan-v1.json'
OUT=Path('.local/wakeword/learned-grid-final-waveform-v1')
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def published(p): assert subprocess.check_output(['git','show','origin/research-hye-nyx-voice-gpu:'+str(p)])==p.read_bytes()
def plan(check_only=False):
    if not EXPORT.exists(): print(json.dumps(dict(status='not_ready',missing=str(EXPORT)))); return 2
    exported=json.loads(EXPORT.read_text()); assert len(exported['models'])==4
    if check_only: print(json.dumps(dict(status='ready',cases=540))); return 0
    paths=[EXPORT,ORIGINAL,Path(__file__),Path(__file__).with_name('learned_grid_folded_float_reference_v1.py'),Path(__file__).with_name('audit_learned_grid_final_waveform_v1.py'),Path('research/wakeword_p4/training/features.py')]
    exports=[]
    for model in exported['models']:
        assert model['static_abi_passed']
        for a in model['artifacts'].values(): assert sha(a['path'])==a['sha256']; paths.append(Path(a['path']))
        exports.append(dict(target=model['target'],condition=model['training_arm'],**model['artifacts']))
    original=json.loads(ORIGINAL.read_text()); assert len(original['requests'])==135
    for row in original['requests']: assert sha(row['path'])==row['sha256']; paths.append(Path(row['path']))
    import pymicro_features, micro_features_cpp
    paths += [Path(pymicro_features.__file__),Path(micro_features_cpp.__file__)]
    lock=dict(files={str(p):sha(p) for p in paths},exports=exports,versions={n:importlib.metadata.version(n) for n in ('numpy','tensorflow','ai-edge-litert','soundfile','pymicro-features')},policy='All4 audited fixedfinal10000 models x135 original complete waveforms. Original gates unchanged: reset/twofresh exact, FP32vsTF1e-4, offline224+zero phase1e-4, original five-step INT8 error gates. Full startup/tail, no checkpoint/threshold selection, retries, converter or freshdata. FP32 uses serialized folded float32 parameters, no oldBN or learned quantizers. CPU reference, no parallel GPU training.')
    with PLAN.open('x') as f: f.write(json.dumps(lock,indent=2)+'\n')
    return 0

def run():
    published(PLAN); lock=json.loads(PLAN.read_text())
    for p,h in lock['files'].items(): assert sha(p)==h,p
    for n,v in lock['versions'].items(): assert importlib.metadata.version(n)==v
    os.environ['CUDA_VISIBLE_DEVICES']='-1'
    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU'); tf.config.experimental.enable_op_determinism()
    from ai_edge_litert.interpreter import Interpreter,OpResolverType
    import soundfile as sf
    from learned_grid_folded_float_reference_v1 import parameters,inspect_graph,forward
    sys.path.insert(0,str(Path('research/wakeword_p4/training').resolve()))
    from features import features
    from run_hey_broad_trained_quantization_v1 import errors
    original=json.loads(ORIGINAL.read_text()); gates=original['numerical_gates']; OUT.mkdir(exist_ok=False); cases=[]
    def interpreter(path):
        it=Interpreter(model_path=path,experimental_op_resolver_type=OpResolverType.BUILTIN_REF,num_threads=1); it.allocate_tensors(); return it
    for model in lock['exports']:
        fp=model['fp32']['path']; ip=model['int8']['path']; configs=inspect_graph(Path(fp).read_bytes())
        for row in original['requests']:
            pcm,rate=sf.read(row['path'],dtype='int16'); assert rate==16000 and pcm.ndim==1 and len(pcm)==row['samples']
            assert hashlib.sha256(pcm.astype('<i2').tobytes()).hexdigest()==row['pcm_sha256']
            raw=features(pcm); values=np.clip(np.pad(raw,((0,(-len(raw))%3),(0,0))).astype(np.float32)/25.6,0,26)
            floats=[]; foutputs=[]; integers=[]
            for repeat in range(2):
                fi=interpreter(fp); pairs=parameters(fi)
                floats.append(forward(values[None],pairs,configs,tf,True).numpy()[0,:,0].astype(np.float64))
                outputs=[]
                for j in range(0,len(values),3):
                    fi.set_tensor(fi.get_input_details()[0]['index'],values[None,j:j+3]); fi.invoke(); outputs.append(float(fi.get_tensor(fi.get_output_details()[0]['index'])[0,0]))
                foutputs.append(np.asarray(outputs,np.float32))
                it=interpreter(ip); inp=it.get_input_details()[0]; out=it.get_output_details()[0]; sc,z=inp['quantization']; oscale,oz=out['quantization']
                assert inp['dtype']==np.int8 and out['dtype']==np.uint8
                q=np.clip(np.rint(values/sc)+z,-128,127).astype(np.int8); outputs=[]
                for j in range(0,len(q),3):
                    it.set_tensor(inp['index'],q[None,j:j+3]); it.invoke(); outputs.append(int(it.get_tensor(out['index'])[0,0]))
                integers.append(np.asarray(outputs,np.uint8))
            phase=[]; nominal=[]
            for end in range(225,len(values)+1,3):
                phase.append(float(forward(np.concatenate((values[end-224:end],np.zeros((1,40),np.float32)))[None],pairs,configs,tf,False).numpy()[0,0,0]))
                nominal.append(float(forward(values[None,end-225:end],pairs,configs,tf,False).numpy()[0,0,0]))
            phase=np.asarray(phase); nominal=np.asarray(nominal); actual=floats[0]
            signed=(integers[0].astype(np.float64)-oz)*oscale-actual; stats=errors(np.abs(np.convolve(signed,np.ones(5)/5,'valid')))
            gap=float(np.max(np.abs(foutputs[0].astype(float)-actual))); pe=float(np.max(np.abs(phase-actual[74:])))
            exact=np.array_equal(*floats) and np.array_equal(*foutputs) and np.array_equal(*integers)
            passed=bool(exact and np.isfinite(actual).all() and all(np.isfinite(a).all() for a in foutputs) and gap<=1e-4 and pe<=gates['float_offline_stream_max_abs'] and stats['mean']<=gates['per_input_five_step_mean_abs_error_max'] and stats['p99']<=gates['per_input_five_step_p99_abs_error_max'] and stats['maximum']<=gates['per_input_five_step_max_abs_error_max'])
            path=OUT/f"{model['target']}-{model['condition']}-{row['order']:03}.npz"; clocks=(3*np.arange(1,len(actual)+1)+2)*160
            np.savez(path,fp32=foutputs[0],fp32_repeat=foutputs[1],raw=raw,quantized_input=q,float=actual,float_repeat=floats[1],uint8=integers[0],uint8_repeat=integers[1],offline_phase=phase,offline_nominal=nominal,support_unclamped=clocks,sample_end=np.minimum(clocks,len(pcm)))
            case=dict(target=model['target'],condition=model['condition'],request_order=row['order'],arrays_path=str(path),arrays_sha256=sha(path),repeat_exact=bool(exact),fp32_max_abs=gap,phase_max_error=pe,smoothed_abs_error=stats,raw_abs_error=errors(np.abs(signed)),input_quantization=[sc,z],output_quantization=[oscale,oz],passed=passed); cases.append(case)
            with (OUT/'cases.jsonl').open('a') as f: f.write(json.dumps(case)+'\n')
            print(json.dumps(dict(target=model['target'],condition=model['condition'],completed=len(cases),passed=passed)),flush=True)
    assert len(cases)==540
    with RESULT.open('x') as f: f.write(json.dumps(dict(execution_sha256=sha(PLAN),cases=cases,all_passed=all(c['passed'] for c in cases),limits=lock['policy']),indent=2)+'\n')

if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('action',choices=['plan','run']); parser.add_argument('--check-only',action='store_true'); a=parser.parse_args(); raise SystemExit(plan(a.check_only) if a.action=='plan' else run())
