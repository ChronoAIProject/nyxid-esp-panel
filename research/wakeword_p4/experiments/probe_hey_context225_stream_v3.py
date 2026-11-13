"""Registered context225 offline/stream numerical equivalence on training PCM."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
import numpy as np

BASE=Path('.local/wakeword');RESULTS=Path('research/wakeword_p4/results')
PLAN=RESULTS/'4090-hey-context225-stream-probe-plan-v3.json';OUT=BASE/'hey-context225-stream-probe-v3'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def plan():
    import pymicro_features,micro_features_cpp
    init=RESULTS/'4090-hey-broad-context225-initialization-v1.json';d=json.loads(init.read_text());requests=[];files={str(init):sha(init),d['snapshot']['path']:d['snapshot']['sha256']}
    for engine in ['piper','kokoro','qwen']:
        root=BASE/f'hey-broad-{engine}-native-v1';rp=RESULTS/f'4090-hey-broad-{engine}-generation-v1.json';report=json.loads(rp.read_text());records=root/'records.jsonl';assert sha(records)==report['records_sha256']
        files.update({str(rp):sha(rp),str(records):sha(records)})
        for r in [json.loads(s) for s in records.read_text().splitlines()][:2]:
            assert r['request']['target'] in ['hey_willow','hey_nyx'] and r['quality_pass']
            p=Path(r['files']['wav']['path'])
            if not p.exists():p=root/p
            requests.append(dict(engine=engine,order=r['order'],target=r['request']['target'],path=str(p),sha256=r['files']['wav']['sha256']))
    files.update({str(p):sha(p) for p in [Path(__file__),Path(__file__).with_name('probe_hey_context_transfer_shapes_v1.py'),Path('research/wakeword_p4/training/features.py'),Path(pymicro_features.__file__),Path(micro_features_cpp.__file__)]})
    files.update({str(p):sha(p) for p in sorted((BASE/'microWakeWord/microwakeword').rglob('*.py'))})
    result=dict(schema='hey-context225-stream-probe-plan-v3',files=files,requests=requests,initial_snapshot=d['snapshot'],
        versions={n:importlib.metadata.version(n) for n in ['tensorflow','numpy','pymicro-features','soundfile']},
        lead_samples=48000,tail_samples=16000,context=225,hop_frames=3,absolute_tolerance=1e-4,
        policy='Six full training-only original PCM at native16k, quiet padding, complete MicroFrontend once per clip. Load actual frozen shared225 H5. Two fresh stream resets exact; compare all steps after225frames with independent original offline225 sliding windows at same3frame clock. No optimization, model selection or wake threshold scoring.',
        prior_failure_sha256=sha(RESULTS/'4090-hey-context225-stream-probe-failure-v2.json'),phase_policy='First conv valid kernel5/stride3: nominal225 has one unused trailing frame. Compare stream with last224 actual frames plus one unused zero frame; also save nominal225 comparison unchanged. Phase correction is analytically fixed, no offset search.',offline_model_policy='Separate instance loaded from same H5 before source model mode conversion.',limits='Initialization float numerical admission only, no trained/INT8 export or recognition claim.')
    with PLAN.open('x') as f:f.write(json.dumps(result,indent=2)+'\n')

def run():
    import probe_hey_context_transfer_shapes_v1 as helper
    import soundfile as sf
    tf=helper.tf
    remote=subprocess.check_output(['git','show','origin/research-hye-nyx-voice-gpu:'+str(PLAN)])
    assert remote==PLAN.read_bytes();d=json.loads(remote)
    for p,h in d['files'].items():assert sha(Path(p))==h
    for n,v in d['versions'].items():assert importlib.metadata.version(n)==v
    from microwakeword import utils
    from microwakeword.layers import modes
    sys.path.insert(0,str(Path('research/wakeword_p4/training').resolve()));from features import features
    assert not OUT.exists();OUT.mkdir()
    model=helper.create(225);model.load_weights(d['initial_snapshot']['path'])
    for layer in model.layers:
        cell=getattr(layer,'cell',layer)
        if isinstance(cell,tf.keras.layers.BatchNormalization):layer.trainable=False;cell.trainable=False
    original=[v.numpy().copy() for v in model.variables]
    offline_model=helper.create(225);offline_model.load_weights(d['initial_snapshot']['path'])
    for layer in offline_model.layers:
        cell=getattr(layer,'cell',layer)
        if isinstance(cell,tf.keras.layers.BatchNormalization):layer.trainable=False;cell.trainable=False
    assert all(np.array_equal(a,v.numpy()) for a,v in zip(original,offline_model.variables,strict=True))
    stream=utils.to_streaming_inference(model,{'stride':3,'spectrogram_length':225},modes.Modes.STREAM_INTERNAL_STATE_INFERENCE)
    stream(np.zeros((1,3,40),dtype=np.float32),training=False)
    states=[v for v in stream.variables if 'states' in v.name];assert len(states)==6
    @tf.function(input_signature=[tf.TensorSpec([1,3,40],tf.float32)])
    def infer(x):return stream(x,training=False)
    @tf.function(input_signature=[tf.TensorSpec([1,225,40],tf.float32)])
    def offline(x):return offline_model(x,training=False)
    results=[]
    for i,r in enumerate(d['requests']):
        p=Path(r['path']);assert sha(p)==r['sha256'];pcm,rate=sf.read(p,dtype='int16');assert rate==16000 and pcm.ndim==1
        padded=np.pad(pcm,(48000,16000));raw=features(padded).astype('<u2')
        values=np.clip(raw.astype(np.float32)/25.6,0,26);values=np.pad(values,((0,(-len(values))%3),(0,0)))
        traces=[]
        for repeat in range(2):
            for state in states:state.assign(tf.zeros_like(state))
            trace=np.asarray([float(infer(values[None,j:j+3]).numpy()[0,0]) for j in range(0,len(values),3)],dtype=np.float32)
            assert np.isfinite(trace).all();traces.append(trace)
        assert np.array_equal(traces[0],traces[1])
        ends=list(range(225,len(values)+1,3))
        expected=np.asarray([float(offline(values[None,end-225:end]).numpy()[0,0]) for end in ends],dtype=np.float32)
        actual=traces[0][74:];assert len(expected)==len(actual)
        nominal_delta=np.abs(expected-actual)
        phase_expected=np.asarray([float(offline(np.concatenate((values[end-224:end],np.zeros((1,40),np.float32)),axis=0)[None]).numpy()[0,0]) for end in ends],dtype=np.float32)
        delta=np.abs(phase_expected-actual);maximum=float(np.max(delta))
        file=OUT/f'{i:02}-arrays.npz';np.savez(file,raw=raw,stream=traces[0],offline_nominal=expected,offline_phase=phase_expected,ends=np.asarray(ends,dtype=np.int64))
        assert maximum<=d['absolute_tolerance'],(r,maximum)
        results.append(dict(request=r,feature_frames=len(raw),stream_steps=len(traces[0]),compared_steps=len(expected),max_absolute_error=maximum,
            nominal_max_absolute_error=float(np.max(nominal_delta)),phase_unused_trailing_frames=1,twice_reset_exact=True,arrays_path=str(file),arrays_sha256=sha(file),pcm_samples=len(padded)))
    assert all(np.array_equal(a,v.numpy()) for a,v in zip(original,model.variables,strict=True))
    result=dict(schema='hey-context225-stream-probe-result-v3',plan_sha256=sha(PLAN),cases=results,
        input_shape=[1,3,40],output_shape=list(stream.output_shape),state_shapes=[list(v.shape) for v in states],
        all_phase_aware_offline_stream_comparisons_pass=True,source_weights_unchanged=True,updates=0,
        limits='CPU initial float model on6fulltrainingPCM only; no INT8/trained recognition or P4 equivalence claim.')
    with (RESULTS/'4090-hey-context225-stream-probe-v3.json').open('x') as f:f.write(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(cases=6,max_error=max(r['max_absolute_error'] for r in results),state_shapes=result['state_shapes'])),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['plan','run']);a=p.parse_args();plan() if a.action=='plan' else run()
