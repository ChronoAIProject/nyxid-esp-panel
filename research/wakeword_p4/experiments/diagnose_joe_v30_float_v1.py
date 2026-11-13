"""Same-original-checkpoint float diagnostic on consumed Joe only, no tuning."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='-1'
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys

import numpy as np
import soundfile as sf
import tensorflow as tf

ROOT=Path('research/wakeword_p4')
sys.path.insert(0,str(ROOT/'training'))
from evaluate import Detector,smooth,event_indices
from features import features

MODEL=Path('.local/wakeword/model-gpu-v30-ami-extra')
MANIFEST=Path('.local/wakeword/joe-development-render-v1/replay-manifest.json')
BASELINE=Path('/tmp/nyx-joe-base-v1')
OUT=Path('.local/wakeword/joe-v30-float-int8-diagnostic-v1')
PINS=ROOT/'results/4090-joe-v30-float-diagnostic-pins-v1.json'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def versions():
    return {p:importlib.metadata.version(p) for p in ['tensorflow','numpy','ai-edge-litert','pymicro-features','soundfile']}


def freeze():
    audit=ROOT/'results/4090-v30-original-float-authentication-v1.json'
    authentication=json.loads(audit.read_text())
    files=[MODEL/'training.json',MODEL/'hey_nyx.tflite',audit,MANIFEST,BASELINE/'report.json',
           ROOT/'training/features.py',ROOT/'training/evaluate.py',
           ROOT/'experiments/joe_v30_float_diagnostic_protocol_v1.md']
    for name,digest in authentication['original_saved_model_files'].items():
        p=MODEL/name;assert sha(p)==digest;files.append(p)
    assert sha(MODEL/'hey_nyx.tflite')==authentication['original_int8_sha256']
    assert sha(MODEL/'training.json')==authentication['original_training_sha256']
    assert sha(MANIFEST)=='edf6f7844f654d2a7e22f588873b917381aa8d0957b1e3309a60a0f1973895bc'
    baseline=json.loads((BASELINE/'report.json').read_text())
    assert sha(BASELINE/'report.json')=='c87278f3fd009a0b4daa02aab1e23e6a15c32c2c2e2af5c7fbafba25f3c5a281'
    assert len(baseline['records'])==360
    for i,row in enumerate(baseline['records']):
        p=BASELINE/f'{i}-v30-scores.npy';assert sha(p)==row['v30']['score_sha256'];files.append(p)
    report=dict(schema='consumed-joe-original-v30-float-diagnostic-pins-v1',
                files={str(p):dict(size=p.stat().st_size,sha256=sha(p)) for p in files},
                source_sha256=sha(__file__),versions=versions(),threshold=.91796875,smoothing=5,
                clips=360,cpu_threads=2,
                scope='Consumed Joe DEVELOPMENT diagnostic only; no candidate/Mike/acceptance/threshold selection')
    with PINS.open('x') as f:json.dump(report,f,indent=2);f.write('\n')
    print(json.dumps(dict(pins_sha256=sha(PINS),new_inference=False,clips=360)))


def run():
    pins=json.loads(PINS.read_text());assert pins['source_sha256']==sha(__file__) and pins['versions']==versions()
    for p,e in pins['files'].items():assert sha(p)==e['sha256'] and Path(p).stat().st_size==e['size']
    tf.config.set_visible_devices([],'GPU')
    tf.config.threading.set_intra_op_parallelism_threads(2);tf.config.threading.set_inter_op_parallelism_threads(2)
    loaded=tf.saved_model.load(str(MODEL/'saved_model'));signature=loaded.signatures['serving_default']
    states=[v for v in loaded.variables if 'states' in v.name];assert states
    detector=Detector(MODEL/'hey_nyx.tflite');scale,zero=detector.input['quantization']
    assert detector.input['dtype']==np.int8 and detector.output['dtype']==np.uint8 and scale>0
    def predict(x):
        for state in states:state.assign(tf.zeros_like(state))
        values=[]
        for at in range(0,len(x)-2,3):
            output=signature(input_audio=tf.convert_to_tensor(x[None,at:at+3],dtype=tf.float32))
            values.append(float(output['output_0'].numpy().reshape(-1)[0]))
        result=np.asarray(values,dtype=np.float32)
        assert np.isfinite(result).all() and np.all((0<=result)&(result<=1))
        return result
    def events(scores):return [float((3*(i+4)+5)*.01) for i in event_indices(scores,pins['threshold'])]
    OUT.mkdir(exist_ok=False);(OUT/'pins.json').write_bytes(PINS.read_bytes())
    original=json.loads((BASELINE/'report.json').read_text());rows=json.loads(MANIFEST.read_text())['records']
    assert [r['record'] for r in original['records']]==rows
    arrays={};details=[]
    with (OUT/'records.jsonl').open('x') as log:
        for i,row in enumerate(rows):
            p=MANIFEST.parent/row['path'];assert sha(p)==row['sha256']
            pcm,rate=sf.read(p,dtype='int16');assert rate==16000 and len(pcm)==row['samples']
            assert hashlib.sha256(pcm.astype('<i2').tobytes()).hexdigest()==row['pcm_sha256']
            raw=features(pcm);x=np.clip(raw.astype(np.float32)/25.6,0,26)
            q=np.clip(np.rint(x/scale)+zero,-128,127).astype(np.int8)
            rounded=(q.astype(np.float32)-zero)*np.float32(scale)
            f=predict(x);rounded_f=predict(rounded)
            if i==0:
                np.testing.assert_array_equal(f,predict(x));np.testing.assert_array_equal(rounded_f,predict(rounded))
            s=np.load(BASELINE/f'{i}-v30-scores.npy',allow_pickle=False)
            assert s.shape==smooth(f,5).shape==smooth(rounded_f,5).shape
            assert len(s)==((len(pcm)+159)//160-2)//3-4
            arms={'float_original':smooth(f,5),'float_input_rounded':smooth(rounded_f,5),'int8':s}
            result=dict(record=row,arms={},float_raw_count=len(f),
                        original_frontend_sha256=hashlib.sha256(raw.tobytes()).hexdigest(),
                        original_int8_input_sha256=hashlib.sha256(q.tobytes()).hexdigest())
            arrays[f'float_raw_{i}']=f;arrays[f'float_rounded_raw_{i}']=rounded_f
            for name,values in arms.items():
                arrays[f'{name}_smoothed_{i}']=values;times=events(values)
                result['arms'][name]=dict(event_seconds=times,post_source_events=[t for t in times if t>=1.8],
                    peak_five_mean=float(values.max()),score_sha256=hashlib.sha256(values.tobytes()).hexdigest(),
                    mae_vs_original_int8=float(np.mean(np.abs(values-s))),max_error_vs_original_int8=float(np.max(np.abs(values-s))))
            assert result['arms']['int8']['event_seconds']==original['records'][i]['v30']['event_seconds']
            details.append(result);log.write(json.dumps(result)+'\n');log.flush()
            if i%15==0:print(json.dumps(dict(completed=i+1,planned=360)),flush=True)
    np.savez_compressed(OUT/'scores.npz',**arrays)
    summaries={}
    for arm in ['float_original','float_input_rounded','int8']:
        summaries[arm]={}
        for target in ['hey_nyx','hi_esp','confusion']:
            group=[r for r in details if r['record']['target']==target]
            summaries[arm][target]=dict(clips=len(group),post_source_activated=sum(bool(r['arms'][arm]['post_source_events']) for r in group),
                any_event_clips=sum(bool(r['arms'][arm]['event_seconds']) for r in group),
                maximum_five_mean=max(r['arms'][arm]['peak_five_mean'] for r in group),
                pre_source_events=sum(sum(t<1.8 for t in r['arms'][arm]['event_seconds']) for r in group))
    report=dict(scope=pins['scope'],pins_sha256=sha(PINS),source_sha256=sha(__file__),versions=versions(),
                original_float_files_authenticated=True,first_stream_both_float_reset_repeated_exact=True,
                input_scale=scale,input_zero_point=zero,threshold=pins['threshold'],smoothing=5,
                scores_npz_sha256=sha(OUT/'scores.npz'),records_sha256=sha(OUT/'records.jsonl'),summaries=summaries,
                candidate_scored=False,acceptance_scored=False,threshold_selected=False,
                limits='Consumed single-Joe diagnostic only. Same selected original float versus original INT8; float-rounded contrast isolates input rounding inside float graph, not a complete additive causal decomposition. Other differences include weight/activation/state/runtime quantization. Frozen INT8 operating point is descriptive for float; no float optimum/FAR or human/phonetic/hardware claim. No candidate choice/gate change.')
    (OUT/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(summaries),flush=True)


if __name__=='__main__':
    if sys.argv[1:]==['--write-pins']:freeze()
    elif sys.argv[1:]==['--run']:run()
    else:raise SystemExit('Use --write-pins or --run')
