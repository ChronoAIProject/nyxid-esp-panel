"""Register and generate only the reserved development native PCM, never score.

120 original requests, no retries or replacement; any failure rejects whole pack.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
import onnxruntime as ort
from scipy.signal import resample_poly
import soundfile as sf
from generate_hey_paired_material_v2 import seeded_model,versions,code_files
from synthetic_output_quality_v1 import audit_output

RESULTS=Path('research/wakeword_p4/results')
CONTRACT=RESULTS/'4090-hey-development-render-contract-v1.json'
PLAN=RESULTS/'4090-hey-development-native-plan-v1.json'
OUT=Path('.local/wakeword/hey-paired-development-native-v1')
CANDIDATES=RESULTS/'4090-hey-v2-fixed-final-candidates-v1.json'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def register():
    c=json.loads(CONTRACT.read_text())
    candidates=json.loads(CANDIDATES.read_text())
    assert set(candidates['models'])=={'hey_willow','hey_nyx'}
    for model in candidates['models'].values():
        assert sha(model['path'])==model['sha256']
    assert len(c['requests'])==120 and all(r['role']=='development' for r in c['requests'])
    paths=code_files()+[Path(__file__),Path(__file__).with_name('generate_hey_paired_material_v2.py'),CONTRACT,CANDIDATES]
    plan=dict(schema='hey-development-native-plan-v1',contract_sha256=sha(CONTRACT),
              candidate_freeze_sha256=sha(CANDIDATES),versions=versions(),
              files={str(p):sha(p) for p in paths},voice=c['voice'],requests=c['requests'],
              noise_scale=.5,noise_w=.333,provider='CPUExecutionProvider',threads=2,
              authorization='Reserved development native generation only; no acceptance generation, detector scoring or threshold selection.',
              quality='Original audit_output, preserve engine/native/PCM; reject entire pack on any failure; no retry/substitution.',
              duration_interface='Preserve original engine tensors; separate original-graph VITS token allocation extraction required before feature/scoring admission; not phonetic truth.')
    with PLAN.open('x') as f:f.write(json.dumps(plan,indent=2)+'\n')
    print(json.dumps(dict(plan_sha256=sha(PLAN),requests=120,generated=False)))


def generate():
    remote=subprocess.check_output(['git','show','origin/research-hye-nyx-voice-gpu:'+str(PLAN)])
    assert remote==PLAN.read_bytes(),'Exact plan must be pushed before generation'
    plan=json.loads(remote)
    assert versions()==plan['versions']
    for path,digest in plan['files'].items():assert sha(path)==digest
    voice=plan['voice'];assert voice['role']=='development'
    for f in voice['files'].values():assert sha(f['path'])==f['sha256']
    OUT.mkdir(exist_ok=False);(OUT/'plan.json').write_bytes(remote)
    options=ort.SessionOptions();options.intra_op_num_threads=2;options.inter_op_num_threads=1
    options.execution_mode=ort.ExecutionMode.ORT_SEQUENTIAL
    passed=count=0
    with (OUT/'records.jsonl').open('x') as log:
        for req in plan['requests']:
            assert req['role']=='development' and req['voice']==voice['voice']
            row=dict(req,model_sha256=voice['files']['model']['sha256'],
                     config_sha256=voice['files']['config']['sha256'],native_rate=22050,
                     phoneme_durations=None,duration_interface=plan['duration_interface'],trimmed=False)
            try:
                model=seeded_model(voice['files']['model']['path'],req['seed'])
                row['derived_model_sha256']=hashlib.sha256(model).hexdigest()
                session=ort.InferenceSession(model,sess_options=options,providers=['CPUExecutionProvider'])
                tokens=np.asarray([req['token_ids']],dtype=np.int64)
                feed=dict(input=tokens,input_lengths=np.asarray([tokens.shape[1]],dtype=np.int64),
                          scales=np.asarray([plan['noise_scale'],req['length_scale'],plan['noise_w']],dtype=np.float32))
                if any(i.name=='sid' for i in session.get_inputs()):feed['sid']=np.asarray([req['speaker_id']],dtype=np.int64)
                outputs=session.run(None,feed);assert len(outputs)==1
                native=outputs[0]
                path=OUT/(req['id']+'.engine.npy');np.save(path,native,allow_pickle=False)
                row.update(engine_tensor_path=path.name,engine_tensor_sha256=sha(path),engine_output_shape=list(native.shape),inference_rerun=False)
                assert native.ndim in (3,4) and all(d==1 for d in native.shape[:-1]) and native.dtype==np.float32
                native=native.reshape(-1)
                raw=OUT/(req['id']+'.npy');np.save(raw,native,allow_pickle=False)
                row.update(raw_path=raw.name,raw_sha256=sha(raw),native_samples=len(native))
                pcm=resample_poly(native,320,441)
                wav=OUT/(req['id']+'.wav');sf.write(wav,pcm,16000,subtype='PCM_16')
                row.update(path=wav.name,sha256=sha(wav),samples=len(pcm),generated_reference_end_sample=len(pcm))
                row['audit']=audit_output(OUT,row);passed+=int(row['audit']['passed'])
            except Exception as exc:row['error']=f'{type(exc).__name__}: {exc}'
            count+=1;log.write(json.dumps(row,ensure_ascii=False)+'\n');log.flush()
            print(json.dumps(dict(recorded=count,passed=passed,error=row.get('error'))),flush=True)
    report=dict(schema='hey-development-native-completion-v1',plan_sha256=sha(PLAN),requested=120,
                recorded=count,passed=passed,pack_qualified=count==passed==120,
                records_sha256=sha(OUT/'records.jsonl'),acceptance_generated=False,wake_model_scoring=False,
                limits='Native mechanical quality only; rendered quality, original-graph token allocation, source history and scoring admission pending.')
    (OUT/'summary.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=('register','generate'))
    a=p.parse_args();{'register':register,'generate':generate}[a.action]()
