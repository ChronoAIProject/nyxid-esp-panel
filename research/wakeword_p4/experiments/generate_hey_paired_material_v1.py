"""Lock native/render schedules, then synthesize training only; never score models."""
import argparse
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import subprocess

import numpy as np
import onnx
import onnxruntime as ort
from piper_phonemize import phonemize_espeak
import piper_phonemize
import piper_phonemize_cpp
from scipy.signal import resample_poly, fftconvolve
import soundfile as sf

from synthetic_output_quality_v1 import audit_output
from synthetic_render_quality_v1 import RIR_DELAYS, RIR_AMPLITUDES, audit_render

RESULTS = Path('research/wakeword_p4/results')
PLAN = RESULTS/'4090-hey-paired-generation-plan-v1.json'
OUT = Path('.local/wakeword/hey-paired-training-native-v1')
RESERVATION = RESULTS/'4090-hey-paired-voice-role-reservation-v1.json'
TEXTS = [('hey_willow','Hey Willow'),('hey_nyx','Hey Nicks'),
         ('common_negative','Hey'),('common_negative','Hello'),
         ('common_negative','Hey William'),('common_negative','Hey pillow'),
         ('common_negative','Hey Nick'),('common_negative','Hey next'),
         ('hi_esp','Hi E S P'),('common_negative','Hey Muse')]
CARRIERS = ['', 'turn on the light', 'what time is it', 'set a timer']


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def versions():
    return {k:importlib.metadata.version(k) for k in
            ('onnx','onnxruntime','numpy','scipy','soundfile','piper-phonemize')}


def code_files():
    files = [Path(__file__), Path(__file__).with_name('synthetic_output_quality_v1.py'),
             Path(__file__).with_name('synthetic_render_quality_v1.py'),
             Path(piper_phonemize.__file__), Path(piper_phonemize_cpp.__file__)]
    files += sorted(p for p in (Path(piper_phonemize.__file__).parent/'espeak-ng-data').rglob('*') if p.is_file())
    return files


def seeded_model(path, seed):
    """Explicit derived ONNX: only random-op seeds change, not learned weights."""
    model = onnx.load(path)
    nodes = [n for n in model.graph.node if n.op_type == 'RandomNormalLike']
    if len(nodes) != 2:
        raise ValueError('Unexpected stochastic graph; require new preregistration')
    for i,node in enumerate(nodes):
        attrs = [a for a in node.attribute if a.name != 'seed']
        del node.attribute[:]
        node.attribute.extend(attrs)
        node.attribute.append(onnx.helper.make_attribute('seed',float(seed+100000*(i+1))))
    return model.SerializeToString()


def make_plan():
    reservation = json.loads(RESERVATION.read_text())
    requests = []
    for vi,voice in enumerate(reservation['roles']):
        for f in voice['files'].values():
            assert sha(f['path']) == f['sha256']
        config = json.loads(Path(voice['files']['config']['path']).read_text())
        ids = config['phoneme_id_map']
        draws = 10 if voice['role']=='training' else 1
        for ti,(target,literal) in enumerate(TEXTS):
            for ci,carrier in enumerate(CARRIERS):
                text = literal+(', '+carrier if carrier else '')+'.'
                sentences = phonemize_espeak(text,config['espeak']['voice'])
                if len(sentences)!=1:
                    raise ValueError('Expected one full sentence')
                phones=sentences[0]
                tokens=list(ids['^'])+list(ids['_'])
                for phone in phones:
                    tokens+=list(ids[phone])+list(ids['_'])
                tokens+=list(ids['$'])
                for si,speed in enumerate((.9,1.,1.1)):
                    for draw in range(draws):
                        seed=1726+vi*1000000+ci*1000+si*100+draw
                        requests.append(dict(id=f'v{vi}-t{ti:02d}-c{ci}-s{si}-r{draw:02d}',
                            voice=voice['voice'],role=voice['role'],speaker_id=voice['speaker_id'],
                            target_id=target,literal=literal,text=text,carrier=carrier,
                            phonemes=phones,token_ids=tokens,length_scale=speed,
                            seed=seed,render_seeds=[seed+9000000+i*100000 for i in range(3)]))
    counts={role:sum(r['role']==role for r in requests) for role in ('training','development','acceptance')}
    assert counts==dict(training=2400,development=120,acceptance=120)
    plan=dict(schema='hey-paired-native-render-generation-plan-v1',
              reservation_sha256=sha(RESERVATION),voices=reservation['roles'],versions=versions(),
              files={str(p):sha(p) for p in code_files()},requests=requests,counts=counts,
              provider='CPUExecutionProvider',threads=2,noise_scale=.5,noise_w=.333,
              random_policy='Derived ONNX RandomNormalLike nodes explicit float seed=request.seed+100000*(node_index+1); all learned tensors unchanged; derived model SHA saved per request',
              phoneme_durations=None,duration_interface='ONNX waveform output only',
              rendering=dict(conditions=['quiet','25dB_noise','15dB_noise_full_RIR'],
                  rir_delays=RIR_DELAYS,rir_amplitudes=RIR_AMPLITUDES,
                  leading_samples=28800,trailing_samples=16000,
                  noise='numpy.default_rng(render_seed).normal; dry-source RMS reference',
                  gain='quiet=1; noisy=min(1,.95/peak)',full_convolution=True),
              native_quality='synthetic_output_quality_v1.POLICY; any failure disqualifies full planned pack, no retries/substitution',
              authorized_stage='training native+render only after pushed plan; development/acceptance reserved, not generated',
              training_admitted=False,background_inventories_locked=False,
              limits='Stage-specific synthesis lock only; no scoring/training admission until independent backgrounds, PCM overlap, features and complete training schedules are locked. Mechanical gates do not prove phonetic truth.')
    with PLAN.open('x') as f:
        f.write(json.dumps(plan,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps(dict(counts=counts,plan_sha256=sha(PLAN),generated=False)))


def generate_training():
    remote=subprocess.check_output(['git','show','origin/research-hye-nyx-voice-gpu:'+str(PLAN)])
    assert remote==PLAN.read_bytes(), 'Push exact plan before synthesis'
    plan=json.loads(remote)
    assert versions()==plan['versions'] and sha(RESERVATION)==plan['reservation_sha256']
    for path,digest in plan['files'].items():assert sha(path)==digest
    voices={v['voice']:v for v in plan['voices']}
    for v in voices.values():
        for f in v['files'].values():assert sha(f['path'])==f['sha256']
    OUT.mkdir(exist_ok=False)
    (OUT/'plan.json').write_bytes(remote)
    options=ort.SessionOptions();options.intra_op_num_threads=2;options.inter_op_num_threads=1
    options.execution_mode=ort.ExecutionMode.ORT_SEQUENTIAL
    passed=0;recorded=0
    with (OUT/'records.jsonl').open('x') as log:
        for req in plan['requests']:
            if req['role']!='training':continue
            voice=voices[req['voice']]
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
                if any(i.name=='sid' for i in session.get_inputs()):
                    feed['sid']=np.asarray([req['speaker_id']],dtype=np.int64)
                outputs=session.run(None,feed)
                assert len(outputs)==1
                native=outputs[0]
                path=OUT/(req['id']+'.engine.npy');np.save(path,native,allow_pickle=False)
                row.update(engine_tensor_path=path.name,engine_tensor_sha256=sha(path),engine_output_shape=list(native.shape))
                assert native.ndim==3 and native.shape[:2]==(1,1) and native.dtype==np.float32
                native=native[0,0,:]
                raw=OUT/(req['id']+'.npy');np.save(raw,native,allow_pickle=False)
                row.update(raw_path=raw.name,raw_sha256=sha(raw),native_samples=len(native))
                pcm=resample_poly(native,320,441)
                wav=OUT/(req['id']+'.wav');sf.write(wav,pcm,16000,subtype='PCM_16')
                row.update(path=wav.name,sha256=sha(wav),samples=len(pcm),generated_reference_end_sample=len(pcm))
                row['audit']=audit_output(OUT,row)
                passed+=int(row['audit']['passed'])
            except Exception as exc:
                row['error']=f'{type(exc).__name__}: {exc}'
            recorded+=1;log.write(json.dumps(row,ensure_ascii=False)+'\n');log.flush()
            if recorded%20==0 or 'error' in row:
                print(json.dumps(dict(recorded=recorded,passed=passed,error=row.get('error'))),flush=True)
    report=dict(schema='hey-paired-training-native-summary-v1',plan_sha256=sha(PLAN),
                requested=plan['counts']['training'],recorded=recorded,passed=passed,
                pack_qualified=recorded==passed==plan['counts']['training'],records_sha256=sha(OUT/'records.jsonl'),
                development_generated=False,acceptance_generated=False,wake_model_scoring=False,
                limits='Native mechanical quality only; rendering/overlap/feature/full training admission pending')
    (OUT/'summary.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)


def render_training():
    remote=subprocess.check_output(['git','show','origin/research-hye-nyx-voice-gpu:'+str(PLAN)])
    assert remote==PLAN.read_bytes()
    plan=json.loads(remote)
    for path,digest in plan['files'].items():assert sha(path)==digest
    assert versions()==plan['versions']
    summary=json.loads((OUT/'summary.json').read_text())
    assert summary['pack_qualified'] and summary['plan_sha256']==sha(PLAN)
    assert summary['records_sha256']==sha(OUT/'records.jsonl')
    dest=Path('.local/wakeword/hey-paired-training-rendered-v1')
    dest.mkdir(exist_ok=False)
    count=0;passed=0
    ir=np.zeros(961);ir[RIR_DELAYS]=RIR_AMPLITUDES
    with (dest/'records.jsonl').open('x') as log:
        for line in (OUT/'records.jsonl').open():
            source=json.loads(line)
            assert source['role']=='training' and audit_output(OUT,source)['passed']
            pcm,rate=sf.read(OUT/source['path'],dtype='int16');assert rate==16000
            dry=pcm.astype(np.float64)/32768
            for variant in range(3):
                row=dict(source,variant=variant,seed=source['render_seeds'][variant],
                         source_samples=len(pcm),source_path=str(OUT/source['path']),
                         source_sha256=source['sha256'],
                         source_pcm_sha256=hashlib.sha256(pcm.astype('<i2').tobytes()).hexdigest())
                speech=fftconvolve(dry,ir,mode='full') if variant==2 else dry
                values=np.pad(speech,(28800,16000));snr=None if variant==0 else 25 if variant==1 else 15
                if variant:
                    noise=np.random.default_rng(row['seed']).normal(0,1,len(values))
                    noise*=np.sqrt(np.mean(dry**2))*10**(-snr/20)/np.sqrt(np.mean(noise**2))
                    values+=noise
                gain=min(1.,.95/max(float(np.max(np.abs(values))),1e-8)) if variant else 1.
                values*=gain
                path=dest/(source['id']+f'-n{variant}.wav')
                sf.write(path,values,16000,subtype='PCM_16')
                actual,rate=sf.read(path,dtype='int16')
                row.update(path=path.name,sha256=sha(path),samples=len(actual),
                           native_generation_end=28800+len(pcm),keyword_end=28800+len(speech),
                           boundary_kind='full generated container including command, not wake phoneme',
                           rir_tail_samples=960 if variant==2 else 0,lead_samples=28800,tail_samples=16000,
                           snr_db=snr,synthetic_rir=variant==2,gain_after_mix=gain)
                row['render_audit']=audit_render(pcm,actual,row)
                count+=1;passed+=int(row['render_audit']['passed'])
                log.write(json.dumps(row)+'\n');log.flush()
            if count%60==0:print(json.dumps(dict(rendered=count,passed=passed)),flush=True)
    report=dict(plan_sha256=sha(PLAN),requested=7200,recorded=count,passed=passed,
                pack_qualified=count==passed==7200,records_sha256=sha(dest/'records.jsonl'),
                development_generated=False,acceptance_generated=False,wake_model_scoring=False)
    (dest/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=('write-plan','generate-training','render-training'))
    a=p.parse_args()
    {'write-plan':make_plan,'generate-training':generate_training,'render-training':render_training}[a.action]()
