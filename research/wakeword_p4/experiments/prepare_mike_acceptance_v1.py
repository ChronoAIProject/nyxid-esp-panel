"""Reserve/lock/generate Mike acceptance audio only after frozen development passes."""
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import subprocess
import sys

import numpy as np
import soundfile as sf
from scipy.signal import fftconvolve,resample_poly

from calibrate_recovered_new_development_v1 import GRID,choose,eligible
from synthetic_output_quality_v1 import audit_output
from synthetic_render_quality_v1 import audit_render,RIR_DELAYS,RIR_AMPLITUDES

ROOT=Path('research/wakeword_p4')
BASE=Path('.local/wakeword')
RESERVATION=ROOT/'results/4090-mike-acceptance-generation-reservation-v1.json'
LOCK=ROOT/'results/4090-mike-acceptance-generation-lock-v1.json'
DEV=BASE/'recovered-new-development-v1'
DEV_PINS=ROOT/'results/4090-recovered-new-development-execution-pins-v1.json'
OUT=BASE/'mike-independent-acceptance-material-v1'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def versions():
    return {p:importlib.metadata.version(p) for p in ['onnxruntime','numpy','scipy','soundfile','piper-phonemize']}


def reserve():
    original=ROOT/'results/4090-joe-mike-generation-plan-v1.json'
    plan=json.loads(original.read_text())
    requests=[r for r in plan['requests'] if r['voice']=='mike']
    assert len(requests)==120 and all(r['role']=='acceptance' for r in requests)
    model=next(m for m in plan['models'] if m['voice']=='mike')
    assert sha(model['model_path'])==model['expected_sha256'] and sha(model['config_path'])==model['config_sha256']
    assert versions()==plan['versions']
    helpers=[ROOT/'experiments/synthetic_output_quality_v1.py',ROOT/'experiments/synthetic_render_quality_v1.py',
             ROOT/'experiments/joe_mike_selection_protocol_v1.md',ROOT/'experiments/mike_acceptance_material_protocol_v1.md',
             ROOT/'results/4090-prospective-background-role-plan-v1.json',original]
    report=dict(schema='mike-acceptance-material-reservation-v1',requests=requests,model=model,
        source_sha256=sha(__file__),versions=versions(),files={str(p):sha(p) for p in helpers},
        lead_samples=28800,tail_samples=16000,rir=dict(delays=RIR_DELAYS,amplitudes=RIR_AMPLITUDES),
        render_seed_rule='first16hex SHA256(request_id + :mike-acceptance-render-v1: + variant), integer',
        noise_scale=.5,noise_w=.333,random_seed=None,phoneme_durations=None,
        native_output_shape_policy='Require original float32 [1,1,1,N], save entire engine tensor before checks; remove singleton axes with exact full sample-byte identity',
        authorization='Reservation only, no generation/scoring until development-passed lock is committed and pushed',
        shared_parent='Joe/Mike share rdh; distinct voice/model roles, not parent/physical/pretraining independence')
    with RESERVATION.open('x') as f:json.dump(report,f,indent=2);f.write('\n')
    print(json.dumps(dict(reservation_sha256=sha(RESERVATION),requests=120,audio_generated=False)))


def validate_selection(calibration,minimum):
    """Reject failed/stale/null/noncanonical development decisions."""
    assert calibration['minimum_macro']==minimum
    assert [p['threshold'] for p in calibration['grid']]==GRID
    for p in calibration['grid']:
        assert p['eligible']==eligible(p['synthetic'],p['background_upper95'],minimum)
    selected=choose(calibration['grid'])
    assert selected is not None and calibration['viable_operating_point'] is True
    assert calibration['threshold']==selected['threshold'] and calibration['selected']==selected
    assert not calibration['candidate_promoted'] and not calibration['acceptance_authorized']
    return selected


def write_lock():
    # All missing/failed-development checks occur before reading Mike PCM,
    # creating an inference session, or making an acceptance output directory.
    for p in [RESERVATION,DEV_PINS,DEV/'report.json',DEV/'calibration.json']:
        if not p.is_file():raise RuntimeError('Acceptance remains unopened; missing development prerequisite: '+str(p))
    reservation=json.loads(RESERVATION.read_text());assert sha(__file__)==reservation['source_sha256']
    assert versions()==reservation['versions']
    for p,h in reservation['files'].items():assert sha(p)==h
    pins=json.loads(DEV_PINS.read_text());development=json.loads((DEV/'report.json').read_text())
    calibration=json.loads((DEV/'calibration.json').read_text())
    for p,e in pins['files'].items():assert sha(p)==e['sha256']
    assert development['execution_pins_sha256']==sha(DEV_PINS)
    assert calibration['execution_pins_sha256']==sha(DEV_PINS)
    assert calibration['candidate_report_sha256']==sha(DEV/'report.json')
    assert calibration['source_sha256']==pins['calibrator_sha256']
    assert calibration['model_sha256']==development['model_sha256']==pins['model_sha256']
    baseline=json.loads(Path('/tmp/nyx-joe-base-v1/report.json').read_text())
    selected=validate_selection(calibration,max(.8,baseline['summaries']['v30']['macro']))
    paths=[RESERVATION,DEV_PINS,DEV/'report.json',DEV/'calibration.json']
    report=dict(schema='development-passed-mike-material-lock-v1',files={str(p):sha(p) for p in paths},
        source_sha256=sha(__file__),model_sha256=calibration['model_sha256'],threshold=selected['threshold'],
        authorizes='Exactly120Mike native outputs and360mechanical renders only; no wake-model scores, acceptance background download or promotion',
        candidate_scoring_authorized=False,background_access_authorized=False)
    with LOCK.open('x') as f:json.dump(report,f,indent=2);f.write('\n')
    print(json.dumps(dict(lock_sha256=sha(LOCK),threshold=report['threshold'],audio_generated=False)))


def generate():
    if not LOCK.is_file():raise RuntimeError('Acceptance remains unopened; no development-passed committed lock')
    lock=json.loads(LOCK.read_text());assert sha(__file__)==lock['source_sha256']
    for p,h in lock['files'].items():assert sha(p)==h
    # Verify actual published lock bytes, not a timestamp or a local intent flag.
    published=subprocess.check_output(['git','show','refs/remotes/origin/research-hye-nyx-voice-gpu:'+str(LOCK)])
    assert published==LOCK.read_bytes()
    reservation=json.loads(RESERVATION.read_text());assert reservation['source_sha256']==sha(__file__)
    assert versions()==reservation['versions']
    for p,h in reservation['files'].items():assert sha(p)==h
    model=reservation['model']
    assert sha(model['model_path'])==model['expected_sha256'] and sha(model['config_path'])==model['config_sha256']
    OUT.mkdir(exist_ok=False);(OUT/'lock.json').write_bytes(LOCK.read_bytes());(OUT/'reservation.json').write_bytes(RESERVATION.read_bytes())
    # Imported lazily: blocked admission cannot initialize the TTS engine.
    import onnxruntime as ort
    options=ort.SessionOptions();options.intra_op_num_threads=2;options.inter_op_num_threads=1
    options.execution_mode=ort.ExecutionMode.ORT_SEQUENTIAL
    session=ort.InferenceSession(model['model_path'],sess_options=options,providers=['CPUExecutionProvider'])
    native_rows=[];renders=[];ir=np.zeros(961);ir[RIR_DELAYS]=RIR_AMPLITUDES
    with (OUT/'native-records.jsonl').open('x') as native_log,(OUT/'render-records.jsonl').open('x') as render_log:
        for request in reservation['requests']:
            row=dict(request,native_rate=model['native_rate'],trimmed=False,model_sha256=model['expected_sha256'],
                     config_sha256=model['config_sha256'],random_seed=None,phoneme_durations=None,
                     duration_interface='Original ONNX exposes waveform only')
            try:
                tokens=np.asarray([request['token_ids']],dtype=np.int64)
                outputs=session.run(None,dict(input=tokens,input_lengths=np.asarray([tokens.shape[1]],dtype=np.int64),
                    scales=np.asarray([.5,request['length_scale'],.333],dtype=np.float32)))
                assert len(outputs)==1
                tensor=outputs[0];p=OUT/(request['id']+'.engine.npy');np.save(p,tensor,allow_pickle=False)
                row.update(engine_tensor_path=p.name,engine_tensor_sha256=sha(p),engine_output_shape=list(tensor.shape))
                assert tensor.dtype==np.float32 and tensor.ndim==4 and tensor.shape[:3]==(1,1,1)
                native=tensor[0,0,0,:];assert native.tobytes()==tensor.tobytes()
                p=OUT/(request['id']+'.npy');np.save(p,native,allow_pickle=False)
                row.update(raw_path=p.name,raw_sha256=sha(p),native_samples=len(native),singleton_removal_sample_bytes_exact=True)
                assert np.isfinite(native).all()
                divisor=math.gcd(model['native_rate'],16000)
                full=resample_poly(native,16000//divisor,model['native_rate']//divisor)
                p=OUT/(request['id']+'-native.wav');sf.write(p,full,16000,subtype='PCM_16')
                row.update(path=p.name,sha256=sha(p),samples=len(full),generated_reference_end_sample=len(full))
                row['audit']=audit_output(OUT,row)
                source,rate=sf.read(p,dtype='int16');assert rate==16000
                dry=source.astype(np.float64)/32768
                for variant in range(3):
                    presentation=dict(request,role='independent-synthetic-acceptance-v1',split='testing',variant=variant,
                        source_samples=len(source),source_sha256=row['sha256'],lead_samples=28800,tail_samples=16000)
                    try:
                        seed=int(hashlib.sha256((request['id']+':mike-acceptance-render-v1:'+str(variant)).encode()).hexdigest()[:16],16)
                        speech=fftconvolve(dry,ir,mode='full') if variant==2 else dry
                        signal=np.pad(speech,(28800,16000));snr=None if variant==0 else 25 if variant==1 else 15
                        if variant:
                            noise=np.random.default_rng(seed).normal(0,1,len(signal))
                            noise*=np.sqrt(np.mean(dry**2))*10**(-snr/20)/np.sqrt(np.mean(noise**2));signal+=noise
                        gain=min(1.,.95/max(float(np.max(np.abs(signal))),1e-8)) if variant else 1.
                        signal*=gain;p=OUT/(request['id']+f'-v{variant}.wav');sf.write(p,signal,16000,subtype='PCM_16')
                        pcm,rate=sf.read(p,dtype='int16');assert rate==16000
                        presentation.update(path=p.name,sha256=sha(p),pcm_sha256=hashlib.sha256(pcm.astype('<i2').tobytes()).hexdigest(),
                            seed=seed,samples=len(pcm),snr_db=snr,gain_after_mix=gain,synthetic_rir=variant==2,
                            rir_tail_samples=960 if variant==2 else 0,native_generation_end=28800+len(source),keyword_end=28800+len(speech),
                            generated_output_end_sample=28800+len(source),full_rir_end_sample=28800+len(speech),
                            boundary_reference='Complete generated output/RIR, not phonetic wake-prefix endpoint')
                        presentation['audit']=audit_render(source,pcm,presentation)
                    except Exception as exc:presentation['error']=f'{type(exc).__name__}: {exc}'
                    renders.append(presentation);render_log.write(json.dumps(presentation)+'\n');render_log.flush()
            except Exception as exc:row['error']=f'{type(exc).__name__}: {exc}'
            native_rows.append(row);native_log.write(json.dumps(row)+'\n');native_log.flush()
            print(json.dumps(dict(native_completed=len(native_rows),renders_recorded=len(renders))),flush=True)
    passed=(len(native_rows)==120 and len(renders)==360 and all(r.get('audit',{}).get('passed',False) for r in native_rows+renders))
    summary=dict(reservation_sha256=sha(RESERVATION),lock_sha256=sha(LOCK),native_requested=120,native_recorded=len(native_rows),
                 renders_requested=360,renders_recorded=len(renders),mechanically_qualified=passed,
                 native_records_sha256=sha(OUT/'native-records.jsonl'),render_records_sha256=sha(OUT/'render-records.jsonl'),
                 wake_models_scored=False,background_accessed=False,scoring_authorized=False,candidate_promoted=False)
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary))
    if not passed:raise SystemExit(1)


if __name__=='__main__':
    if sys.argv[1:]==['--reserve']:reserve()
    elif sys.argv[1:]==['--write-lock']:write_lock()
    elif sys.argv[1:]==['--generate']:generate()
    else:raise SystemExit('Use --reserve, --write-lock or --generate')
