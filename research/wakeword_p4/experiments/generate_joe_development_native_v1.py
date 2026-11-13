"""Execute the preregistered Joe-only native synthesis; never score wake models."""
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import sys

import numpy as np
import onnxruntime as ort
from piper_phonemize import phonemize_espeak
from scipy.signal import resample_poly
import soundfile as sf

from synthetic_output_quality_v1 import audit_output

ROOT = Path('research/wakeword_p4')
PLAN = ROOT / 'results/4090-joe-mike-generation-plan-v1.json'
OUT = Path('.local/wakeword/joe-development-native-v1')
TEXTS = [
    ('hey_nyx', 'Hey Nicks.'), ('hey_nyx', 'Hey Nicks!'),
    ('hey_nyx', 'Hey Nicks, what time is it?'),
    ('hey_nyx', 'Hey Nicks, turn on the light.'),
    ('hi_esp', 'Hi E S P.'), ('hi_esp', 'Hi E S P!'),
    ('hi_esp', 'Hi E S P, what time is it?'),
    ('hi_esp', 'Hi E S P, turn on the light.'),
    ('confusion', 'Hey Nick.'), ('confusion', 'Hey Next.'),
    ('confusion', 'Hey Mix.'), ('confusion', 'Hey Nexus.'),
    ('confusion', 'Hey Fix.'), ('confusion', 'Hey Nice.'),
    ('confusion', 'Hey.'), ('confusion', 'Nicks.'),
    ('confusion', 'Hi.'), ('confusion', 'E S P.'),
    ('confusion', 'High speed.'), ('confusion', 'Hi E S B.'),
]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def versions():
    return {p: importlib.metadata.version(p) for p in
            ['onnxruntime', 'numpy', 'scipy', 'soundfile', 'piper-phonemize']}


def make_plan():
    inventory = ROOT / 'results/4090-prospective-joe-mike-source-plan-v1.json'
    models = json.loads(inventory.read_text())['models']
    requests = []
    for model in models:
        assert sha(model['model_path']) == model['expected_sha256']
        assert sha(model['config_path']) == model['config_sha256']
        config = json.loads(Path(model['config_path']).read_text())
        ids = config['phoneme_id_map']
        for ti, (target, text) in enumerate(TEXTS):
            sentences = phonemize_espeak(text, config['espeak']['voice'])
            assert len(sentences) == 1
            phones = sentences[0]
            tokens = list(ids['^']) + list(ids['_'])
            for phone in phones:
                tokens += list(ids[phone]) + list(ids['_'])
            tokens += list(ids['$'])
            for speed in [.9, 1., 1.1]:
                for draw in range(2):
                    requests.append(dict(id=f"{model['voice']}-t{ti:02d}-s{speed:.1f}-r{draw}",
                                         voice=model['voice'], role='development' if model['voice']=='joe' else 'acceptance',
                                         target=target, text=text, phonemes=phones,
                                         token_ids=tokens, length_scale=speed, draw=draw))
    plan = dict(schema='joe-mike-native-generation-reservation-v1',
                inventory_sha256=sha(inventory), models=models, versions=versions(),
                generator_sha256=sha(__file__),
                quality_source_sha256=sha(ROOT/'experiments/synthetic_output_quality_v1.py'),
                protocol_sha256=sha(ROOT/'experiments/joe_mike_generation_protocol_v1.md'),
                provider='CPUExecutionProvider', threads=2, noise_scale=.5, noise_w=.333,
                random_seed=None, exposed_durations=None, requests=requests,
                authorization='Joe native development only; no detector scoring or Mike inference',
                shared_parent='rdh; voice/model holdout, not parent-family independence')
    with PLAN.open('x') as f:
        json.dump(plan,f,indent=2,ensure_ascii=False); f.write('\n')
    print(json.dumps(dict(requests=len(requests),plan_sha256=sha(PLAN),audio_generated=False)))


def generate():
    plan = json.loads(PLAN.read_text())
    assert versions() == plan['versions']
    assert sha(__file__) == plan['generator_sha256']
    assert sha(ROOT/'experiments/synthetic_output_quality_v1.py') == plan['quality_source_sha256']
    assert sha(ROOT/'experiments/joe_mike_generation_protocol_v1.md') == plan['protocol_sha256']
    OUT.mkdir(exist_ok=False)
    (OUT/'plan.json').write_bytes(PLAN.read_bytes())
    (OUT/'generator.py').write_bytes(Path(__file__).read_bytes())
    model = next(m for m in plan['models'] if m['voice']=='joe')
    assert sha(model['model_path']) == model['expected_sha256']
    assert sha(model['config_path']) == model['config_sha256']
    options = ort.SessionOptions()
    options.intra_op_num_threads=2; options.inter_op_num_threads=1
    options.execution_mode=ort.ExecutionMode.ORT_SEQUENTIAL
    session = ort.InferenceSession(model['model_path'], sess_options=options,
                                   providers=['CPUExecutionProvider'])
    requests = [r for r in plan['requests'] if r['voice']=='joe']
    rows = []
    with (OUT/'records.jsonl').open('x') as log:
        for request in requests:
            row = dict(request, model_sha256=model['expected_sha256'],
                       config_sha256=model['config_sha256'], random_seed=None,
                       phoneme_durations=None, duration_interface='ONNX waveform output only',
                       native_rate=model['native_rate'], trimmed=False)
            try:
                tokens=np.asarray([request['token_ids']],dtype=np.int64)
                outputs=session.run(None,dict(input=tokens,input_lengths=np.asarray([tokens.shape[1]],dtype=np.int64),
                    scales=np.asarray([plan['noise_scale'],request['length_scale'],plan['noise_w']],dtype=np.float32)))
                assert len(outputs)==1
                native=outputs[0]
                row['engine_output_shape']=list(native.shape)
                # Preserve the exact full engine tensor as well as its mono vector.
                tensor_path=OUT/(request['id']+'.engine.npy')
                np.save(tensor_path,native,allow_pickle=False)
                row['engine_tensor_path']=tensor_path.name; row['engine_tensor_sha256']=sha(tensor_path)
                assert native.dtype==np.float32 and native.ndim==3 and native.shape[:2]==(1,1)
                native=native[0,0,:]
                raw=OUT/(request['id']+'.npy'); np.save(raw,native,allow_pickle=False)
                row.update(raw_path=raw.name,raw_sha256=sha(raw),native_samples=len(native))
                assert np.isfinite(native).all()
                divisor=math.gcd(model['native_rate'],16000)
                pcm=resample_poly(native,16000//divisor,model['native_rate']//divisor)
                wav=OUT/(request['id']+'.wav'); sf.write(wav,pcm,16000,subtype='PCM_16')
                row.update(path=wav.name,sha256=sha(wav),samples=len(pcm),generated_reference_end_sample=len(pcm))
                row['audit']=audit_output(OUT,row)
            except Exception as exc:
                row['error']=f'{type(exc).__name__}: {exc}'
            rows.append(row); log.write(json.dumps(row,ensure_ascii=False)+'\n'); log.flush()
            print(json.dumps(dict(id=request['id'],passed=row.get('audit',{}).get('passed',False),error=row.get('error'))),flush=True)
    report=dict(plan_sha256=sha(PLAN),versions=versions(),provider=session.get_providers(),
                requested=len(requests),recorded=len(rows),
                passed=sum(r.get('audit',{}).get('passed',False) for r in rows),
                pack_qualified=len(rows)==len(requests) and all(r.get('audit',{}).get('passed',False) for r in rows),
                records_sha256=sha(OUT/'records.jsonl'),mike_inference=False,wake_model_scoring=False,
                limits='Native mechanical gate only; not phonetic truth, rendered-pack qualification or source-family independence.')
    (OUT/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)


if __name__ == '__main__':
    if sys.argv[1:]==['--write-plan']:
        make_plan()
    elif sys.argv[1:]==['--generate-development']:
        generate()
    else:
        raise SystemExit('Use --write-plan or --generate-development')
