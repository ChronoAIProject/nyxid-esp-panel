"""Locked training-only synthesis; no detector, development or acceptance calls."""
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import subprocess
import sys

import numpy as np
import onnxruntime as ort
from piper_phonemize import phonemize_espeak
from scipy.signal import resample_poly
import soundfile as sf

from synthetic_output_quality_v1 import audit_output

R = Path('research/wakeword_p4/results')
PLAN = R / '4090-multispeaker-coverage-native-plan-v1.json'
OUT = Path('.local/wakeword/multispeaker-coverage-native-v1')
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


def write_plan():
    source = R / '4090-multispeaker-training-source-review-v1.json'
    admission = R / '4090-multispeaker-training-role-admission-v1.json'
    history = R / '4090-multispeaker-source-history-v1.json'
    assert json.loads(admission.read_text())['history_report_sha256'] == sha(history)
    assert json.loads(admission.read_text())['known_protected_model_use_found'] is False
    models = json.loads(source.read_text())['models']
    requests = []
    narrow = {}
    for model in models:
        assert sha(model['model_path']) == model['sha256']
        assert sha(model['config_path']) == model['config_sha256']
        config = json.loads(Path(model['config_path']).read_text())
        voices = model['speaker_id_map']
        assert len(voices) == model['num_speakers'] and 36 % len(voices) == 0
        ordered = sorted(voices, key=lambda name: hashlib.sha256(
            ('nyx-multispeaker-coverage-v1:' + model['voice'] + ':' + name).encode()).hexdigest())
        narrow[model['voice']] = ordered[:3]
        for name in sorted(voices):
            for ti, (target, text) in enumerate(TEXTS):
                sentences = phonemize_espeak(text, config['espeak']['voice'])
                assert len(sentences) == 1
                phones = sentences[0]
                mapping = config['phoneme_id_map']
                tokens = list(mapping['^']) + list(mapping['_'])
                for phone in phones:
                    tokens += list(mapping[phone]) + list(mapping['_'])
                tokens += list(mapping['$'])
                for speed in [.9, 1., 1.1]:
                    broad_draws = 36 // len(voices)
                    draws = 12 if name in ordered[:3] else broad_draws
                    for draw in range(draws):
                        arms = (['narrow'] if name in ordered[:3] else [])
                        if draw < broad_draws:
                            arms += ['broad']
                        requests.append(dict(
                            id=f"{model['voice']}-s{voices[name]:02d}-t{ti:02d}-l{int(speed*10)}-d{draw:02d}",
                            voice=model['voice'], speaker_name=name, speaker_id=voices[name],
                            role='training', arms=arms, target=target,
                            label=int(target == 'hey_nyx'), text=text,
                            phonemes=phones, token_ids=tokens,
                            length_scale=speed, draw=draw))
    counts = {arm: sum(arm in r['arms'] for r in requests) for arm in ['narrow', 'broad']}
    assert counts == dict(narrow=4320, broad=4320)
    assert len(requests) == 7740
    assert len({r['id'] for r in requests}) == len(requests)
    bound = [source, admission, history, Path(__file__),
             Path(__file__).with_name('synthetic_output_quality_v1.py'),
             Path(__file__).with_name('synthetic_output_quality_protocol_v1.md'),
             Path(__file__).with_name('multispeaker_coverage_hypothesis_v1.md'),
             R / '4090-mike-acceptance-generation-reservation-v1.json',
             R / '4090-jenny-development-native-generation-v1.json']
    plan = dict(schema='multispeaker-coverage-native-plan-v1', models=models,
                requests=requests, arm_requests=counts, narrow_speakers=narrow,
                versions=versions(), python_version=sys.version,
                files={str(p): sha(p) for p in bound},
                provider='CPUExecutionProvider', threads=2,
                noise_scale=.667, noise_w=.8, random_seed=None,
                duration_interface='Unmodified ONNX waveform output only; no exposed token durations',
                native_return_policy='Save whole float32 tensor before any shape or quality check; require rank4 [1,1,1,N], preserve every sample in mono vector',
                trim=False, failure_policy='retain-all-stop-pack-no-substitution',
                role_registry=dict(training=[m['voice'] for m in models],
                                   development='Author-direct Jenny reserved; Joe consumed diagnosis only',
                                   acceptance='Mike and HFC male protected; no inference under this lock'),
                maximum_native_storage_gib=9.,
                scope='Native training source preparation only; no render/feature/training/detector/development/acceptance authorization. Unknown physical/pretraining overlap remains a limit, not human listening prerequisite.')
    with PLAN.open('x') as stream:
        stream.write(json.dumps(plan, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps(dict(requests=len(requests), arm_requests=counts, narrow_speakers=narrow, plan_sha256=sha(PLAN))))


def generate():
    # Authenticate all prerequisites before creating output or initializing TTS.
    plan = json.loads(PLAN.read_text())
    assert versions() == plan['versions'] and sys.version == plan['python_version']
    for path, digest in plan['files'].items():
        assert sha(path) == digest, path
    subprocess.run(['git', 'diff', '--exit-code', '--', str(PLAN), *plan['files']], check=True)
    subprocess.run(['/usr/bin/python3', 'research/wakeword_p4/tools/check_disk_budget.py',
                    '--expected-gib', str(plan['maximum_native_storage_gib'])], check=True)
    for model in plan['models']:
        assert sha(model['model_path']) == model['sha256']
        assert sha(model['config_path']) == model['config_sha256']
    OUT.mkdir(exist_ok=False)
    (OUT / 'plan.json').write_bytes(PLAN.read_bytes())
    (OUT / 'generator.py').write_bytes(Path(__file__).read_bytes())
    sessions = {}
    for model in plan['models']:
        options = ort.SessionOptions()
        options.intra_op_num_threads = plan['threads']
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        session = ort.InferenceSession(model['model_path'], sess_options=options,
                                       providers=[plan['provider']])
        assert session.get_providers() == [plan['provider']]
        assert {i.name for i in session.get_inputs()} == {'input', 'input_lengths', 'scales', 'sid'}
        sessions[model['voice']] = session
    rows = []
    models = {m['voice']: m for m in plan['models']}
    with (OUT / 'records.jsonl').open('x') as log:
        for index, request in enumerate(plan['requests']):
            model = models[request['voice']]
            config = json.loads(Path(model['config_path']).read_text())
            row = dict(request, model_sha256=model['sha256'], config_sha256=model['config_sha256'],
                       native_rate=config['audio']['sample_rate'], trimmed=False,
                       random_seed=None, phoneme_durations=None,
                       duration_interface=plan['duration_interface'])
            try:
                tokens = np.asarray([request['token_ids']], dtype=np.int64)
                outputs = sessions[request['voice']].run(None, dict(
                    input=tokens, input_lengths=np.asarray([tokens.shape[1]], dtype=np.int64),
                    scales=np.asarray([plan['noise_scale'], request['length_scale'], plan['noise_w']], dtype=np.float32),
                    sid=np.asarray([request['speaker_id']], dtype=np.int64)))
                assert len(outputs) == 1
                tensor = outputs[0]
                tensor_path = OUT / (request['id'] + '.engine.npy')
                np.save(tensor_path, tensor, allow_pickle=False)
                row.update(engine_tensor_path=tensor_path.name, engine_tensor_sha256=sha(tensor_path),
                           engine_output_shape=list(tensor.shape))
                assert tensor.dtype == np.float32 and tensor.ndim == 4 and tensor.shape[:3] == (1, 1, 1)
                native = tensor[0, 0, 0, :]
                raw = OUT / (request['id'] + '.npy')
                np.save(raw, native, allow_pickle=False)
                row.update(raw_path=raw.name, raw_sha256=sha(raw), native_samples=len(native))
                assert np.isfinite(native).all()
                divisor = math.gcd(row['native_rate'], 16000)
                pcm = resample_poly(native, 16000 // divisor, row['native_rate'] // divisor)
                wav = OUT / (request['id'] + '.wav')
                sf.write(wav, pcm, 16000, subtype='PCM_16')
                row.update(path=wav.name, sha256=sha(wav), samples=len(pcm),
                           generated_reference_end_sample=len(pcm))
                row['audit'] = audit_output(OUT, row)
            except Exception as exc:
                row['error'] = f'{type(exc).__name__}: {exc}'
            rows.append(row)
            log.write(json.dumps(row, ensure_ascii=False) + '\n')
            log.flush()
            if (index + 1) % 50 == 0 or index == 0:
                print(json.dumps(dict(recorded=len(rows), requested=len(plan['requests']),
                                      quality_passes=sum(r.get('audit', {}).get('passed', False) for r in rows))), flush=True)
    report = dict(schema='multispeaker-coverage-native-original-v1', plan_sha256=sha(PLAN),
                  versions=versions(), requested=len(plan['requests']), recorded=len(rows),
                  quality_passes=sum(r.get('audit', {}).get('passed', False) for r in rows),
                  failures=[r['id'] for r in rows if not r.get('audit', {}).get('passed', False)],
                  records_path=str(OUT / 'records.jsonl'), records_sha256=sha(OUT / 'records.jsonl'),
                  detector_invocations=0, development_opened=False, acceptance_opened=False,
                  limits='Synthetic output/transport process quality only, not phonetic truth or source independence. Every failure retained; no retries, substitutions or filtering.')
    report['pack_qualified'] = report['recorded'] == report['requested'] and not report['failures']
    (OUT / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    if sys.argv[1:] == ['--write-plan']:
        write_plan()
    elif sys.argv[1:] == ['--generate']:
        generate()
    else:
        raise SystemExit('Use --write-plan or --generate')
