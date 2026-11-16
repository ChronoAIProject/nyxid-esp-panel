"""Training-only conditioned synthesis; all96 requests once, no scoring-driven retries."""
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time

SOURCE = Path('.local/wakeword/chatterbox-source-v1')
PLAN = Path('research/wakeword_p4/results/4090-chatterbox-conditioned-training-plan-v1.json')
LOCK = PLAN.with_name('4090-chatterbox-conditioned-training-lock-v1.json')
OUT = Path('.local/wakeword/chatterbox-conditioned-training-v1')


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def bindings():
    paths = [PLAN, Path(__file__), SOURCE / 'result.json',
             SOURCE / 'environment-result.json', SOURCE / 'environment-freeze.txt']
    paths += [Path(row['reference']) for row in json.loads(PLAN.read_text())['requests']]
    return {str(path): sha(path) for path in paths}


def plan():
    references = Path('research/wakeword_p4/results/4090-chatterbox-training-reference-result-v1.json')
    rows = json.loads(references.read_text())['records']
    assert len(rows) == 4 and all(row['split'] == 'training' for row in rows)
    texts = [('Hey Nicks.', 1), ('Hey Nicks, turn on the light.', 1),
             ('Hey Nicks, what time is it?', 1), ('Hey Nicks, set a timer.', 1),
             ("What's next?", 0), ('Please mix the ingredients.', 0),
             ('Turn on the lights.', 0), ('Set a timer.', 0)]
    requests = []
    for reference in rows:
        assert sha(Path(reference['path'])) == reference['sha256']
        for text, label in texts:
            for seed in [726, 727, 728]:
                requests.append(dict(role='training-only', voice=reference['voice'],
                    reference=reference['path'], reference_sha256=reference['sha256'],
                    text=text, requested_label=label, seed=seed, temperature=.8,
                    top_p=.95, top_k=1000, repetition_penalty=1.2))
    with PLAN.open('x') as stream:
        stream.write(json.dumps(dict(requests=requests, references_sha256=sha(references),
            policy='Training-only synthetic conditioning, no builtin development voice. All96 retained, requested labels not phonetic truth. No wake model scoring or acceptance. Native watermark retained. Command prefix timing requires separate verification.'), indent=2) + '\n')


def lock():
    record = dict(schema='chatterbox-conditioned-training-lock-v1', bindings=bindings(),
                  output=str(OUT), requests=96, role='training-only',
                  native='watermarked float32 numpy, full waveform',
                  replay='16kHz mono PCM16, scipy resample_poly, no trimming',
                  selection='All requests once in locked order; retain failures and partial outputs. No pronunciation correctness claimed by mechanical checks.')
    with LOCK.open('x') as stream:
        stream.write(json.dumps(record, indent=2) + '\n')


def run():
    record = json.loads(LOCK.read_text())
    assert record['bindings'] == bindings()
    assert subprocess.check_output(['git', 'show',
        'origin/research-hye-nyx-voice-gpu:' + str(LOCK)]) == LOCK.read_bytes()
    assert Path(sys.prefix).resolve() == Path('.local/wakeword/chatterbox-env-v1').resolve()
    assert shutil.disk_usage(SOURCE).free > 100 * 1024**3
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['OMP_NUM_THREADS'] = '2'
    os.environ['MKL_NUM_THREADS'] = '2'
    import math
    import numpy as np
    import soundfile as sf
    import torch
    from scipy.signal import resample_poly
    from chatterbox.tts_turbo import ChatterboxTurboTTS
    torch.set_num_threads(2)
    assert torch.cuda.is_available() and '4090' in torch.cuda.get_device_name(0)
    requests = json.loads(PLAN.read_text())['requests']
    assert len(requests) == 96
    OUT.mkdir(exist_ok=False)
    model = ChatterboxTurboTTS.from_local(SOURCE / 'model', device='cuda')
    last_reference = None
    for index, request in enumerate(requests):
        if request['reference'] != last_reference:
            model.prepare_conditionals(request['reference'])
            last_reference = request['reference']
        seed = request['seed']
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        start = time.monotonic()
        with torch.inference_mode():
            output = model.generate(request['text'], temperature=request['temperature'],
                top_p=request['top_p'], top_k=request['top_k'],
                repetition_penalty=request['repetition_penalty'])
        native = output.detach().cpu().numpy().astype(np.float32)
        name = f'{index:03d}'
        native_path = OUT / (name + '-native.npy')
        np.save(native_path, native, allow_pickle=False)
        assert native.ndim == 2 and native.shape[0] == 1
        assert native.size > 0 and np.isfinite(native).all()
        rate = int(model.sr)
        divisor = math.gcd(rate, 16000)
        resampled = resample_poly(native[0], 16000 // divisor, rate // divisor)
        pcm = np.rint(np.clip(resampled, -1, 32767 / 32768) * 32768).astype(np.int16)
        wav_path = OUT / (name + '-16k.wav')
        sf.write(wav_path, pcm, 16000, subtype='PCM_16')
        row = dict(index=index, request=request, native_rate=rate,
                   native_shape=list(native.shape), seconds=native.shape[-1]/rate,
                   peak=float(np.max(np.abs(native))),
                   native_outside_pcm_range=int(np.count_nonzero((native < -1) | (native > 32767/32768))),
                   resampled_outside_pcm_range=int(np.count_nonzero((resampled < -1) | (resampled > 32767/32768))),
                   elapsed_seconds=time.monotonic()-start,
                   files=[dict(path=str(p), bytes=p.stat().st_size, sha256=sha(p))
                          for p in [native_path, wav_path]],
                   phoneme_timing=None, mechanical_checks_only=True)
        (OUT / (name + '.json')).write_text(json.dumps(row, indent=2) + '\n')
        print(json.dumps(row), flush=True)
    (OUT / 'completed.json').write_text(json.dumps(dict(requests=96,
        lock_sha256=sha(LOCK), pronunciation_validated=False), indent=2) + '\n')


if __name__ == '__main__':
    {'plan': plan, 'lock': lock, 'run': run}[sys.argv[1]]()
