"""Lock then generate the 96 daily-language requests; no scoring-driven retries."""
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
PLAN = Path('research/wakeword_p4/results/4090-chatterbox-daily-language-plan-v1.json')
LOCK = PLAN.with_name('4090-chatterbox-daily-generation-lock-v1.json')
OUT = Path('.local/wakeword/chatterbox-daily-language-v1')


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def bindings():
    paths = [PLAN, Path(__file__), SOURCE / 'result.json',
             SOURCE / 'environment-result.json', SOURCE / 'environment-freeze.txt']
    return {str(path): sha(path) for path in paths}


def plan():
    categories = {
        'conversation': ["Good morning, how are you?", "Could you help me with this?", "I will be home in ten minutes.", "Thank you, that sounds good.", "Where did you put the keys?", "Please close the window.", "The weather looks nice today.", "Let us talk about it later."],
        'household': ["Turn on the light.", "Turn off the bedroom lights.", "Set a timer for five minutes.", "What time is it?", "Play some music.", "Turn the volume down.", "Open the curtains.", "Remind me to call tomorrow."],
        'next_mix': ["What is next on the list?", "The next train arrives soon.", "Please go to the next page.", "I will see you next week.", "Mix the ingredients together.", "Do not mix those colors.", "This is a nice mix of songs.", "We need six cups of water."],
        'name_address': ["Hey Nick, can you help me?", "Hey Nick, what time is it?", "Nick, please turn on the light.", "Nicky, are you coming?", "Hey Alex, are you there?", "Hey Max, come over here.", "Hey Mike, good morning.", "Hey Chris, how was your day?"]}
    requests=[]
    for category,texts in categories.items():
        for text in texts:
            for seed in [726,727,728]:
                requests.append(dict(role='development-only',voice='turbo-builtin-default',
                    group=category,text=text,seed=seed,temperature=.8,top_p=.95,
                    top_k=1000,repetition_penalty=1.2))
    assert len(requests)==96
    with PLAN.open('x') as stream:
        stream.write(json.dumps(dict(requests=requests,
            policy='Common-language synthetic diagnostic expansion, no training or threshold changes.32sentences x3seeds from one consumed builtin conditioning voice. Name addresses are potential real lexical conflicts, not automatic zero-required negatives. Whole-waveform transcription/intent not phonetic truth or real occurrence weighting. No final acceptance accessed.'),indent=2)+'\n')


def lock():
    record = dict(schema='chatterbox-generation-lock-v1', bindings=bindings(),
                  output=str(OUT), requests=96, role='development-only',
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
    for index, request in enumerate(requests):
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
