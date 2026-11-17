"""Lock then generate the 48 daily-language requests; no scoring-driven retries."""
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
PLAN = Path('research/wakeword_p4/results/4090-parallel-prospective-language-plan-v1.json')
LOCK = PLAN.with_name('4090-parallel-prospective-generation-lock-v1.json')
OUT = Path('.local/wakeword/parallel-prospective-language-v1')


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def bindings():
    paths = [PLAN, Path(__file__), SOURCE / 'result.json',
             SOURCE / 'environment-result.json', SOURCE / 'environment-freeze.txt']
    paths += [Path(m['path']) for m in json.loads(PLAN.read_text())['models']]
    return {str(path): sha(path) for path in paths}


def plan():
    categories = {
      'nyx_connected_command': [
        'Hey Nicks, could you switch off the study lamp?',
        'Hey Nicks, please remind me to take the bread out in twelve minutes.',
        'Hey Nicks, is there any rain expected this afternoon?',
        'Hey Nicks, lower the music so I can answer the phone.',
        'Hey Nicks, I would like to hear the news while I make breakfast.',
        'Hey Nicks, put milk and apples on my shopping list.',
        'Hey Nicks, leave the hallway light on until I get upstairs.',
        'Hey Nicks, tell me how long the journey to the airport will take.',
        'Hey Nicks, pause the television for a moment please.',
        'Hey Nicks, make the room a little warmer before bedtime.',
        'Hey Nicks, cancel the reminder about the laundry.',
        'Hey Nicks, do I have any appointments after lunch?'],
      'ordinary_conversation': [
        'I thought the parcel was arriving on Thursday, but it came this morning.',
        'Could you move your bag so that I can sit beside the window?',
        'We have already eaten, so please do not wait for us.',
        'She said the shop closes early on Sundays.',
        'There is a little more soup in the pot if you are still hungry.',
        'I cannot find my glasses, although I was wearing them a moment ago.',
        'The children would rather stay indoors until the rain stops.',
        'Let me finish this message before we leave for the station.'],
      'ordinary_common_words': [
        'After the next announcement, we should know which platform to use.',
        'I was hoping to visit next month, if the trains are running.',
        'You need to mix it slowly or the sauce will become lumpy.',
        'This playlist has a mixture of old songs and newer recordings.',
        'There are six people waiting outside and another two on the stairs.',
        'The price was sixty pounds, including delivery.',
        'I found a nice little cafe near the museum yesterday.',
        'That fixes the problem with the kitchen cupboard.'],
      'ordinary_household': [
        'Would you leave a small light on in the kitchen tonight?',
        'The timer should ring just before the potatoes are ready.',
        'I turned the bedroom speaker down because the baby was asleep.',
        'Please check whether the back door is locked before you go upstairs.',
        'The washing machine has finished, but the clothes are still wet.',
        'We can open the windows once the traffic has settled down.',
        'I will call you when the kettle starts boiling.',
        'Please put the remote control back beside the sofa.'],
      'ordinary_name_address': [
        'Hey Nick, could you switch off the study lamp?',
        'Hey Nick, please remind me to take the bread out in twelve minutes.',
        'Hey Nick, is there any rain expected this afternoon?',
        'Hey Nick, lower the music so I can answer the phone.',
        'Nick, did you remember to bring the charger from the car?',
        'I spoke to Nick yesterday and he is feeling much better.',
        'Hey Nicky, your breakfast is getting cold.',
        'Hey Alex, would you mind answering the door?',
        'Hey Mike, I think you left your coat in the hallway.',
        'Hey Chris, the taxi will be here in a couple of minutes.',
        'Hey Max, can you carry the other bag upstairs?',
        'Hey Nicole, are we still meeting outside the cinema?']}
    requests=[]
    old_paths=[Path('research/wakeword_p4/results')/f for f in ['4090-chatterbox-daily-language-plan-v1.json','4090-chatterbox-daily-training-plan-v1.json']]
    old_texts={r['text'] for p in old_paths for r in json.loads(p.read_text())['requests']}
    for category,texts in categories.items():
        for text in texts:
            assert text not in old_texts
            requests.append(dict(role='prospective-targeted-development-not-acceptance',voice='turbo-builtin-default-already-consumed',group=category,text=text,seed=731,temperature=.8,top_p=.95,top_k=1000,repetition_penalty=1.2))
    assert len(requests)==48 and len({r['text'] for r in requests})==48
    models=[dict(name='c161',path='.local/wakeword/causal-source-balanced-int8-v1/int8.tflite',bytes=60952,sha256='c16154040378b8acccdb57c19917c46cea0a9ea6a39f105004b964ddd093ab81'),dict(name='binary40',path='.local/wakeword/binary40-trained-int8-v1/int8.tflite',bytes=64800,sha256='454058039dd41e729ce82ac7f33a84edec4af9554615a632e3b33d47de38009c')]
    for m in models:assert Path(m['path']).stat().st_size==m['bytes'] and sha(Path(m['path']))==m['sha256']
    record=dict(schema='parallel-prospective-language-plan-v1',requests=requests,models=models,exact_text_disjointness_checked_against={str(p):sha(p) for p in old_paths},operating_profile=dict(threshold=.91796875,smoothing_steps=5,refractory_steps=67,prime_samples=81600,native_samples=480,sample_rate=16000),policy='Targeted new-context development after consumed Nick observations, not independent voice/general acceptance. One existing builtin synthetic voice,48new literal requests once each at fixed seed731; preserve all failures/outputs, no score-based retry. Twelve intended Nyx connected commands,24ordinary sentences,12name addresses. Names are collision diagnostics, not zero-error gate. Intended text is not phonetic truth. No training or threshold/policy changes from this data; fixed C161/originalB40/parallel reported jointly. No protected acceptance material accessed. No new continuous background exposure/FAR claim. Whole native waveforms and complete16kPCM retained. Final replay must cover all48without result selection and preserve both raw scores/native clocks/own and emitted/suppressed events. Original priming and frontend no-reset rules retained.')
    with PLAN.open('x') as stream:stream.write(json.dumps(record,indent=2)+'\n')


def lock():
    record = dict(schema='chatterbox-generation-lock-v1', bindings=bindings(),
                  output=str(OUT), requests=48, role='prospective-targeted-development-not-acceptance',
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
    assert len(requests) == 48
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
    (OUT / 'completed.json').write_text(json.dumps(dict(requests=48,
        lock_sha256=sha(LOCK), pronunciation_validated=False), indent=2) + '\n')


if __name__ == '__main__':
    {'plan': plan, 'lock': lock, 'run': run}[sys.argv[1]]()
