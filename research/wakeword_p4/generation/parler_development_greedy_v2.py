"""Pinned development-only synthesis. Prepare and push lock before run; no retries."""
import os
os.environ['USE_TF'] = '0'
os.environ['USE_FLAX'] = '0'
import argparse
import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'research/wakeword_p4/experiments'))
MODEL = ROOT / '.local/wakeword/parler-mini-pinned-v1'
PLAN = ROOT / 'research/wakeword_p4/results/4090-parler-greedy-request-plan-v2.json'
LOCK = ROOT / 'research/wakeword_p4/results/4090-parler-greedy-execution-lock-v2.json'
OUT = ROOT / '.local/wakeword/parler-development-greedy-v2'

def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()

def identity():
    import torch
    assert torch.cuda.is_available() and '4090' in torch.cuda.get_device_name(0)
    assert (MODEL / 'model.safetensors').is_file(), 'Pinned download incomplete'
    return dict(plan_sha256=sha(PLAN), generator_sha256=sha(__file__),
                files={p.name: dict(bytes=p.stat().st_size, sha256=sha(p))
                       for p in sorted(MODEL.iterdir()) if p.is_file()},
                packages={n: metadata.version(n) for n in
                          ['torch', 'transformers', 'parler-tts', 'descript-audio-codec',
                           'numpy', 'scipy', 'soundfile', 'sentencepiece', 'huggingface-hub']},
                device=torch.cuda.get_device_name(0), dtype='float32',
                generation=dict(json.loads((MODEL / 'generation_config.json').read_text()), do_sample=False),
                source_commit=subprocess.check_output(['git', '-C', str(ROOT / '.local/wakeword/parler-source-v1'), 'rev-parse', 'HEAD'], text=True).strip(),
                role='development', requests=144,
                policy='retain-all-stop-pack-no-substitution', training_prohibited=True,
                stopping='Capture unmodified _sample token sequences before audio decoding; reject length >=2580 or missing EOS in any codebook row.',
                quality='synthetic_output_quality_v1.audit_output; unchanged 0.2..5 seconds',
                quality_code_sha256=sha(ROOT / 'research/wakeword_p4/experiments/synthetic_output_quality_v1.py'),
                overlap='Physical-speaker/pretraining overlap unknown. Shared TTS parent across roles is an explicit exception to old role auditor; no claim it passed.',
                metadata='No authenticated generated phonemes/durations; null. Input tokens are text tokens, not phonetic truth.',
                scope='Clean development native source only. Noise/RIR/background protocol must be locked separately before wake scoring.')

def main():
    a = argparse.ArgumentParser()
    a.add_argument('action', choices=['prepare', 'run'])
    args = a.parse_args()
    ident = identity()
    if args.action == 'prepare':
        assert not LOCK.exists(), 'Never overwrite an execution lock'
        LOCK.write_text(json.dumps(ident, indent=2) + '\n')
        print(LOCK, flush=True)
        return
    assert json.loads(LOCK.read_text()) == ident, 'Execution identity changed'
    subprocess.run(['git', '-C', str(ROOT), 'diff', '--exit-code', 'HEAD', '--', str(LOCK), __file__], check=True)
    # Require exact lock bytes to exist in the pushed branch before synthesis.
    remote = subprocess.check_output(['git', '-C', str(ROOT), 'show', 'origin/research-hye-nyx-voice-gpu:' + str(LOCK.relative_to(ROOT))])
    assert remote == LOCK.read_bytes(), 'Push execution lock first'
    assert not OUT.exists(), 'No overwrite or implicit resume/retry'
    OUT.mkdir(parents=True)
    (OUT / 'execution-lock.json').write_bytes(LOCK.read_bytes())
    import numpy as np
    import soundfile as sf
    import torch
    from scipy.signal import resample_poly
    from transformers import AutoTokenizer, set_seed
    from parler_tts import ParlerTTSForConditionalGeneration
    from synthetic_output_quality_v1 import audit_output
    requests = [r for r in json.loads(PLAN.read_text())['requests'] if r['role'] == 'development']
    assert len(requests) == 144
    model = ParlerTTSForConditionalGeneration.from_pretrained(str(MODEL), local_files_only=True, torch_dtype=torch.float32).to('cuda:0').eval()
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL), local_files_only=True)
    assert model.config.sampling_rate == 44100
    captured = {}
    original = model._sample
    def capture(*args, **kwargs):
        result = original(*args, **kwargs)
        captured['tokens'] = result.sequences.detach().cpu().clone()
        return result
    model._sample = capture
    ledger = OUT / 'records.jsonl'
    try:
        for r in requests:
            captured.clear()
            set_seed(r['seed'])
            desc = tokenizer(r['description'], return_tensors='pt').to('cuda:0')
            prompt = tokenizer(r['text'], return_tensors='pt').to('cuda:0')
            with torch.inference_mode():
                result = model.generate(input_ids=desc.input_ids, attention_mask=desc.attention_mask,
                                        prompt_input_ids=prompt.input_ids, prompt_attention_mask=prompt.attention_mask,
                                        return_dict_in_generate=True, do_sample=False)
            wave = result.sequences.detach().cpu().numpy()
            name = r['request_id']
            # Save exact engine return (shape retained) before any transport operation.
            np.save(OUT / (name + '.engine.npy'), wave, allow_pickle=False)
            assert wave.ndim == 2 and wave.shape[0] == 1 and wave.dtype == np.float32
            native = wave[0]
            raw = OUT / (name + '.native.npy')
            np.save(raw, native, allow_pickle=False)
            tokens = captured['tokens'].numpy()
            token_path = OUT / (name + '.tokens.npy')
            np.save(token_path, tokens, allow_pickle=False)
            pcm = resample_poly(native, 160, 441)
            path = OUT / (name + '.wav')
            sf.write(path, pcm, 16000, subtype='PCM_16')
            record = dict(r, raw_path=raw.name, raw_sha256=sha(raw), path=path.name, sha256=sha(path),
                          engine_path=name + '.engine.npy', engine_sha256=sha(OUT / (name + '.engine.npy')),
                          native_samples=len(native), samples=len(pcm), generated_reference_end_sample=len(pcm),
                          generated_phonemes=None, generated_phone_durations=None,
                          engine_shape=list(wave.shape), engine_audio_lengths=list(result.audios_length),
                          tokens_path=token_path.name, tokens_sha256=sha(token_path), token_shape=list(tokens.shape),
                          description_input_ids=desc.input_ids.cpu().tolist(), prompt_input_ids=prompt.input_ids.cpu().tolist(),
                          reached_token_limit=bool(tokens.shape[-1] >= 2580),
                          eos_each_row=np.any(tokens == 1024, axis=-1).tolist())
            record['quality'] = audit_output(OUT, record)
            record['passed'] = bool(record['quality']['passed'] and not record['reached_token_limit'] and all(record['eos_each_row']))
            with ledger.open('a') as f:
                f.write(json.dumps(record) + '\n')
                f.flush()
                os.fsync(f.fileno())
            print(name, record['passed'], len(pcm) / 16000, flush=True)
            if not record['passed']:
                raise RuntimeError('Retained failed output; stop pack without substitutions: ' + name)
        (OUT / 'completed.json').write_text(json.dumps(dict(requests=len(requests), passed=True)) + '\n')
    except BaseException as e:
        (OUT / 'failure.json').write_text(json.dumps(dict(type=type(e).__name__, error=str(e))) + '\n')
        raise

if __name__ == '__main__':
    main()
