"""Prepare training-only paired windows and offline phoneme posterior targets."""
import argparse
from collections import Counter
import hashlib
from importlib.metadata import version
import json
from pathlib import Path

import numpy as np
import soundfile as sf

from near_word_windows import load_windows
from paired_training import bank as paired_bank
from probe_causal_phone_verifier import sha
from stream_data import features, render_stream

FRAMES, STEPS, VOCAB = 335, 67, 392


def align_posteriors(probabilities):
    """Average CTC frames ending within each trailing 30 ms encoder step."""
    values = np.asarray(probabilities, dtype=np.float32)
    if values.ndim != 2 or len(values) != 168 or values.shape[1] != VOCAB:
        raise ValueError('Expected the pinned 337-hop PCM window and phone vocabulary')
    ends = 400+320*np.arange(len(values))
    target_ends = (139+3*np.arange(STEPS))*160
    output = []
    for end in target_ends:
        available = (ends > end-480) & (ends <= end)
        if not available.any():
            raise ValueError('Missing teacher frame for an encoder endpoint')
        output.append(values[available].mean(axis=0))
    return np.asarray(output, dtype=np.float32)


def selected_pairs(rows, bounds, count):
    result = []
    for stratum, entries in paired_bank(rows, bounds, FRAMES).items():
        seen = set()
        def rank(entry):
            row = entry[0][0]
            return hashlib.sha256(f"{row['sha256']}:{row['variant']}".encode()).digest()
        for (negative, nband), partners in sorted(entries, key=rank):
            if negative['sha256'] in seen:
                continue
            seen.add(negative['sha256'])
            positive, pband = min(partners, key=lambda entry: hashlib.sha256(
                f"{negative['sha256']}:{entry[0]['sha256']}".encode()).digest())
            result.append((stratum, ((positive, sum(pband)//2), (negative, sum(nband)//2))))
            if len(seen) >= count:
                break
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('features', 'windows', 'backgrounds', 'phone-model', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--pairs-per-stratum', type=int, default=8)
    p.add_argument('--retain-teacher-frames', action='store_true',
                   help='Also retain the original 168-step teacher posteriors before alignment/averaging')
    a = p.parse_args()
    if a.output.exists() or a.pairs_per_stratum != 8:
        p.error('Use a fresh directory and the fixed eight-pair source screen')
    previous = Path(__file__).parent/'plans'
    parent = json.loads((previous/'microwake-phrase-output-v1.json').read_bytes())
    phone = json.loads((previous/'microwake-near-word-errors-v1.json').read_bytes())
    if (sha(a.features/'features.json') != parent['feature_index_sha256']['features.json']
            or sha(a.windows) != parent['windows_sha256']
            or sha(a.backgrounds/'manifest.json') != phone['background_manifest_sha256']
            or any(sha(a.phone_model/name) != value for name, value in phone['phoneme_model_files'].items())
            or any(version(name) != value for name, value in phone['runtime_packages'].items())):
        raise ValueError('Use the pinned feature, pairing, renderer and teacher inputs')
    index_bytes = (a.features/'features.json').read_bytes()
    index = json.loads(index_bytes)
    bounds, _ = load_windows(a.windows, index_bytes, index, FRAMES)
    pairs = selected_pairs(index, bounds, a.pairs_per_stratum)
    dataset = a.features/'dataset'
    reserved = {r['sha256'] for r in index if r['split'] != 'training'}
    bg = [str(a.backgrounds/r['path']) for r in json.loads((a.backgrounds/'manifest.json').read_bytes())['records']
          if r['split'] == 'training']
    generation = json.loads((dataset/'generation.json').read_bytes())
    raw = np.memmap(a.features/'features.bin', dtype='<u2', mode='r').reshape(-1, 40)

    import torch
    from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForCTC
    torch.set_num_threads(3)
    torch.set_num_interop_threads(1)
    extractor = Wav2Vec2FeatureExtractor.from_pretrained(a.phone_model, local_files_only=True)
    teacher = Wav2Vec2ForCTC.from_pretrained(a.phone_model, local_files_only=True, weights_only=True).eval()
    if teacher.config.vocab_size != VOCAB or teacher.config.pad_token_id != 0:
        raise ValueError('Changed teacher vocabulary/blank convention')
    a.output.mkdir(parents=True)
    samples, targets, teacher_frames, rows = [], [], [], []
    for number, (stratum, items) in enumerate(pairs, 1):
        for role, (row, end) in enumerate(items):
            path = Path(row['path'])
            if (row['split'] != 'training' or row['sha256'] in reserved or row['label'] != 1-role
                    or path.is_absolute() or '..' in path.parts or sha(dataset/path) != row['sha256']
                    or not FRAMES <= end <= row['frames']):
                raise ValueError('Invalid source or training crop')
            pcm, rate = sf.read(dataset/path, dtype='int16')
            if rate != 16000 or pcm.shape != (row['samples'],):
                raise ValueError('Unexpected source PCM')
            seed = int(hashlib.sha256(row['path'].encode()).hexdigest()[:8], 16)+row['variant']
            wave, lead = render_stream(row, dataset, np.random.default_rng(seed), row['variant'], bg, generation['lead_seconds'])
            cached = raw[row['offset']:row['offset']+row['frames']]
            if lead//160 != row['lead_frame'] or not np.array_equal(features(wave), cached):
                raise ValueError('Rendered training input does not match the original feature cache')
            # Feature u consumes PCM [u*160, (u+3)*160). Retain the full
            # 335-feature support, not a resampled/cropped keyword recording.
            lo, hi = (end-FRAMES)*160, (end+2)*160
            clip = np.rint(wave[lo:hi]).astype(np.int16)
            if len(clip) != (FRAMES+2)*160:
                raise ValueError('Incomplete teacher window')
            with torch.inference_mode():
                inputs = extractor(clip.astype(np.float32)/32768, sampling_rate=16000, return_tensors='pt')
                logits = teacher(**inputs).logits[0]
                original = logits.softmax(-1).numpy()
                posterior = align_posteriors(original)
            if not np.isfinite(posterior).all() or not np.allclose(posterior.sum(-1), 1, atol=1e-5):
                raise ValueError('Invalid teacher posterior')
            sample = np.array(cached[end-FRAMES:end], copy=True)
            samples.append(sample)
            targets.append(posterior)
            if a.retain_teacher_frames:
                teacher_frames.append(original)
            rows.append({'stratum': list(stratum), 'pair': number-1, 'label': row['label'],
                         'path': row['path'], 'source_sha256': row['sha256'], 'variant': row['variant'],
                         'end_frame': end, 'feature_sha256': hashlib.sha256(sample.tobytes()).hexdigest(),
                         'teacher_input_sha256': hashlib.sha256(clip.astype('<i2').tobytes()).hexdigest()})
        if number % 10 == 0 or number == len(pairs):
            print(f'{number}/{len(pairs)} pairs', flush=True)
    data = a.output/'bank.npz'
    arrays = {'features': np.stack(samples), 'posteriors': np.stack(targets)}
    if a.retain_teacher_frames:
        arrays['teacher_posteriors'] = np.stack(teacher_frames)
    np.savez_compressed(data, **arrays)
    report = {'scope': 'Training-only offline phone targets; original binary labels retained.',
              'source_sha256': sha(Path(__file__)), 'feature_index_sha256': sha(a.features/'features.json'),
              'windows_sha256': sha(a.windows), 'teacher_files_sha256': phone['phoneme_model_files'],
              'background_manifest_sha256': sha(a.backgrounds/'manifest.json'),
              'pairs_per_stratum_limit': a.pairs_per_stratum, 'pair_count': len(pairs),
              'strata': dict(Counter(':'.join(r['stratum']) for r in rows[::2])),
              'frames': FRAMES, 'encoder_steps': STEPS, 'vocabulary': VOCAB, 'blank': 0,
              'retained_teacher_frames': 168 if a.retain_teacher_frames else 0,
              'teacher_frame_stride_samples': 320, 'teacher_frame_support_samples': 400,
              'data_sha256': sha(data), 'rows': rows,
              'limits': 'Teacher transformer sees the complete training crop. Posterior timestamps are model coordinates, not human phone endpoints. No teacher prediction replaces a source wake label. No nontraining waveform or score was accessed.'}
    (a.output/'bank.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k: v for k, v in report.items() if k not in ('rows','teacher_files_sha256')}, indent=2))


if __name__ == '__main__':
    main()
