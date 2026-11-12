"""Check saved development audio independently of the synthesis implementation.

Never synthesize, load a wake detector, play audio or access a microphone.
"""
import argparse
from collections import Counter
import hashlib
import io
import json
from pathlib import Path

import numpy as np
from scipy.signal import fftconvolve, resample_poly
import soundfile as sf


def digest(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('data', 'models', 'generator', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    manifest_bytes = (args.data / 'manifest.json').read_bytes()
    manifest = json.loads(manifest_bytes)
    plan_bytes = (args.data / 'plan.json').read_bytes()
    plan = json.loads(plan_bytes)
    assert digest(plan_bytes) == manifest['plan_sha256']
    assert digest(args.generator.read_bytes()) == manifest['generator_sha256'] == plan['generator_sha256']
    assert manifest['role'] == plan['role'] == 'synthetic-development-only'
    assert not manifest['wake_models_scored'] and not manifest['training_data']
    assert not plan['wake_model_scoring'] and not plan['training'] and not plan['existing_final_data_access']
    assert plan['length_scale'] == 1.0
    assert plan['draws_per_voice'] == 3
    assert plan['conditions'] == ['quiet', 'noise25', 'combined15']
    assert len(manifest['native_records']) == 42
    assert len(manifest['records']) == 126
    assert manifest['all_mechanical_pass'] and not manifest['mechanical_failures']
    voices = {v['name']: v for v in plan['voices']}
    assert set(voices) == {'kristin', 'norman'}
    phrases = {p['id']: p for p in plan['phrases']}
    assert len(phrases) == 7
    expected = {(v, d, p) for v in voices for d in range(3) for p in phrases}
    assert {(r['voice'], r['draw'], r['id']) for r in manifest['native_records']} == expected
    configs = {}
    for voice, spec in voices.items():
        config_path = args.models / (voice + '.onnx.json')
        assert digest(config_path.read_bytes()) == spec['files'][config_path.name]
        configs[voice] = json.loads(config_path.read_text())

    dry_sources, native, rendered = {}, {}, {}
    groups = {}
    prefix_ms = {}
    for row in manifest['native_records']:
        key = row['key']
        assert key == f"{row['voice']}-{row['draw']}-{row['id']}"
        assert key not in native
        assert row['text'] == phrases[row['id']]['text']
        assert row['target'] == phrases[row['id']]['target']
        assert row['source_rate'] == 22050
        path = args.data / row['native_path']
        assert path.name == key + '-native.npy'
        raw = path.read_bytes()
        assert digest(raw) == row['native_sha256']
        audio = np.load(io.BytesIO(raw), allow_pickle=False)
        assert audio.ndim == 1 and audio.dtype == np.float32
        assert np.isfinite(audio).all() and len(audio) == row['source_samples']
        assert digest(audio.tobytes()) == row['native_pcm_sha256']
        assert .2 <= len(audio) / 22050 <= 10 and np.max(np.abs(audio)) < 2
        phones = row['phonemes']
        assert phones == voices[row['voice']]['expected_phonemes'][row['id']]
        config = configs[row['voice']]
        assert config['audio']['sample_rate'] == 22050
        phone_map = config['phoneme_id_map']
        token_ids, alignment = [], []
        widths = np.asarray(row['phoneme_id_samples'], dtype=np.int64)
        assert len(widths) == len(row['phoneme_ids']) and np.all(widths >= 0)
        assert np.all(widths % 256 == 0) and int(widths.sum()) == len(audio)
        for phone in ['^', *phones, '$']:
            ids = list(phone_map.get(phone, []))
            if phone != '$':
                ids += phone_map.get('_', [])
            start = int(widths[:len(token_ids)].sum())
            token_ids.extend(ids)
            stop = int(widths[:len(token_ids)].sum())
            alignment.append(dict(phone=phone, start_sample=start, end_sample=stop))
        assert token_ids == row['phoneme_ids']
        assert alignment == row['generator_alignment']
        punctuation = next(i for i, p in enumerate(phones) if p in ',.?!')
        # One leading ^ record; the last prefix phone is alignment[punctuation].
        end = alignment[punctuation]['end_sample']
        assert 0 < end <= len(audio) and end == row['generator_prefix_end_sample']
        dry = resample_poly(audio.astype(np.float64), up=320, down=441)
        assert len(dry) == row['resampled_samples']
        assert abs(len(dry) / 16000 - len(audio) / 22050) <= 1 / 16000
        assert np.sqrt(np.mean(dry * dry)) >= 1e-6 and row['mechanical_pass']
        native[key], dry_sources[key] = row, dry
        groups.setdefault((row['voice'], row['draw']), []).append(key)
        prefix_ms.setdefault(row['voice'] + '/' + row['id'], []).append(1000 * end / 22050)
    assert len(native) == 42
    assert {p.name for p in args.data.glob('*-native.npy')} == {r['native_path'] for r in native.values()}
    for row in manifest['records']:
        key = (row['source_key'], row['condition'])
        assert key not in rendered and row['source_key'] in native
        assert row['condition'] in plan['conditions']
        rendered[key] = row
    assert len(rendered) == 126
    assert {p.name for p in args.data.glob('*.wav')} == {r['path'] for r in rendered.values()}

    ir = np.zeros(961, dtype=np.float64)
    for offset, amplitude in ((0, 1.), (160, .25), (480, -.12), (960, .06)):
        ir[offset] = amplitude
    max_pcm_difference, hashes_verified = 0, 0
    for group, keys in groups.items():
        samples = 32000 + max(len(dry_sources[k]) + 960 for k in keys) + 16000
        waves = {}
        for key in keys:
            noise_seed = int(digest(('heldout-bryce-v1:' + key).encode())[:16], 16)
            noise = np.random.default_rng(noise_seed).standard_normal(samples)
            noise /= np.sqrt(np.mean(noise * noise))
            for condition in plan['conditions']:
                row = rendered[(key, condition)]
                source = native[key]
                signal = fftconvolve(dry_sources[key], ir) if condition == 'combined15' else dry_sources[key]
                gain = .08 / np.sqrt(np.mean(signal * signal))
                assert gain == row['level_gain']
                wave = np.zeros(samples, dtype=np.float64)
                wave[32000:32000 + len(signal)] = signal * gain
                if condition != 'quiet':
                    snr = {'noise25': 25, 'combined15': 15}[condition]
                    wave += .08 * 10 ** (-snr / 20) * noise
                waves[(key, condition)] = wave
                assert row['noise_seed'] == noise_seed
                assert row['voice'] == source['voice'] and row['draw'] == source['draw']
                assert row['target'] == source['target'] and row['intended_phrase'] == source['text']
                assert row['path'] == key + '-' + condition + '.wav'
                assert row['samples'] == samples and row['sample_rate'] == 16000
                assert row['lead_samples'] == 32000 and row['signal_samples'] == len(signal)
                assert row['full_generated_source_end_sample'] == 32000 + len(dry_sources[key])
                assert row['rir_tail_samples'] == (960 if condition == 'combined15' else 0)
                assert row['generator_prefix_end_sample'] == 32000 + round(source['generator_prefix_end_sample'] * 320 / 441)
                assert samples - (32000 + len(signal)) >= 16000
        headroom = min(1., .95 / max(np.max(np.abs(w)) for w in waves.values()))
        for key, wave in waves.items():
            row = rendered[key]
            assert headroom == row['common_group_headroom_gain']
            path = args.data / row['path']
            payload = path.read_bytes()
            assert digest(payload) == row['sha256']
            info = sf.info(path)
            assert info.format == 'WAV' and info.subtype == 'PCM_16' and info.channels == 1
            pcm, rate = sf.read(path, dtype='int16')
            assert rate == 16000 and len(pcm) == samples and np.any(pcm)
            assert not np.any((pcm == -32768) | (pcm == 32767))
            assert digest(pcm.astype('<i2').tobytes()) == row['pcm_sha256']
            # The ideal floor oracle can differ by 1LSB at libsndfile's
            # conversion boundaries. Additionally require the reconstructed
            # float waveform re-encoded by libsndfile to match every WAV byte.
            expected_pcm = np.floor(wave * headroom * 32768).astype(np.int16)
            difference = int(np.max(np.abs(pcm.astype(np.int32) - expected_pcm.astype(np.int32))))
            assert difference <= 1, (path.name, difference)
            reencoded = io.BytesIO()
            sf.write(reencoded, wave * headroom, 16000, format='WAV', subtype='PCM_16')
            assert reencoded.getvalue() == payload, path.name
            max_pcm_difference = max(max_pcm_difference, difference)
            hashes_verified += 1
    report = dict(
        schema='bryce-heldout-development-independent-audit-v1',
        scope='Saved source/config/phoneme-duration/PCM integrity only; no wake-model inference or acoustic quality acceptance.',
        auditor_sha256=digest(Path(__file__).read_bytes()),
        plan_sha256=digest(plan_bytes), manifest_sha256=digest(manifest_bytes),
        generator_sha256=manifest['generator_sha256'],
        native_files_verified=len(native), wav_files_verified=hashes_verified,
        native_distribution=dict(Counter(r['voice'] for r in native.values())),
        render_distribution=dict(Counter(r['condition'] for r in rendered.values())),
        alignment_duration_samples_exact=True, all_raw_sources_and_rir_tails_preserved=True,
        independent_quantized_pcm_max_abs_difference=max_pcm_difference,
        independently_reconstructed_wav_bytes_exact=True,
        prefix_reference_ms={k: dict(min=min(v), max=max(v)) for k, v in sorted(prefix_ms.items())},
        source_speed='Native length_scale=1; rational sample-rate conversion; wall-clock duration within one16k sample.',
        wake_models_scored=False, training_data=False, acceptance_pass=None,
        limits=manifest['limits'])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k not in ('limits', 'prefix_reference_ms')}))


if __name__ == '__main__':
    main()
