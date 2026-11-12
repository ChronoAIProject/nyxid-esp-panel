"""Materialize frozen held-out synthetic development; never load wake models."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform

import numpy as np
import onnx
import onnxruntime as ort
from piper import PiperVoice, SynthesisConfig
from piper.config import PiperConfig
from scipy.signal import fftconvolve, resample_poly
import soundfile as sf


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('plan', 'models', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    plan = json.loads(args.plan.read_text())
    assert sha(Path(__file__)) == plan['generator_sha256']
    assert platform.python_version() == plan['python']
    for package, version in plan['versions'].items():
        assert importlib.metadata.version(package) == version, package
    assert not plan['wake_model_scoring'] and plan['role'] == 'synthetic-development-only'
    args.output.mkdir(parents=True)
    (args.output / 'plan.json').write_bytes(args.plan.read_bytes())
    native_records, records, failures = [], [], []
    ir = np.zeros(961, np.float64)
    ir[[0, 160, 480, 960]] = [1., .25, -.12, .06]
    for voice_index, voice_spec in enumerate(plan['voices']):
        name = voice_spec['name']
        for filename, digest in voice_spec['files'].items():
            assert sha(args.models / filename) == digest
        original = onnx.load(args.models / (name + '.onnx'))
        aligned = onnx.load(args.models / (name + '-alignment.onnx'))
        assert [x.SerializeToString() for x in original.graph.node] == [x.SerializeToString() for x in aligned.graph.node]
        assert [x.SerializeToString() for x in original.graph.initializer] == [x.SerializeToString() for x in aligned.graph.initializer]
        assert [o.name for o in aligned.graph.output] == [original.graph.output[0].name, '/Ceil_output_0']
        config = PiperConfig.from_dict(json.loads((args.models / (name + '.onnx.json')).read_text()))
        assert config.sample_rate == 22050 and config.hop_length == 256
        prefixes = {}
        for key, phones in voice_spec['expected_phonemes'].items():
            stop = next(i for i, phone in enumerate(phones) if phone in (',', '.', '?', '!'))
            prefixes[key] = tuple(phones[:stop])
        assert prefixes['hey_nyx'] == prefixes['hey_nyx_command']
        assert prefixes['hi_esp'] == prefixes['hi_esp_command']
        assert len({prefixes[key] for key in ('hey_nyx', 'hey_nick', 'hey_next', 'hey_mix')}) == 4
        for draw in range(plan['draws_per_voice']):
            seed = plan['seed_base'] + 1000 * voice_index + draw
            ort.set_seed(seed)
            options = ort.SessionOptions()
            options.intra_op_num_threads = 2
            options.inter_op_num_threads = 1
            session = ort.InferenceSession(str(args.models / (name + '-alignment.onnx')),
                                          sess_options=options, providers=['CPUExecutionProvider'])
            voice = PiperVoice(session=session, config=config)
            synth = SynthesisConfig(length_scale=1., noise_scale=.667, noise_w_scale=.8,
                                    normalize_audio=False)
            group = []
            for phrase in plan['phrases']:
                key = name + '-' + str(draw) + '-' + phrase['id']
                sentences = voice.phonemize(phrase['text'])
                assert len(sentences) == 1
                phones = sentences[0]
                assert phones == voice_spec['expected_phonemes'][phrase['id']]
                ids = voice.phonemes_to_ids(phones)
                audio, durations = voice.phoneme_ids_to_audio(ids, synth, include_alignments=True)
                audio = np.asarray(audio, np.float32)
                # Persist the original unnormalized, unclipped model output
                # before any engineering/quality assertion can fail.
                raw_path = args.output / (key + '-native.npy')
                np.save(raw_path, audio, allow_pickle=False)
                assert audio.ndim == 1 and np.isfinite(audio).all()
                assert durations is not None and len(durations) == len(ids)
                durations = np.asarray(durations, np.int64)
                assert np.all(durations >= 0) and int(durations.sum()) == len(audio)
                at, sample, alignment = 0, 0, []
                for phone in ['^', *phones, '$']:
                    expected = list(config.phoneme_id_map.get(phone, []))
                    if phone != '$':
                        expected += list(config.phoneme_id_map.get('_', []))
                    assert ids[at:at + len(expected)] == expected
                    width = int(durations[at:at + len(expected)].sum())
                    alignment.append(dict(phone=phone, start_sample=sample, end_sample=sample + width))
                    at += len(expected)
                    sample += width
                assert at == len(ids) and sample == len(audio)
                # This is a synthesis duration reference. Decoder acoustics
                # can straddle it; it is not a human-annotated phone boundary.
                delimiter = next(i for i, p in enumerate(alignment)
                                 if p['phone'] in (',', '.', '?', '!'))
                keyword_end = alignment[delimiter - 1]['end_sample']
                assert 0 < keyword_end <= len(audio)
                dry = resample_poly(audio.astype(np.float64), 320, 441)
                assert abs(len(dry) / 16000 - len(audio) / 22050) <= 1 / 16000
                rms = float(np.sqrt(np.mean(dry ** 2)))
                mechanical = .2 <= len(audio) / 22050 <= 10 and rms >= 1e-6 and np.max(np.abs(audio)) < 2
                if not mechanical:
                    failures.append(dict(key=key, reason='native mechanical quality'))
                row = dict(key=key, voice=name, draw=draw, generation_seed=seed, **phrase,
                           source_rate=22050, source_samples=len(audio), native_path=raw_path.name,
                           native_sha256=sha(raw_path), native_pcm_sha256=hashlib.sha256(audio.tobytes()).hexdigest(),
                           phonemes=phones, phoneme_ids=ids, phoneme_id_samples=durations.tolist(),
                           generator_alignment=alignment, generator_prefix_end_sample=keyword_end,
                           resampled_samples=len(dry), mechanical_pass=bool(mechanical))
                native_records.append(row)
                group.append((row, dry, rms))
            size = 32000 + max(len(dry) + 960 for _, dry, _ in group) + 16000
            waves = []
            for row, dry, rms in group:
                seed_text = 'heldout-bryce-v1:' + row['key']
                noise_seed = int(hashlib.sha256(seed_text.encode()).hexdigest()[:16], 16)
                noise = np.random.default_rng(noise_seed).normal(0, 1, size)
                noise /= np.sqrt(np.mean(noise ** 2))
                for condition in plan['conditions']:
                    signal = fftconvolve(dry, ir, mode='full') if condition == 'combined15' else dry
                    gain = .08 / np.sqrt(np.mean(signal ** 2))
                    wave = np.pad(signal * gain, (32000, size - 32000 - len(signal)))
                    snr = 25 if condition == 'noise25' else 15
                    if condition != 'quiet':
                        wave += noise * (.08 * 10 ** (-snr / 20))
                    waves.append((row, condition, wave, noise_seed, gain, len(signal)))
            common_gain = min(1., .95 / max(np.max(np.abs(w)) for _, _, w, _, _, _ in waves))
            for row, condition, wave, noise_seed, level_gain, signal_samples in waves:
                wave *= common_gain
                path = args.output / (row['key'] + '-' + condition + '.wav')
                sf.write(path, wave, 16000, subtype='PCM_16')
                pcm, rate = sf.read(path, dtype='int16')
                assert rate == 16000 and len(pcm) == size and np.any(pcm)
                assert not np.any((pcm == -32768) | (pcm == 32767))
                records.append(dict(voice=row['voice'], draw=draw, target=row['target'],
                    intended_phrase=row['text'], source_key=row['key'], condition=condition,
                    path=path.name, samples=size, sample_rate=rate, sha256=sha(path),
                    pcm_sha256=hashlib.sha256(pcm.astype('<i2').tobytes()).hexdigest(),
                    lead_samples=32000, noise_seed=noise_seed, level_gain=level_gain,
                    common_group_headroom_gain=float(common_gain), signal_samples=signal_samples,
                    generator_prefix_end_sample=32000 + round(row['generator_prefix_end_sample'] * 16000 / 22050),
                    full_generated_source_end_sample=32000 + row['resampled_samples'],
                    rir_tail_samples=960 if condition == 'combined15' else 0))
            # Keep complete groups even when a mechanical gate fails. Never
            # regenerate or filter sources using wake scores.
            (args.output / 'native-progress.json').write_text(json.dumps(native_records, indent=2) + '\n')
            print(json.dumps(dict(voice=name, draw=draw, native=len(native_records), renders=len(records))), flush=True)
            del voice, session
    assert len(native_records) == 42 and len(records) == 126
    manifest = dict(role=plan['role'], plan_sha256=sha(args.plan), generator_sha256=sha(Path(__file__)),
        native_records=native_records, records=records, mechanical_failures=failures,
        all_mechanical_pass=not failures, wake_models_scored=False, training_data=False,
        limits=plan['limits'])
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(native=42, renders=126, mechanical_failures=len(failures),
                         manifest_sha256=sha(args.output / 'manifest.json'))))


if __name__ == '__main__':
    main()
