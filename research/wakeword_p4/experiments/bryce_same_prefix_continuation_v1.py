"""Frozen v30 diagnostic: identical wake prefix with three dry-tail cutoffs.

Consumed synthetic development only. No new model, threshold selection, TTS,
playback, recording, training or firmware operation. See the companion protocol.
"""
import argparse
from collections import Counter
import hashlib
from importlib.metadata import version
import io
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np
from scipy.signal import fftconvolve, resample_poly
import soundfile as sf

from score_bryce_v30_baseline_v1 import classify, summarize

TRAINING = Path(__file__).resolve().parents[1] / 'training'
sys.path.insert(0, str(TRAINING))
from evaluate import Detector
from features import features
from replay_gpu_mac_regression import policy_result


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def array_sha(value):
    return hashlib.sha256(value.tobytes()).hexdigest()


def wav_bytes_and_pcm(wave):
    buffer = io.BytesIO()
    sf.write(buffer, wave, 16000, format='WAV', subtype='PCM_16')
    blob = buffer.getvalue()
    pcm, rate = sf.read(io.BytesIO(blob), dtype='int16')
    assert rate == 16000 and pcm.ndim == 1
    return blob, pcm.astype('<i2')


def render(dry, row, noise, ir):
    signal = fftconvolve(dry, ir, mode='full') if row['condition'] == 'combined15' else dry
    assert len(signal) == row['signal_samples']
    # Original full-source gains and common headroom stay fixed even when the
    # dry suffix has been removed. Never normalize a modified signal.
    wave = np.pad(signal * row['level_gain'],
                  (row['lead_samples'], row['samples'] - row['lead_samples'] - len(signal)))
    if row['condition'] != 'quiet':
        snr = 25 if row['condition'] == 'noise25' else 15
        wave += noise * (.08 * 10 ** (-snr / 20))
    return wave * row['common_group_headroom_gain']


def prepare(data, manifest, offsets_ms):
    """Authenticate/reconstruct all sources and controls before loading model."""
    ir = np.zeros(961, np.float64)
    ir[[0, 160, 480, 960]] = [1., .25, -.12, .06]
    native = {r['key']: r for r in manifest['native_records']}
    dry_cache = {}
    for key, source in native.items():
        path = data / source['native_path']
        assert sha(path) == source['native_sha256']
        audio = np.load(path, allow_pickle=False)
        assert audio.dtype == np.float32 and len(audio) == source['source_samples']
        assert audio.ndim == 1 and np.isfinite(audio).all()
        assert array_sha(audio) == source['native_pcm_sha256']
        dry = resample_poly(audio.astype(np.float64), 320, 441)
        assert len(dry) == source['resampled_samples']
        dry_cache[key] = dry
    prepared = []
    for row in manifest['records']:
        path = data / row['path']
        assert not Path(row['path']).is_absolute() and '..' not in Path(row['path']).parts
        assert sha(path) == row['sha256']
        original, rate = sf.read(path, dtype='int16')
        original = original.astype('<i2')
        assert rate == 16000 and len(original) == row['samples'] and original.ndim == 1
        assert array_sha(original) == row['pcm_sha256']
        dry = dry_cache[row['source_key']]
        noise = np.random.default_rng(row['noise_seed']).normal(0, 1, row['samples'])
        noise /= np.sqrt(np.mean(noise ** 2))
        full_float = render(dry, row, noise, ir)
        full_blob, reconstructed = wav_bytes_and_pcm(full_float)
        assert full_blob == path.read_bytes()
        np.testing.assert_array_equal(original, reconstructed)
        controls, details = {'original': original}, {}
        for offset in offsets_ms:
            name = f'dry_stop_{offset}ms'
            requested = row['generator_prefix_end_sample'] + offset * 16
            dry_cut = min(len(dry), requested - row['lead_samples'])
            assert 0 < dry_cut <= len(dry)
            cutoff = row['lead_samples'] + dry_cut
            if dry_cut == len(dry):
                modified, rounding_changes, max_prefix_float_difference = original.copy(), 0, 0.
            else:
                cut_dry = dry.copy()
                cut_dry[dry_cut:] = 0
                cut_float = render(cut_dry, row, noise, ir)
                _, modified = wav_bytes_and_pcm(cut_float)
                # FFT length stays fixed, but different suffixes may change
                # roundoff in prefix values. Explicitly preserve exact causal
                # original PCM, including any rounding-boundary samples.
                rounding_changes = int(np.count_nonzero(modified[:cutoff] != original[:cutoff]))
                max_prefix_float_difference = float(np.max(np.abs(cut_float[:cutoff] - full_float[:cutoff])))
                assert max_prefix_float_difference < 1e-12
                modified[:cutoff] = original[:cutoff]
            assert len(modified) == len(original)
            np.testing.assert_array_equal(modified[:cutoff], original[:cutoff])
            changed = np.flatnonzero(modified != original)
            assert len(changed) == 0 or int(changed[0]) >= cutoff
            controls[name] = modified
            details[name] = dict(requested_cutoff_sample=requested,
                effective_dry_cutoff_sample=cutoff, dry_samples_removed=len(dry) - dry_cut,
                exact_original_pcm_prefix_samples=cutoff,
                fft_rounding_prefix_pcm_values_restored=rounding_changes,
                max_prefix_float_roundoff=max_prefix_float_difference,
                changed_pcm_samples=len(changed),
                first_changed_pcm_sample=int(changed[0]) if len(changed) else None,
                original_noise_seed_and_gains_retained=True, rir_tail_retained=True)
        prepared.append((row, native[row['source_key']], controls, details))
    return prepared


def run(detector, pcm):
    tail = (-len(pcm)) % 480
    padded = np.pad(pcm, (56160, tail)).astype('<i2')
    raw = features(padded)
    detector.reset()
    scores = detector.predict(raw)
    assert len(scores) == len(padded) // 480 - 1
    quantized = np.rint(scores * 256).astype(np.uint8)
    np.testing.assert_array_equal(scores, quantized.astype(np.float32) / 256)
    policy = dict(smoothing_steps=5, v30_threshold=235 / 256,
                  v30_block_samples=480, v30_prime_samples=56160)
    host, firmware = policy_result(scores, policy)
    assert host == firmware and not firmware['prime_event_offsets']
    return raw, quantized, firmware


def score_summary(quantized):
    times = (np.arange(len(quantized)) + 2) * 480 - 56160
    sums = np.convolve(quantized.astype(np.int64), np.ones(5, dtype=np.int64), mode='valid')
    return dict(raw_max_uint8=int(quantized[times > 0].max()),
                five_score_max_sum_uint8=int(sums[times[4:] > 0].max()))


def paired(rows, variant):
    original = [r['variants']['original']['classification'] for r in rows]
    changed = [r['variants'][variant]['classification'] for r in rows]
    result = dict(clips=len(rows))
    for field in ('any_event', 'reference_window_hit', 'false_activation'):
        pairs = [(a[field], b[field]) for a, b in zip(original, changed, strict=True) if field in a]
        result[field] = dict(clips=len(pairs), original=sum(a for a, _ in pairs),
                             modified=sum(b for _, b in pairs),
                             lost=sum(a and not b for a, b in pairs),
                             gained=sum(b and not a for a, b in pairs))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('protocol', 'data', 'study', 'model', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    protocol = json.loads(args.protocol.read_text())
    assert protocol['schema'] == 'bryce-same-prefix-continuation-protocol-v1'
    assert protocol['source_sha256'] == sha(Path(__file__))
    assert protocol['cutoffs_after_generator_prefix_ms'] == [0, 100, 200]
    for package, expected in protocol['host_versions'].items():
        assert version(package) == expected
    repo = Path(__file__).resolve().parents[3]
    for relative, expected in protocol['helper_source_sha256'].items():
        assert sha(repo / relative) == expected
    for path, key in ((args.data / 'manifest.json', 'manifest_sha256'),
                      (args.data / 'plan.json', 'generation_plan_sha256'),
                      (args.study / 'report.json', 'baseline_report_sha256'),
                      (args.study / 'scores.npz', 'baseline_scores_sha256'),
                      (args.model, 'model_sha256')):
        assert sha(path) == protocol[key], str(path)
    manifest = json.loads((args.data / 'manifest.json').read_text())
    baseline = json.loads((args.study / 'report.json').read_text())
    assert manifest['role'] == 'synthetic-development-only' and not manifest['training_data']
    assert len(manifest['records']) == len(baseline['records']) == 126
    assert dict(Counter(r['target'] for r in manifest['records'])) == protocol['target_counts']
    prepared = prepare(args.data, manifest, protocol['cutoffs_after_generator_prefix_ms'])
    args.output.mkdir(parents=True)
    shutil.copyfile(__file__, args.output / 'source.py')
    shutil.copyfile(args.protocol, args.output / 'protocol.json')
    detector = Detector(args.model)
    traces, original_features, original_policies = {}, [], []
    started = time.monotonic()
    # Replay every original before scoring any intervention. No partial
    # original match permits interpreting an altered wave.
    with np.load(args.study / 'scores.npz', allow_pickle=False) as old:
        for index, (row, _, controls, _) in enumerate(prepared):
            reference = baseline['records'][index]
            assert reference['record'] == row
            raw, scores, policy = run(detector, controls['original'])
            np.testing.assert_array_equal(scores, old[f'output_{index}'])
            assert array_sha(raw) == reference['frontend_sha256']
            assert policy == reference['firmware_policy']
            traces[f'output_{index}__original'] = scores
            original_features.append(raw)
            original_policies.append(policy)
    print(json.dumps(dict(all_126_original_frontends_scores_events_exact=True,
                          seconds=round(time.monotonic() - started, 2))), flush=True)
    details = []
    variants = ['original'] + [f'dry_stop_{n}ms' for n in protocol['cutoffs_after_generator_prefix_ms']]
    for index, (row, source, controls, control_details) in enumerate(prepared):
        output = dict(path=row['path'], record=row, native_phrase_id=source['id'], variants={})
        for name in variants:
            pcm = controls[name]
            if name == 'original':
                raw = original_features[index]
                scores, policy = traces[f'output_{index}__original'], original_policies[index]
                preservation = dict(original_replay_exact=True)
            else:
                raw, scores, policy = run(detector, pcm)
                control = control_details[name]
                # MicroFrontend result j arrives after (j+3)*160 samples.
                # Each streaming invocation consumes three such frames.
                identical_frames = min(len(raw), max(0, (56160 + control['effective_dry_cutoff_sample']) // 160 - 2))
                identical_scores = identical_frames // 3
                np.testing.assert_array_equal(raw[:identical_frames], original_features[index][:identical_frames])
                np.testing.assert_array_equal(scores[:identical_scores], traces[f'output_{index}__original'][:identical_scores])
                traces[f'output_{index}__{name}'] = scores
                preservation = dict(**control, causal_feature_frames_exact=identical_frames,
                                    causal_output_scores_exact=identical_scores)
            output['variants'][name] = dict(
                pcm_sha256=array_sha(pcm), pcm_samples=len(pcm),
                frontend_sha256=array_sha(raw), quantized_output_sha256=array_sha(scores),
                output_count=len(scores), policy=policy, preservation=preservation,
                classification=classify(policy['offsets'], row, source, protocol['reference_window_samples']),
                **score_summary(scores))
        details.append(output)
        if (index + 1) % 21 == 0:
            print(json.dumps(dict(completed=index + 1, total=126,
                                  seconds=round(time.monotonic() - started, 2))), flush=True)
    summaries, comparisons = {}, {}
    for name in variants:
        formatted = [dict(classification=r['variants'][name]['classification']) for r in details]
        summaries[name] = summarize(formatted)
        if name == 'original':
            assert summaries[name] == baseline['summaries']['all']
        else:
            groups = {'all': details}
            for voice in ('kristin', 'norman'):
                for phrase in ('hey_nyx', 'hey_nyx_command'):
                    groups[voice + '/' + phrase] = [r for r in details
                        if r['record']['voice'] == voice and r['native_phrase_id'] == phrase]
            comparisons[name] = {group: paired(rows, name) for group, rows in groups.items()}
    np.savez_compressed(args.output / 'scores.npz', **traces)
    report = dict(schema='bryce-same-prefix-continuation-result-v1',
        source_sha256=sha(Path(__file__)), protocol_sha256=sha(args.protocol),
        model_sha256=sha(args.model), manifest_sha256=sha(args.data / 'manifest.json'),
        baseline_report_sha256=sha(args.study / 'report.json'),
        scores_sha256=sha(args.output / 'scores.npz'),
        original_wav_reconstruction_all_126_byte_exact=True,
        original_frontends_scores_events_all_126_exact=True,
        all_modified_causal_pcm_feature_output_prefixes_exact=True,
        threshold=235 / 256, smoothing_steps=5, refractory_steps=67,
        source_clips=126, interventions_per_clip=3,
        summaries=summaries, paired_comparisons=comparisons, records=details,
        training=False, final_data_accessed=False, threshold_search=False,
        hardware_operations=False, acceptance_claim=False, limits=protocol['limits'])
    (args.output / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(summaries=summaries, paired_comparisons=comparisons)), flush=True)


if __name__ == '__main__':
    main()
