"""Score every frozen Kristin/Norman development WAV with historical v30 only."""
import argparse
from collections import Counter
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import soundfile as sf

TRAINING = Path(__file__).resolve().parents[1] / 'training'
sys.path.insert(0, str(TRAINING))
from evaluate import Detector
from features import features
from prepare_runtime_selftest import inspect_model
from replay_gpu_mac_regression import policy_result


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def classify(offsets, row, native, window):
    if row['target'] != 'hey_nyx':
        kind = 'opposite_wake_phrase' if row['target'] == 'hi_esp' else 'common_confuser'
        return dict(negative_kind=kind, false_events=len(offsets), false_activation=bool(offsets))
    end = row['generator_prefix_end_sample']
    hits = [t for t in offsets if end + window[0] <= t <= end + window[1]]
    return dict(positive_kind='command' if native['id'].endswith('_command') else 'bare',
                any_event=bool(offsets), reference_window_hit=bool(hits),
                first_any_event_reference_latency_ms=(offsets[0] - end) / 16 if offsets else None,
                first_window_event_reference_latency_ms=(hits[0] - end) / 16 if hits else None,
                premature_events=sum(t < end + window[0] for t in offsets),
                late_events=sum(t > end + window[1] for t in offsets))


def summarize(rows):
    positives = [r['classification'] for r in rows if 'positive_kind' in r['classification']]
    latencies = [r['first_window_event_reference_latency_ms'] for r in positives if r['reference_window_hit']]
    result = dict(clips=len(rows), positive_clips=len(positives),
                  positive_any_event_clips=sum(r['any_event'] for r in positives),
                  positive_reference_window_hits=sum(r['reference_window_hit'] for r in positives),
                  premature_events=sum(r['premature_events'] for r in positives),
                  late_events=sum(r['late_events'] for r in positives),
                  reference_latency_ms_median=float(np.median(latencies)) if latencies else None,
                  reference_latency_ms_p95=float(np.quantile(latencies, .95)) if latencies else None)
    for kind in ('common_confuser', 'opposite_wake_phrase'):
        negatives = [r['classification'] for r in rows if r['classification'].get('negative_kind') == kind]
        result[kind] = dict(clips=len(negatives), false_events=sum(r['false_events'] for r in negatives),
                            false_activation_clips=sum(r['false_activation'] for r in negatives))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('protocol', 'data', 'model-dir', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    protocol = json.loads(args.protocol.read_text())
    assert protocol['schema'] == 'bryce-historical-baseline-protocol-v1'
    assert protocol['v30_scorer_sha256'] == sha(Path(__file__))
    repo = Path(__file__).resolve().parents[3]
    for relative, expected in protocol['helper_source_sha256'].items():
        assert sha(repo / relative) == expected, relative
    for package, expected in protocol['host_versions'].items():
        assert version(package) == expected, package
    manifest_path = args.data / 'manifest.json'
    assert sha(manifest_path) == protocol['data_manifest_sha256']
    assert sha(args.data / 'plan.json') == protocol['generation_plan_sha256']
    manifest = json.loads(manifest_path.read_text())
    rows = manifest['records']
    assert manifest['role'] == 'synthetic-development-only'
    assert not manifest['training_data'] and not manifest['wake_models_scored']
    assert len(rows) == 126 and manifest['all_mechanical_pass']
    assert dict(Counter(r['target'] for r in rows)) == protocol['target_counts']
    assert dict(Counter(r['voice'] for r in rows)) == {'kristin': 63, 'norman': 63}
    assert len({r['path'] for r in rows}) == 126
    native = {r['key']: r for r in manifest['native_records']}
    model = args.model_dir / 'hey_nyx.tflite'
    model_spec = protocol['v30']
    for name, field in [('hey_nyx.tflite', 'model_sha256'), ('training.json', 'training_report_sha256'),
                        ('4090-v30-expanded-calibration.json', 'calibration_sha256')]:
        assert sha(args.model_dir / name) == model_spec[field]
    calibration = json.loads((args.model_dir / '4090-v30-expanded-calibration.json').read_text())
    assert calibration['model_sha256'] == model_spec['model_sha256']
    assert calibration['threshold'] == model_spec['threshold'] == .91796875
    assert calibration['sliding_window'] == model_spec['smoothing_steps'] == 5
    assert model_spec['refractory_steps'] == 67 and model_spec['prime_samples'] == 56160
    assert model_spec['block_samples'] == 480
    # Validate the complete cohort before constructing/loading a detector.
    for row in rows:
        relative = Path(row['path'])
        assert not relative.is_absolute() and '..' not in relative.parts
        path = args.data / relative
        assert sha(path) == row['sha256']
        info = sf.info(path)
        pcm, rate = sf.read(path, dtype='int16')
        assert rate == 16000 and pcm.ndim == 1 and len(pcm) == row['samples']
        assert info.subtype == 'PCM_16' and info.channels == 1
        assert hashlib.sha256(pcm.astype('<i2').tobytes()).hexdigest() == row['pcm_sha256']
        source = native[row['source_key']]
        assert 0 < row['generator_prefix_end_sample'] <= len(pcm)
        assert source['target'] == row['target'] and source['text'] == row['intended_phrase']
    args.output.mkdir(parents=True)
    shutil.copyfile(args.protocol, args.output / 'protocol.json')
    shutil.copyfile(__file__, args.output / 'source.py')
    detector = Detector(model)
    interface = inspect_model(model.read_bytes(), detector)
    policy = dict(v30_threshold=model_spec['threshold'], smoothing_steps=5,
                  v30_block_samples=480, v30_prime_samples=56160)
    traces, details, differences = {}, [], []
    started = time.monotonic()
    for index, row in enumerate(rows):
        pcm, _ = sf.read(args.data / row['path'], dtype='int16')
        tail = (-len(pcm)) % 480
        padded = np.pad(pcm, (56160, tail)).astype('<i2')
        raw = features(padded)
        detector.reset()
        scores = detector.predict(raw)
        detector.reset()
        np.testing.assert_array_equal(scores, detector.predict(raw))
        assert len(scores) == len(padded) // 480 - 1
        quantized = np.rint(scores * 256).astype(np.uint8)
        np.testing.assert_array_equal(scores, quantized.astype(np.float32) / 256)
        traces[f'output_{index}'] = quantized
        host, firmware = policy_result(scores, policy)
        if host != firmware:
            differences.append(row['path'])
        assert all(0 < t <= len(pcm) + tail for t in firmware['offsets'])
        details.append(dict(record=row, native_phrase_id=native[row['source_key']]['id'],
                            prime_samples=56160, final_zero_pad_samples=tail,
                            frontend_sha256=hashlib.sha256(raw.tobytes()).hexdigest(),
                            frontend_shape=list(raw.shape), output_count=len(scores),
                            quantized_output_sha256=hashlib.sha256(quantized.tobytes()).hexdigest(),
                            reset_exact=True, host_policy=host, firmware_policy=firmware,
                            classification=classify(firmware['offsets'], row, native[row['source_key']],
                                                    protocol['reference_window_samples'])))
        if (index + 1) % 21 == 0:
            print(json.dumps(dict(completed=index + 1, total=126,
                                  seconds=round(time.monotonic() - started, 2))), flush=True)
    np.savez_compressed(args.output / 'scores.npz', **traces)
    result = dict(
        schema='bryce-v30-historical-development-baseline-v1',
        protocol_sha256=sha(args.protocol), source_sha256=sha(Path(__file__)),
        data_manifest_sha256=sha(manifest_path), model_sha256=sha(model),
        scores_sha256=sha(args.output / 'scores.npz'), interface=interface,
        all_sources_hash_verified=True, all_reset_replays_exact=True,
        threshold_selected=False, threshold=model_spec['threshold'],
        background_hours=0, far_estimate=None, accepted=None,
        host_firmware_policy_differences=differences,
        summaries=dict(all=summarize(details), by_voice={
            voice: summarize([r for r in details if r['record']['voice'] == voice]) for voice in ('kristin', 'norman')},
            by_condition={c: summarize([r for r in details if r['record']['condition'] == c])
                          for c in ('quiet', 'noise25', 'combined15')},
            by_positive_phrase={p: summarize([r for r in details if r['native_phrase_id'] == p])
                                for p in ('hey_nyx', 'hey_nyx_command')}),
        hiesp_result=None, records=details,
        limits=protocol['limits'])
    (args.output / 'report.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(sources=126, summary=result['summaries']['all'],
                          host_firmware_policy_differences=differences, accepted=None)))


if __name__ == '__main__':
    main()
