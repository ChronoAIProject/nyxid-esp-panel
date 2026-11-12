"""Historical v30 on every complete reserved Earnings22 development recording."""
import argparse
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np
from scipy.stats import chi2

TRAINING = Path(__file__).resolve().parents[1] / 'training'
sys.path.insert(0, str(TRAINING))
from evaluate import Detector
from detector_policy import smooth
from prepare_runtime_selftest import events, inspect_model


def sha(path):
    result = hashlib.sha256()
    with path.open('rb') as source:
        while data := source.read(1024 * 1024):
            result.update(data)
    return result.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('protocol', 'prepared', 'model-dir', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    protocol = json.loads(args.protocol.read_text())
    assert protocol['schema'] == 'earnings22-historical-v30-score-protocol-v1'
    assert sha(Path(__file__)) == protocol['scorer_sha256']
    repo = Path(__file__).resolve().parents[3]
    for relative, expected in protocol['helper_source_sha256'].items():
        assert sha(repo / relative) == expected
    for package, expected in protocol['host_versions'].items():
        assert version(package) == expected
    manifest_path = args.prepared / 'manifest.json'
    assert sha(manifest_path) == protocol['prepared_manifest_sha256']
    manifest = json.loads(manifest_path.read_text())
    rows = manifest['records']
    assert len(rows) == 12 and manifest['complete_recordings']
    assert not manifest['training'] and not manifest['final_data_accessed']
    assert not manifest['inserted_intermediate_gaps'] and not manifest['normalized']
    assert not manifest['wake_models_scored']  # Immutable preparation-stage ledger.
    model_spec = protocol['v30']
    for name, field in [('hey_nyx.tflite', 'model_sha256'), ('training.json', 'training_report_sha256'),
                        ('4090-v30-expanded-calibration.json', 'calibration_sha256')]:
        assert sha(args.model_dir / name) == model_spec[field]
    calibration = json.loads((args.model_dir / '4090-v30-expanded-calibration.json').read_text())
    assert calibration['threshold'] == model_spec['threshold'] == .91796875
    assert calibration['sliding_window'] == model_spec['smoothing_steps'] == 5
    assert model_spec['refractory_steps'] == 67 and model_spec['extra_prime_samples'] == 0
    # Validate every full source and feature payload before loading the model.
    for row in rows:
        for field, expected in [('path', 'sha256'), ('features_path', 'features_sha256')]:
            relative = Path(row[field])
            assert not relative.is_absolute() and '..' not in relative.parts
            assert sha(args.prepared / relative) == row[expected]
        assert row['source_seconds'] == row['samples'] / 16000
        assert row['frontend_final_zero_pad_samples'] == (-row['samples']) % 160
        assert not row['transcript_audit']['exact_target_text_matches']
        raw = np.load(args.prepared / row['features_path'], mmap_mode='r', allow_pickle=False)
        assert raw.dtype == np.uint16 and list(raw.shape) == row['features_shape']
        assert raw.shape == ((row['samples'] + 159) // 160 - 2, 40)
        assert hashlib.sha256(raw.tobytes()).hexdigest() == row['features_array_sha256']
    args.output.mkdir(parents=True)
    shutil.copyfile(args.protocol, args.output / 'protocol.json')
    shutil.copyfile(__file__, args.output / 'source.py')
    detector = Detector(args.model_dir / 'hey_nyx.tflite')
    interface = inspect_model((args.model_dir / 'hey_nyx.tflite').read_bytes(), detector)
    traces, records, policy_differences = {}, [], []
    started = time.monotonic()
    for index, row in enumerate(rows):
        raw = np.load(args.prepared / row['features_path'], mmap_mode='r', allow_pickle=False)
        detector.reset()
        scores = detector.predict(raw)
        detector.reset()
        np.testing.assert_array_equal(scores, detector.predict(raw))
        assert len(scores) == len(raw) // 3 and np.isfinite(scores).all()
        quantized = np.rint(scores * 256).astype(np.uint8)
        np.testing.assert_array_equal(scores, quantized.astype(np.float32) / 256)
        traces[f'output_{index}'] = quantized
        host = smooth(scores, 5)
        firmware = np.lib.stride_tricks.sliding_window_view(scores.astype(np.float64), 5).sum(axis=1) / 5
        host_steps = events(host, model_spec['threshold'], 5)
        firmware_steps = events(firmware, float(np.float32(model_spec['threshold'])), 5)
        if host_steps != firmware_steps:
            policy_differences.append(row['source_id'])
        # First raw output consumes feature0..2, available at native sample800.
        # These are feature-availability times, not application block returns.
        offsets = [(3 * (step - 1) + 5) * 160 for step in firmware_steps]
        assert all(0 < at <= row['samples'] + row['frontend_final_zero_pad_samples'] for at in offsets)
        records.append(dict(source_id=row['source_id'], samples=row['samples'],
                            source_seconds=row['source_seconds'], pcm_sha256=row['pcm_sha256'],
                            features_sha256=row['features_sha256'], output_count=len(scores),
                            quantized_output_sha256=hashlib.sha256(quantized.tobytes()).hexdigest(),
                            reset_exact=True, host_event_steps=host_steps, firmware_event_steps=firmware_steps,
                            event_sample_offsets=offsets, events=len(offsets)))
        print(json.dumps(dict(completed=index + 1, total=12, source_id=row['source_id'],
                              events=len(offsets), source_hours=row['source_seconds'] / 3600,
                              seconds=round(time.monotonic() - started, 2))), flush=True)
    np.savez_compressed(args.output / 'scores.npz', **traces)
    hours = sum(r['source_seconds'] for r in records) / 3600
    count = sum(r['events'] for r in records)
    report = dict(schema='earnings22-historical-v30-continuous-development-v1',
                  protocol_sha256=sha(args.protocol), prepared_manifest_sha256=sha(manifest_path),
                  source_sha256=sha(Path(__file__)), model_sha256=model_spec['model_sha256'],
                  scores_sha256=sha(args.output / 'scores.npz'), threshold=model_spec['threshold'],
                  interface=interface, records=records, recordings=12, source_hours=hours,
                  negative_intended_background_events=count, event_recordings=sum(r['events'] > 0 for r in records),
                  nominal_events_per_hour=count / hours,
                  nominal_poisson_upper_95_per_hour=float(.5 * chi2.ppf(.95, 2 * (count + 1)) / hours),
                  host_firmware_policy_differences=policy_differences,
                  all_sources_authenticated_before_model_load=True, all_reset_replays_exact=True,
                  threshold_selected=False, current_gpu_candidate_scored=False, hiesp_result=None,
                  final_data_accessed=False, training=False, accepted=None,
                  limits=protocol['limits'])
    (args.output / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k not in ('records', 'interface', 'limits')}))


if __name__ == '__main__':
    main()
