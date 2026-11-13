"""Authenticate original v30 and compare float/input-rounded/INT8 on consumed126."""
import argparse
from collections import Counter
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import soundfile as sf

TRAINING = Path(__file__).resolve().parents[1] / 'training'
PACKAGES = ('tensorflow', 'numpy', 'soundfile', 'ai-edge-litert', 'pymicro-features', 'h5py')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fingerprint(value):
    return str(value.dtype), tuple(value.shape), hashlib.sha256(value.tobytes()).hexdigest()


def events(scores, threshold=.91796875):
    means = np.convolve(scores, np.ones(5) / 5, 'valid')
    armed, last, found = True, -1000, []
    for index, value in enumerate(means):
        if value < threshold / 2 and index - last >= 67:
            armed = True
        if armed and value >= threshold and index - last >= 67:
            found.append((index + 6) * 480 - 56160)
            last, armed = index, False
    return found, means


def protocol(args, manifest):
    return dict(schema='bryce-consumed-original-v30-float-protocol-v1',
        source_sha256=sha(Path(__file__)), original_model_manifest_sha256=hashlib.sha256(manifest).hexdigest(),
        input_manifest_sha256=sha(args.data / 'manifest.json'),
        original_baseline_report_sha256=sha(args.saved_study / 'report.json'),
        original_baseline_scores_sha256=sha(args.saved_study / 'scores.npz'),
        helper_source_sha256={name: sha(TRAINING / name) for name in ('features.py', 'evaluate.py', 'detector_policy.py')},
        versions={p: importlib.metadata.version(p) for p in PACKAGES},
        model_sha256='04aa45646610f0e3a866a17d4098263adf5da912bb6ddfe03c44807a295baa0c',
        original_weights_sha256='b40bf705297a71f208e073897ff517e90f9813337e73a959e27627b2d2668fa9',
        selected_step=70000, clips=126, threshold=.91796875, smoothing=5, refractory_steps=67,
        prime_samples=56160, final_padding_multiple=480, model_cadence_samples=480,
        input_scale=.10196078568696976, input_zero_point=-128,
        arms=['original_float', 'same_float_input_quant_dequant', 'retained_original_int8'],
        float_reset='All loaded state variables zero before every stream; every float arm repeated exactly',
        float_input='clip(raw MicroFrontend /25.6,0,26); rounded arm uses original INT8 input scale and zero',
        timestamp='End of returning480sample block, original firmware policy; all prime/early/repeat events retained',
        raw_int8='Original saved126 raw traces; every new frontend SHA must equal original saved SHA',
        selection='No threshold, checkpoint, input subset, gain, crop or speed selection',
        independent_data=False, audio_playback_or_recording=False, training=False, accepted=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('authenticate', 'run'))
    for name in ('archive', 'model-dir', 'data', 'saved-study', 'protocol', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    assert args.archive.stat().st_size == 570686
    assert sha(args.archive) == '6f19e79f63aedde9e8afb0e3229e9f260179099cfaa27590745856233889f83b'
    with tarfile.open(args.archive) as archive:
        members = archive.getmembers()
        names = [m.name for m in members]
        assert len(names) == len(set(names)) == 16 and all(m.isfile() for m in members)
        for name in names:
            relative = Path(name)
            assert not relative.is_absolute() and '..' not in relative.parts and relative.as_posix() == name
        manifest_blob = archive.extractfile('manifest.json').read()
        assert manifest_blob == subprocess.check_output(['git', 'show',
            '76130cc:research/wakeword_p4/results/4090-v30-original-float-handoff-manifest-v1.json'])
        entries = json.loads(manifest_blob)['files']
        assert len(entries) == len({e['path'] for e in entries}) == 15
        assert set(names) == {e['path'] for e in entries} | {'manifest.json'}
        for entry in entries:
            blob = archive.extractfile(entry['path']).read()
            assert len(blob) == entry['size'] and hashlib.sha256(blob).hexdigest() == entry['sha256']
        if args.action == 'authenticate':
            assert not args.model_dir.exists() and not args.protocol.exists()
            args.model_dir.mkdir(parents=True)
            for member in members:
                destination = args.model_dir / member.name
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(archive.extractfile(member).read())
        for member in members:
            assert (args.model_dir / member.name).read_bytes() == archive.extractfile(member).read()
    assert sha(args.saved_study / 'report.json') == 'de27257b414ffa6eb90ce8aad5649310662e1db18bb93febfcf1f94c097184ee'
    assert sha(args.saved_study / 'scores.npz') == '61e2fbfb31fdec528d9f572b9302c6d75267a49be524689f20ba2038dcf87711'
    prior = json.loads((args.model_dir / 'historical_mac350_float_int8_report.json').read_text())
    historical_blob = subprocess.check_output(['git', 'show',
        'ee0becbf3d8716d14a977afbe52b2e5e5aa0ecee:research/wakeword_p4/results/4090-mac-float-int8-v1.json'])
    assert (args.model_dir / 'historical_mac350_float_int8_report.json').read_bytes() == historical_blob
    for relative, digest in prior['saved_model_files'].items():
        assert sha(args.model_dir / relative) == digest
    frozen = protocol(args, manifest_blob)
    if args.action == 'authenticate':
        args.protocol.write_text(json.dumps(frozen, indent=2) + '\n')
        print(json.dumps(dict(payloads_authenticated=15, historical_saved_model_hashes_exact=True,
            protocol_sha256=sha(args.protocol), model_inference=False)))
        return
    assert json.loads(args.protocol.read_text()) == frozen and not args.output.exists()
    baseline = json.loads((args.saved_study / 'report.json').read_text())
    saved = np.load(args.saved_study / 'scores.npz', allow_pickle=False)
    assert len(baseline['records']) == len(saved.files) == 126
    for record in baseline['records']:
        assert sha(args.data / record['record']['path']) == record['record']['sha256']
    import h5py
    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')
    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    loaded = tf.saved_model.load(str(args.model_dir / 'saved_model'))
    signature = loaded.signatures['serving_default']
    states = [v for v in loaded.variables if 'states' in v.name]
    variables = [v.numpy() for v in loaded.variables if 'states' not in v.name]
    assert states and len(variables) == 31
    hdf_arrays = []
    with h5py.File(args.model_dir / 'best.weights.h5') as checkpoint:
        assert int(checkpoint['optimizer/vars/0'][()]) == 70000
        def visit(name, value):
            if name.startswith('layers/') and isinstance(value, h5py.Dataset):
                hdf_arrays.append(value[()])
        checkpoint.visititems(visit)
    assert Counter(map(fingerprint, hdf_arrays)) == Counter(map(fingerprint, variables))
    sys.path.insert(0, str(TRAINING))
    from features import features
    args.output.mkdir(parents=True)
    arrays, rows = {}, []

    def predict(values):
        for state in states:
            state.assign(tf.zeros_like(state))
        values = np.asarray(values, np.float32)
        output = []
        for start in range(0, len(values) - 2, 3):
            result = signature(input_audio=tf.convert_to_tensor(values[None, start:start + 3]))
            output.append(float(result['output_0'].numpy()[0, 0]))
        scores = np.asarray(output, np.float32)
        assert np.isfinite(scores).all() and np.all((0 <= scores) & (scores <= 1))
        return scores

    with (args.output / 'records.jsonl').open('x') as log:
        for index, item in enumerate(baseline['records']):
            row = item['record']
            pcm, rate = sf.read(args.data / row['path'], dtype='int16')
            assert rate == 16000 and len(pcm) == row['samples']
            assert hashlib.sha256(pcm.astype('<i2').tobytes()).hexdigest() == row['pcm_sha256']
            raw = features(np.pad(pcm, (56160, (-len(pcm)) % 480)))
            assert hashlib.sha256(raw.tobytes()).hexdigest() == item['frontend_sha256']
            x = np.clip(raw.astype(np.float32) / 25.6, 0, 26)
            quantized = np.clip(np.rint(x / frozen['input_scale']) - 128, -128, 127).astype(np.int8)
            rounded = (quantized.astype(np.float32) + 128) * np.float32(frozen['input_scale'])
            original_float, rounded_float = predict(x), predict(rounded)
            np.testing.assert_array_equal(original_float, predict(x))
            np.testing.assert_array_equal(rounded_float, predict(rounded))
            int8 = saved[f'output_{index}'].astype(np.float32) / 256
            assert original_float.shape == rounded_float.shape == int8.shape == (item['output_count'],)
            assert events(int8)[0] == item['firmware_policy']['offsets']
            record = dict(record=row, native_phrase_id=item['native_phrase_id'],
                frontend_sha256=item['frontend_sha256'], all_float_reset_repeats_exact=True, arms={})
            source_steps = (np.arange(len(int8)) + 2) * 480 - 56160 > 0
            for name, values in (('original_float', original_float), ('same_float_input_quant_dequant', rounded_float), ('retained_original_int8', int8)):
                offsets, means = events(values)
                arrays[f'{name}_{index}'] = values
                end = row['generator_prefix_end_sample']
                record['arms'][name] = dict(offsets=offsets, original_threshold=.91796875,
                    any_source_event=any(0 < at <= len(pcm) + (-len(pcm)) % 480 for at in offsets),
                    reference_window_hit=any(end - 1600 <= at <= end + 12800 for at in offsets),
                    events_before_source_start=sum(at < row['lead_samples'] for at in offsets),
                    maximum_source_callback_raw_score=float(values[source_steps].max()),
                    maximum_five_mean=float(means.max()), raw_output_sha256=hashlib.sha256(values.tobytes()).hexdigest())
            rows.append(record)
            log.write(json.dumps(record) + '\n')
            log.flush()
            if (index + 1) % 21 == 0:
                print(json.dumps(dict(completed=index + 1, total=126)), flush=True)
    np.savez_compressed(args.output / 'scores.npz', **arrays)
    summaries = {}
    for arm in frozen['arms']:
        groups = {}
        for voice in ('kristin', 'norman'):
            for phrase in ('hey_nyx', 'hey_nyx_command'):
                group = [r['arms'][arm] for r in rows if r['record']['voice'] == voice and r['native_phrase_id'] == phrase]
                groups[voice + '/' + phrase] = dict(clips=len(group),
                    any_source_event_clips=sum(r['any_source_event'] for r in group),
                    reference_window_hit_clips=sum(r['reference_window_hit'] for r in group),
                    raw_peak_range=[min(r['maximum_source_callback_raw_score'] for r in group), max(r['maximum_source_callback_raw_score'] for r in group)])
        positives = [r['arms'][arm] for r in rows if r['record']['target'] == 'hey_nyx']
        confusers = [r['arms'][arm] for r in rows if r['record']['target'] not in ('hey_nyx', 'hi_esp')]
        summaries[arm] = dict(positive_any_source_clips=sum(r['any_source_event'] for r in positives),
            positive_reference_window_hits=sum(r['reference_window_hit'] for r in positives),
            common_confuser_clips=sum(bool(r['offsets']) for r in confusers), groups=groups)
    result = dict(schema='bryce-consumed-original-v30-float-diagnostic-v1', protocol_sha256=sha(args.protocol),
        records_sha256=sha(args.output / 'records.jsonl'), scores_sha256=sha(args.output / 'scores.npz'),
        original_saved_model_authenticated_to_historical_hashes=True, checkpoint_iteration=70000,
        all31_model_variables_match_original_checkpoint=True, all126_frontend_and_saved_int8_events_exact=True,
        all_float_streams_repeated_exact=True, summaries=summaries, threshold_selected=False,
        training=False, accepted=False, limits='Consumed synthetic development and frozen INT8 operating point only. '
        'Same-checkpoint float comparison, not an optimized float threshold, pronunciation truth or independent '
        'acceptance. Rounded-float isolates input rounding inside float graph; remaining INT8 differences include '
        'weight, activation, state and runtime effects. No retraining, export or firmware change.')
    (args.output / 'report.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
