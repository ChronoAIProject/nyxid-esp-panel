"""Authenticate GPU evidence and independently recount consumed saved score/UART data."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
import struct
import subprocess
import tarfile
import wave
import zlib

import numpy as np


ARTIFACTS = {
    'joe': ('gpu-joe-fixed-baseline-evidence-v1.tar.gz', 29839545,
        'e2361d3835a192ccbd25567fad33ede42a63126771306b3b5c7566f732fc4160',
        '90bca76', '4090-joe-fixed-baseline-evidence-manifest-v1.json', 2526),
    'background': ('gpu-new-background-fixed-baseline-evidence-v1.tar.gz', 229537,
        'cf298670f15c77f4b005e9dd3391d5050e08b2c4937eafd2d87574be6ec794f8',
        '0b8286b', '4090-new-background-fixed-baseline-evidence-manifest-v1.json', 256),
    'joe-float': ('gpu-joe-v30-float-diagnostic-evidence-v1.tar.gz', 1255888,
        '9605f66fc5b3a535123e05b82bb12e6513aa4e705e9cf4496cac1ac1da75ee9d',
        '73f4562', '4090-joe-v30-float-diagnostic-manifest-v1.json', 7),
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def historical(commit, name):
    return subprocess.check_output(['git', 'show', commit + ':research/wakeword_p4/results/' + name])


def authenticate(root, label):
    name, size, digest, commit, manifest_name, count = ARTIFACTS[label]
    directory = root / ('mac-' + label + '-score-receipt-v1')
    archive = directory / name
    assert archive.stat().st_size == size and sha(archive) == digest
    with tarfile.open(archive) as tar:
        members = tar.getmembers()
        names = [m.name for m in members]
        assert len(names) == len(set(names)) == count + 1
        assert all(m.isfile() and not Path(m.name).is_absolute()
                   and '..' not in Path(m.name).parts and Path(m.name).as_posix() == m.name for m in members)
        manifest_blob = tar.extractfile('manifest.json').read()
        assert manifest_blob == historical(commit, manifest_name)
        manifest = json.loads(manifest_blob)
        assert len(manifest['files']) == count
        assert set(names) == {'manifest.json'} | {f['path'] for f in manifest['files']}
        for entry in manifest['files']:
            blob = tar.extractfile(entry['path']).read()
            assert len(blob) == entry['size'] and hashlib.sha256(blob).hexdigest() == entry['sha256']
            assert blob == (directory / 'extracted' / entry['path']).read_bytes()
    return directory / 'extracted', dict(archive=name, sha256=digest, bytes=size,
        payloads_authenticated=count, manifest_commit=commit)


def events(means):
    armed, last, found = True, -1000, []
    for i, mean in enumerate(means):
        if i - last >= 67:
            if mean < .91796875 / 2:
                armed = True
            if armed and mean >= .91796875:
                found.append((3 * (i + 4) + 5) * 160 / 16000)
                armed, last = False, i
    return found


def same_times(left, right):
    assert len(left) == len(right)
    assert all(abs(x - y) < 1e-12 for x, y in zip(left, right))


def official(directory, index, row, freeze, raw=None):
    report_path = directory / 'replay' / str(index) / 'report.json'
    report = json.loads(report_path.read_text())
    assert report['record'] == row
    manifest = freeze.get('manifest', freeze['files']['development_manifest'])
    assert report['manifest_sha256'] == manifest['sha256']
    assert report['source_sha256'] == freeze['files']['runner']['sha256']
    for name in ('qemu', 'elf', 'flash'):
        assert report[name + '_sha256'] == freeze['files'][name]['sha256']
    assert report['input_samples'] == row['samples']
    assert report['pcm_sha256'] == row['pcm_sha256']
    assert report['calls'] == (row['samples'] + 511) // 512
    assert report['final_chunk_zero_pad_samples'] == (-row['samples']) % 512
    assert report['threshold_requested'] == .63 and report['threshold_getter_after_set'] == .62496001
    assert report['firmware_stage'] == 104 and report['firmware_crc32_matched'] is True
    serial_path = report_path.with_name('serial.log')
    assert sha(serial_path) == report['serial_sha256']
    text = serial_path.read_text()
    params = re.findall(r'NYX_PCM_PARAMETERS chunk=(\d+) rate=(\d+) channels=(\d+) set_result=(\d+) actual=([^\s]+)', text)
    assert params == [('512', '16000', '1', '1', '0.62496001')]
    done = re.findall(r'NYX_PCM_DONE samples=(\d+) calls=(\d+) events=(\d+) verified=(\d+) crc32=([0-9a-f]+)', text)
    assert len(done) == 1
    samples, calls, count, verified, crc = done[0]
    assert (int(samples), int(calls), int(verified), crc) == (
        row['samples'], report['calls'], report['channel_verified_returns'], report['pcm_crc32'])
    parsed = [dict(call=int(c), sample_end=int(s), result=int(r), audio_seconds=int(s) / 16000)
              for c, s, r in re.findall(r'NYX_PCM_EVENT call=(\d+) sample_end=(\d+) result=(\d+)', text)]
    assert parsed == report['events'] and int(count) == len(parsed)
    assert all(e['sample_end'] == e['call'] * 512 and e['result'] > 0 for e in parsed)
    assert ': 0x00000068' in report_path.with_name('stage-monitor.txt').read_text()
    if raw is not None:
        assert len(raw) == row['samples'] * 2
        assert hashlib.sha256(raw).hexdigest() == row['pcm_sha256']
        assert f'{zlib.crc32(raw):08x}' == crc
        assert report_path.with_name('input.pcm').read_bytes() == struct.pack('<IIfI', 0x4e595852, row['samples'], .63, 0) + raw
    return [e['audio_seconds'] for e in parsed], sha(report_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'render', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    archives = {}
    directories = {}
    for label in ARTIFACTS:
        directories[label], archives[label] = authenticate(args.root, label)
    render = json.loads((args.render / 'replay-manifest.json').read_text())
    assert sha(args.render / 'replay-manifest.json') == 'edf6f7844f654d2a7e22f588873b917381aa8d0957b1e3309a60a0f1973895bc'
    results, common_texts = {}, Counter()
    reports = {}
    for label, intended in (('joe', 360), ('background', 42)):
        directory = directories[label]
        report = json.loads((directory / 'replay/report.json').read_text())
        freeze = json.loads((directory / 'replay/freeze.json').read_text())
        assert sha(directory / 'replay/freeze.json') == report['execution_pins_sha256']
        assert freeze['v30_threshold'] == .91796875 and freeze['smoothing'] == 5
        rows = report['records']
        assert len(rows) == intended
        if label == 'joe':
            assert [r['record'] for r in rows] == render['records']
        else:
            manifest_blob = historical('0b8286b', '4090-new-background-fixed-baseline-replay-manifest-v1.json')
            assert hashlib.sha256(manifest_blob).hexdigest() == freeze['manifest']['sha256']
            assert [r['record'] for r in rows] == json.loads(manifest_blob)['records']
        events_by_engine = {'v30': [], 'hiesp': []}
        for i, item in enumerate(rows):
            row = item['record']
            raw = None
            if label == 'joe':
                path = args.render / row['path']
                assert sha(path) == row['sha256']
                with wave.open(str(path)) as stream:
                    assert (stream.getframerate(), stream.getnchannels(), stream.getsampwidth(), stream.getnframes()) == (16000, 1, 2, row['samples'])
                    raw = stream.readframes(stream.getnframes())
            esp_times, report_hash = official(directory, i, row, freeze, raw)
            assert report_hash == item['hiesp']['report_sha256']
            same_times(esp_times, item['hiesp']['event_seconds'])
            path = directory / 'replay' / f'{i}-v30-scores.npy'
            assert sha(path) == item['v30']['score_sha256']
            scores = np.load(path, allow_pickle=False)
            assert scores.shape == (((row['samples'] + 159) // 160 - 2) // 3 - 4,)
            assert np.isfinite(scores).all() and np.all((0 <= scores) & (scores <= 1))
            nyx_times = events(scores)
            same_times(nyx_times, item['v30']['event_seconds'])
            for engine, times in (('v30', nyx_times), ('hiesp', esp_times)):
                events_by_engine[engine].append(times)
                if label == 'joe':
                    same_times([t for t in times if t >= 1.8], item[engine]['post_source_events'])
                    for field, end in (('all_event_signed_generated_end_ms', row['generated_output_end_sample']),
                                       ('all_event_signed_full_rir_end_ms', row['full_rir_end_sample'])):
                        expected = [(t - end / 16000) * 1000 for t in times]
                        same_times(expected, item[engine][field])
            if label == 'joe' and row['target'] == 'confusion' and esp_times:
                common_texts[row['text']] += 1
        counts = {}
        for engine in ('v30', 'hiesp'):
            times = events_by_engine[engine]
            if label == 'background':
                samples = sum(r['record']['samples'] for r in rows)
                assert samples == report['original_samples'] == 455204864
                hours = samples / 16000 / 3600
                count = sum(map(len, times))
                assert count == 0
                upper = -math.log(.05) / hours
                expected = report['summaries'][engine]
                assert (expected['events'], expected['original_hours']) == (count, hours)
                assert abs(expected['upper_one_sided95_per_hour'] - upper) < 1e-15
                counts[engine] = dict(events=count, samples=samples, hours=hours, upper95_per_hour=upper)
            else:
                target = 'hey_nyx' if engine == 'v30' else 'hi_esp'
                positive = [i for i, r in enumerate(rows) if r['record']['target'] == target]
                confuser = [i for i, r in enumerate(rows) if r['record']['target'] == 'confusion']
                cross = [i for i, r in enumerate(rows) if r['record']['target'] not in (target, 'confusion')]
                activated = sum(any(t >= 1.8 for t in times[i]) for i in positive)
                standalone = [i for i in positive if ', ' not in rows[i]['record']['text']]
                command = [i for i in positive if ', ' in rows[i]['record']['text']]
                counts[engine] = dict(positive_clips=len(positive), positive_activated=activated,
                    standalone_clips=len(standalone), standalone_activated=sum(bool(times[i]) for i in standalone),
                    command_clips=len(command), command_activated=sum(bool(times[i]) for i in command),
                    common_clips=len(confuser), common_activated=sum(bool(times[i]) for i in confuser),
                    common_events=sum(len(times[i]) for i in confuser), cross_clips=len(cross),
                    cross_activated=sum(bool(times[i]) for i in cross),
                    pre_source_positive_events=sum(sum(t < 1.8 for t in times[i]) for i in positive))
                summary = report['summaries'][engine]
                assert (activated, counts[engine]['common_activated'], counts[engine]['cross_activated']) == (
                    summary['post_source_activated'], summary['common']['activated_clips'], summary['cross_target']['activated_clips'])
                assert counts[engine]['standalone_activated'] / len(standalone) == summary['standalone_rate']
                assert counts[engine]['command_activated'] / len(command) == summary['command_rate']
        results[label] = counts
        reports[label] = report
    directory = directories['joe-float'] / 'diagnostic'
    report = json.loads((directory / 'report.json').read_text())
    records = [json.loads(x) for x in (directory / 'records.jsonl').read_text().splitlines()]
    assert len(records) == 360 and [r['record'] for r in records] == render['records']
    traces = np.load(directory / 'scores.npz', allow_pickle=False)
    assert len(traces.files) == 1800
    floats = {arm: {} for arm in ('float_original', 'float_input_rounded', 'int8')}
    for i, row in enumerate(records):
        for arm, raw_name in (('float_original', 'float_raw'), ('float_input_rounded', 'float_rounded_raw'), ('int8', None)):
            values = traces[f'{arm}_smoothed_{i}']
            assert np.isfinite(values).all() and np.all((0 <= values) & (values <= 1))
            if raw_name:
                raw = traces[f'{raw_name}_{i}']
                assert len(raw) == row['float_raw_count']
                means = [math.fsum(float(x) for x in raw[k:k + 5]) / 5 for k in range(len(raw) - 4)]
                np.testing.assert_allclose(values, means, rtol=0, atol=4e-16)
            else:
                np.testing.assert_array_equal(values, np.load(directories['joe'] / 'replay' / f'{i}-v30-scores.npy', allow_pickle=False))
            assert hashlib.sha256(values.tobytes()).hexdigest() == row['arms'][arm]['score_sha256']
            assert max(values) == row['arms'][arm]['peak_five_mean']
            same_times(events(values), row['arms'][arm]['event_seconds'])
            assert not events(values)
            target = row['record']['target']
            item = floats[arm].setdefault(target, dict(clips=0, activated=0, maximum_five_mean=0.))
            item['clips'] += 1
            item['maximum_five_mean'] = max(item['maximum_five_mean'], float(max(values)))
    for arm in floats:
        for target, item in floats[arm].items():
            summary = report['summaries'][arm][target]
            assert (item['clips'], item['activated'], item['maximum_five_mean']) == (
                summary['clips'], summary['any_event_clips'], summary['maximum_five_mean'])
    result = dict(schema='mac-independent-joe-background-saved-score-audit-v1',
        source_sha256=sha(Path(__file__)), archives=archives,
        joe_full_input_pcm_bytes_independently_match_all360_frozen_render_wavs=True,
        all402_uart_events_done_crc_report_fields_stage_getter_and_trace_events_match=True,
        joe_counts=results['joe'], official_common_activation_texts=dict(common_texts),
        background=results['background'], joe_float=floats,
        independent_model_inference=False, qemu_invoked=False, training=False, accepted=False,
        limits='Authenticated saved evidence and independent accounting only. No Mac QEMU/float/INT8 inference '
            'or engine binary authentication. Background omitted input.pcm/original WAV bytes were not '
            'downloaded, decoded or independently authenticated; their exposure and CRC compare saved '
            'manifest/UART/report fields only. Joe full input bytes/CRC were independently recomputed. '
            'Reset-repeat claims were not rerun. All audio is consumed development; no independent '
            'acceptance, human phonetic/hardware latency or model promotion.')
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
