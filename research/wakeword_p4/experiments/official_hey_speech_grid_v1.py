"""Freeze/audit the agreed official grid on 12 already consumed synthetic clips.

This supplies same-PCM native runtime witnesses, not fresh development,
threshold selection, FAR, new-model quality or independent acceptance.
No device, speaker, microphone or TTS access occurs in this helper.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import struct
import wave
import zlib


REPO = Path(__file__).resolve().parents[3]
SOURCE_MANIFEST = 'cb536c6e9c2176c51a88d3af6a1867ad00602053b0b4d1f788294acf8cbd8b2e'
REQUESTS = (None, .4, .5, .6, .644, .7, .8, .9, .95, .98, .99, .999, .9999)
ENGINE = 'wn9_heywillow_tts'
ZERO = 'research/wakeword_p4/results/official-hey-threshold-native-zero-v1.json'
PILOT = 'research/wakeword_p4/results/official-hey-native-speech-pilot-v1.json'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pcm(path):
    with wave.open(str(path), 'rb') as wav:
        assert (wav.getnchannels(), wav.getsampwidth(), wav.getframerate(), wav.getcomptype()) == (1, 2, 16000, 'NONE')
        count = wav.getnframes()
        data = wav.readframes(count)
        assert len(data) == count * 2
        return data


def expanded_rows(sources):
    return [dict(source, source_index=index, point=point, requested_decimal=request,
                 reference_threshold_f32_hex=None if request is None else struct.pack('>f', request).hex())
            for point, request in enumerate(REQUESTS) for index, source in enumerate(sources)]


def prepare(args):
    source_manifest = args.source_fixtures / 'manifest.jsonl'
    assert sha(source_manifest) == SOURCE_MANIFEST
    sources = list(map(json.loads, source_manifest.read_text().splitlines()))
    assert len(sources) == 12 and len({r['path'] for r in sources}) == 12
    args.output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(source_manifest, args.output / 'source-manifest.jsonl')
    for row in sources:
        path = args.source_fixtures / row['path']
        assert sha(path) == row['sha256']
        data = pcm(path)
        assert len(data) == row['samples'] * 2
        assert hashlib.sha256(data).hexdigest() == row['pcm_sha256']
        shutil.copyfile(path, args.output / row['path'])
    manifest = args.output / 'manifest.jsonl'
    manifest.write_text(''.join(json.dumps(r) + '\n' for r in expanded_rows(sources)))
    zero = json.loads((REPO / ZERO).read_text())['arms'][ENGINE]
    points = zero['points'][:len(REQUESTS)]
    assert tuple(p['requested_decimal'] for p in points) == REQUESTS
    build = json.loads(args.build_report.read_text())
    assert '-DBASELINE_THRESHOLD_CONTROL=1' in build['products'][ENGINE]['defines']
    paths = ['research/wakeword_baseline/main/baseline.c',
             'research/wakeword_baseline/main/CMakeLists.txt',
             'research/wakeword_baseline/run.py', str(Path(__file__).relative_to(REPO)), ZERO, PILOT]
    protocol = dict(schema='official-hey-consumed-speech-grid-v1',
        source_manifest_sha256=SOURCE_MANIFEST, manifest_sha256=sha(manifest),
        sources={name: sha(REPO / name) for name in paths},
        build_report_sha256=sha(args.build_report),
        probe=build['products'][ENGINE]['files']['nyx_wake_baseline.bin'],
        engine=ENGINE, mode='DET_MODE_95', model='Unmodified pinned ESP-SR 2.6.1',
        requests=REQUESTS, points=len(REQUESTS), streams_per_point=12, streams=156,
        expected_getters_from_native_zero=[p['initialization']['control']['getter_after_prime_f32_hex'] for p in points],
        sample_rate=16000, uart_baud=921600,
        ordering='Point-major locked grid, then original 12-source manifest order.',
        lifecycle='Per-stream create / explicit DEFAULT or SET / all 63 zero-prime returns / complete PCM / final-only partial512 padding / destroy.',
        event_policy='All raw positive events; preserve 0 and -1 counts separately. No custom smoothing/refractory/rearm.',
        waveform_policy='Copy original 12 WAV files verbatim; no new synthesis, resampling, gain, trim or selection.',
        decision_policy='Authenticate runtime/getters/all events and retain every point. No operating-point choice or recognition pass/fail; preserve failures without retry/extension.',
        scope='Consumed synthetic same-PCM reference-runtime witnesses. No background exposure, FAR, phonetic/real wake latency, parity or independent acceptance.',
        reserved_acceptance_accessed=False, candidate_scored=False, speaker_or_microphone_used=False)
    (args.output / 'protocol.json').write_text(json.dumps(protocol, indent=2) + '\n')
    print(json.dumps(dict(streams=156, manifest_sha256=sha(manifest),
        protocol_sha256=sha(args.output / 'protocol.json'),
        total_pcm_samples=sum(r['samples'] for r in expanded_rows(sources)))))


def load_runner():
    path = REPO / 'research/wakeword_baseline/run.py'
    spec = importlib.util.spec_from_file_location('speech_grid_runner', path)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    return runner


def audit(args):
    runner = load_runner()
    protocol = json.loads((args.fixtures / 'protocol.json').read_text())
    assert protocol['schema'] == 'official-hey-consumed-speech-grid-v1'
    assert tuple(protocol['requests']) == REQUESTS
    for name, digest in protocol['sources'].items():
        assert sha(REPO / name) == digest
    source_manifest = args.fixtures / 'source-manifest.jsonl'
    assert sha(source_manifest) == SOURCE_MANIFEST == protocol['source_manifest_sha256']
    sources = list(map(json.loads, source_manifest.read_text().splitlines()))
    rows = expanded_rows(sources)
    manifest = args.fixtures / 'manifest.jsonl'
    assert sha(manifest) == protocol['manifest_sha256']
    assert list(map(json.loads, manifest.read_text().splitlines())) == rows and len(rows) == 156
    assert sha(args.build_report) == protocol['build_report_sha256']
    report = json.loads(args.report.read_text())
    assert report['probe_sha256'] == protocol['probe']['sha256']
    assert report['manifest_sha256'] == sha(manifest) and len(report['clips']) == len(rows)
    assert report['engine'] == ENGINE and report['control_protocol'] == runner.CONTROL_PROTOCOL
    assert report['uart_baud'] == 921600 and report['sample_rate'] == 16000
    runner.validate_ready(report['firmware_ready'], ENGINE, True, True)
    assert report['before']['app_sha256'] == report['restored']['app_sha256'] == args.expected_restored_elf
    assert report['before']['slot'] == report['restored']['slot']
    assert report['restored']['voice_idle'] and report['restored']['wake_listening'] and report['restored']['service_selection_unchanged']
    pilot = json.loads((REPO / PILOT).read_text())['arms'][ENGINE]['clips']
    points = []
    for point, request in enumerate(REQUESTS):
        clips = report['clips'][point * 12:(point + 1) * 12]
        for offset, actual in enumerate(clips):
            index = point * 12 + offset
            row = rows[index]
            assert all(actual[key] == value for key, value in row.items())
            path = args.fixtures / row['path']; data = pcm(path)
            assert sha(path) == row['sha256'] and len(data) == row['samples'] * 2
            assert hashlib.sha256(data).hexdigest() == row['pcm_sha256']
            runner.validate_control_initialization(actual['initialization'], row, index)
            runner.validate_control_result(actual['result'], actual['raw_events'], row, index, 512)
            assert actual['result']['crc32'] == zlib.crc32(data)
        getters = {c['initialization']['control']['getter_after_prime_f32_hex'] for c in clips}
        assert getters == {protocol['expected_getters_from_native_zero'][point]}
        target = [c for c in clips if c['target'] == 'hey_willow']
        cross = [c for c in clips if c['target'] != 'hey_willow']
        points.append(dict(point=point, requested_decimal=request,
            requested_f32_hex=clips[0]['reference_threshold_f32_hex'], actual_getter=clips[0]['threshold'],
            actual_getter_f32_hex=next(iter(getters)),
            intended_target_streams=len(target), target_streams_with_any_event=sum(bool(c['raw_events']) for c in target),
            cross_streams=len(cross), cross_streams_with_any_event=sum(bool(c['raw_events']) for c in cross),
            prime_positive_events=sum(v == 1 for c in clips for v in c['initialization']['control']['prime_returns']),
            input_positive_events=sum(len(c['raw_events']) for c in clips),
            clips=clips))
    defaults_match = all(c['path'] == p['path'] and c['result']['offsets'] == p['offsets']
                         for c, p in zip(points[0]['clips'], pilot, strict=True))
    result = dict(schema='official-hey-consumed-speech-grid-native-v1', protocol=protocol,
        protocol_sha256=sha(args.fixtures / 'protocol.json'), original_report_sha256=sha(args.report),
        points=points, all_156_transfers_and_raw_accounting_verified=True,
        all_actual_getters_match_native_zero_certificate=True,
        default_offsets_equal_retained_native_speech_pilot=defaults_match,
        before=report['before'], restored=report['restored'], restore_readiness=report['restore_readiness'],
        recognition_accepted=False, threshold_selected=False, equal_far_comparison=False,
        candidate_promoted=False, reserved_acceptance_accessed=False, speaker_or_microphone_used=False,
        scope=protocol['scope'])
    with args.output.open('x') as output:
        output.write(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(points=[{k: v for k, v in p.items() if k != 'clips'} for p in points],
        default_offsets_equal_retained_native_speech_pilot=defaults_match), indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    prep = commands.add_parser('prepare')
    prep.add_argument('--source-fixtures', type=Path, required=True)
    prep.add_argument('--build-report', type=Path, required=True)
    prep.add_argument('--output', type=Path, required=True)
    check = commands.add_parser('audit')
    for name in ('fixtures', 'report', 'build-report', 'output'):
        check.add_argument('--' + name, type=Path, required=True)
    check.add_argument('--expected-restored-elf', required=True)
    args = parser.parse_args()
    prepare(args) if args.command == 'prepare' else audit(args)


if __name__ == '__main__':
    main()
