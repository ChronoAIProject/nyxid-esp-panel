"""Freeze/check/audit opt-in native complete-stream capacity, never wake quality.

Two-hour input reuses consumed synthetic PCM twice, surrounded by exact zeros.
No new TTS, playback, microphone, background corpus or acceptance data is used.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import struct
import subprocess
import wave
import zlib


REPO = Path(__file__).resolve().parents[3]
SOURCE_MANIFEST = 'cb536c6e9c2176c51a88d3af6a1867ad00602053b0b4d1f788294acf8cbd8b2e'
MAX_SAMPLES = 16000 * 7200
ENGINE = 'wn9_heywillow_tts'


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def load_runner(path=None):
    path = path or REPO / 'research/wakeword_baseline/run.py'
    spec = importlib.util.spec_from_file_location('long_stream_runner_' + str(id(path)), path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def waveform_info(path):
    digest = hashlib.sha256(); crc = 0
    with wave.open(str(path), 'rb') as wav:
        assert (wav.getnchannels(), wav.getsampwidth(), wav.getframerate(), wav.getcomptype()) == (1, 2, 16000, 'NONE')
        count = wav.getnframes(); consumed = 0
        while block := wav.readframes(512 * 1024):
            consumed += len(block) // 2; digest.update(block); crc = zlib.crc32(block, crc)
        assert consumed == count
    return dict(samples=count, pcm_sha256=digest.hexdigest(), crc32=crc)


def host_check(args):
    args.output.mkdir(parents=True, exist_ok=False)
    old_path = args.output / 'original-runner.py'
    old_path.write_bytes(subprocess.check_output(['git', 'show', '00b36d4:research/wakeword_baseline/run.py'], cwd=REPO))
    old = load_runner(old_path); new = load_runner()
    accepted = []; rejected = []
    def reject(label, action):
        try:
            action()
        except (ValueError, RuntimeError):
            rejected.append(label)
        else:
            raise AssertionError('Not rejected: ' + label)
    for request in (None, .4, .5, .6, .644, .7, .8, .9, .95, .98, .99, .999, .9999):
        bits = None if request is None else struct.pack('>f', request).hex()
        for count in (1, 16001, 16000 * 3600):
            row = dict(samples=count, reference_threshold_f32_hex=bits)
            assert new.threshold_command(row, 7, True) == old.threshold_command(row, 7, True)
    assert new.threshold_command(dict(samples=1), 7, False) == old.threshold_command(dict(samples=1), 7, False)
    accepted.append('Original default/controlled command bytes unchanged at every agreed request and short boundary')
    for count in (16000 * 3600 + 1, 68499840, 6269 * 16000, MAX_SAMPLES):
        row = dict(samples=count, reference_threshold_f32_hex=None)
        assert new.threshold_command(row, 0, True, True) == f'CLIP 0 {count} DEFAULT\n'
        reject('Original profile rejects sample count ' + str(count), lambda row=row: new.threshold_command(row, 0, True))
    accepted.append('Explicit long profile admits actual locked lengths and inclusive two-hour limit')
    for count in (0, -1, True, 1.5, MAX_SAMPLES + 1):
        reject('Invalid long count ' + str(count), lambda count=count: new.threshold_command(dict(samples=count, reference_threshold_f32_hex=None), 0, True, True))
    reject('Long commands require threshold control', lambda: new.threshold_command(dict(samples=1), 0, False, True))
    for engine in ('wn9_hiesp', 'wn9_heywillow_tts'):
        base = dict(engine=engine, mode='DET_MODE_95', rate=16000, resource_probe=True)
        old.validate_ready(base, engine, True); new.validate_ready(base, engine, True)
        control = dict(base, control_protocol=new.CONTROL_PROTOCOL)
        old.validate_ready(control, engine, True, True); new.validate_ready(control, engine, True, True)
        long = dict(control, stream_protocol=new.LONG_STREAM_PROTOCOL, max_stream_samples=MAX_SAMPLES)
        new.validate_ready(long, engine, True, True, True)
        reject(engine + ' long firmware requires opt-in', lambda: new.validate_ready(long, engine, True, True))
        reject(engine + ' old firmware cannot claim long support', lambda: new.validate_ready(control, engine, True, True, True))
        reject(engine + ' long handshake requires threshold control', lambda: new.validate_ready(long, engine, True, False, True))
        reject(engine + ' wrong negotiated limit', lambda: new.validate_ready(dict(long, max_stream_samples=MAX_SAMPLES-1), engine, True, True, True))
    accepted.append('Both engines retain original handshakes and require exact explicit long protocol/limit')
    samples = MAX_SAMPLES - 1; calls = (samples + 511) // 512
    events = [dict(id=0, call=c, sample_end=min(c*512, samples), chunk_end=c*512, result=1) for c in (78, calls)]
    result = dict(id=0, complete=True, samples=samples, calls=calls, events=2, events_emitted=2,
                  offsets_complete=True, no_detect_calls=calls-2, channel_verified_calls=0,
                  offsets=[e['sample_end'] for e in events])
    new.validate_control_result(result, events, dict(samples=samples), 0, 512)
    accepted.append('Constructed full-limit event/call clock with final partial chunk')
    report = dict(schema='official-long-stream-host-check-v1', old_commit='00b36d4',
        runner_sha256=sha(REPO / 'research/wakeword_baseline/run.py'), helper_sha256=sha(Path(__file__)),
        accepted=accepted, rejected=rejected, native_inference=False, device_operated=False,
        scope='Host protocol and constructed clocks only, not actual long transfer or recognition.')
    (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


def prepare(args):
    manifest = args.source_fixtures / 'manifest.jsonl'
    assert sha(manifest) == SOURCE_MANIFEST
    source = next(r for r in map(json.loads, manifest.read_text().splitlines())
                  if r['voice'] == 'Samantha' and r['target'] == 'hey_willow' and r['form'] == 'bare')
    original = args.source_fixtures / source['path']
    assert sha(original) == source['sha256']
    assert waveform_info(original)['pcm_sha256'] == source['pcm_sha256']
    with wave.open(str(original), 'rb') as wav:
        marker = wav.readframes(wav.getnframes())
    args.output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(manifest, args.output / 'source-manifest.jsonl')
    shutil.copyfile(original, args.output / 'marker.wav')
    with wave.open(str(args.output / 'partial-zero.wav'), 'wb') as wav:
        wav.setparams((1, 2, 16000, 16001, 'NONE', 'not compressed'))
        wav.writeframes(bytes(16001 * 2))
    with wave.open(str(args.output / 'two-hours.wav'), 'wb') as wav:
        wav.setparams((1, 2, 16000, MAX_SAMPLES, 'NONE', 'not compressed'))
        wav.writeframes(marker)
        remaining = MAX_SAMPLES * 2 - 2 * len(marker)
        while remaining:
            size = min(remaining, 1024 * 1024); wav.writeframes(bytes(size)); remaining -= size
        wav.writeframes(marker)
    rows = [dict(path=name, sha256=sha(args.output / name), **waveform_info(args.output / name),
                 reference_threshold_f32_hex=None, target='consumed_willow_transport',
                 diagnostic_role=role) for name, role in [('partial-zero.wav', 'short partial-chunk control'),
                                                        ('two-hours.wav', 'inclusive two-hour capacity and late consumed-speech marker')]]
    target = args.output / 'manifest.jsonl'
    target.write_text(''.join(json.dumps(r) + '\n' for r in rows))
    build = json.loads(args.build_report.read_text())
    assert '-DBASELINE_LONG_STREAMS=1' in build['products'][ENGINE]['defines']
    sources = ['research/wakeword_baseline/main/CMakeLists.txt', 'research/wakeword_baseline/main/baseline.c',
               'research/wakeword_baseline/run.py', str(Path(__file__).relative_to(REPO))]
    protocol = dict(schema='official-long-stream-native-plan-v1', engine=ENGINE, mode='DET_MODE_95',
        sources={p: sha(REPO / p) for p in sources}, manifest_sha256=sha(target),
        source_manifest_sha256=SOURCE_MANIFEST, marker_source=source,
        marker_start_samples=[0, MAX_SAMPLES-source['samples']], max_stream_samples=MAX_SAMPLES,
        build_report_sha256=sha(args.build_report), probe=build['products'][ENGINE]['files']['nyx_wake_baseline.bin'],
        uart_baud=1500000, sample_rate=16000, control_protocol='official-threshold-v1', stream_protocol='official-long-stream-v1',
        host_completion_policy='board-result-ack-no-tcdrain-v1',
        lifecycle='One create/63-zero-prime/full input/final-only padding/destroy per stream. No intermediate reset.',
        decision='Retain failure and every event. Full two-hour transfer/accounting proves capacity; late positive-event coverage reported separately, not rescued by changing input/grid.',
        scope='Complete-stream capacity and detector state/clock witnesses on exact zeros plus two verbatim consumed synthetic containers. Not natural background/FAR/recognition/real wake latency/voice acceptance.',
        reserved_acceptance_accessed=False, new_synthesis=False, speaker_or_microphone_used=False)
    (args.output / 'protocol.json').write_text(json.dumps(protocol, indent=2) + '\n')
    print(json.dumps(dict(manifest_sha256=sha(target), protocol_sha256=sha(args.output / 'protocol.json'), streams=2, max_samples=MAX_SAMPLES)))


def audit(args):
    runner = load_runner(); protocol = json.loads((args.fixtures / 'protocol.json').read_text())
    for path, digest in protocol['sources'].items():
        assert sha(REPO / path) == digest
    assert sha(args.build_report) == protocol['build_report_sha256']
    assert sha(args.fixtures / 'source-manifest.jsonl') == SOURCE_MANIFEST
    assert sha(args.fixtures / 'marker.wav') == protocol['marker_source']['sha256']
    with wave.open(str(args.fixtures / 'marker.wav'), 'rb') as wav:
        marker = wav.readframes(wav.getnframes())
    with wave.open(str(args.fixtures / 'two-hours.wav'), 'rb') as wav:
        assert wav.getnframes() == MAX_SAMPLES
        assert wav.readframes(len(marker)//2) == marker
        remaining = MAX_SAMPLES-2*(len(marker)//2)
        while remaining:
            count = min(remaining, 512*1024)
            assert wav.readframes(count) == bytes(count*2); remaining -= count
        assert wav.readframes(len(marker)//2) == marker
    manifest = args.fixtures / 'manifest.jsonl'
    assert sha(manifest) == protocol['manifest_sha256']
    rows = list(map(json.loads, manifest.read_text().splitlines()))
    assert [r['samples'] for r in rows] == [16001, MAX_SAMPLES]
    report = json.loads(args.report.read_text())
    assert len(report['clips']) == 2 and report['manifest_sha256'] == sha(manifest)
    assert report['probe_sha256'] == protocol['probe']['sha256']
    assert report['engine'] == ENGINE and report['control_protocol'] == protocol['control_protocol']
    assert report['uart_baud'] == protocol['uart_baud'] and report['sample_rate'] == 16000
    assert report['stream_protocol'] == protocol['stream_protocol'] and report['max_stream_samples'] == MAX_SAMPLES
    assert report['host_completion_policy'] == protocol['host_completion_policy'] == 'board-result-ack-no-tcdrain-v1'
    runner.validate_ready(report['firmware_ready'], ENGINE, True, True, True)
    for index, (row, actual) in enumerate(zip(rows, report['clips'], strict=True)):
        assert all(actual[k] == v for k, v in row.items())
        path = args.fixtures / row['path']; assert sha(path) == row['sha256']
        info = waveform_info(path)
        assert all(info[k] == row[k] for k in ('samples', 'pcm_sha256', 'crc32'))
        runner.validate_control_initialization(actual['initialization'], row, index)
        runner.validate_control_result(actual['result'], actual['raw_events'], row, index, 512)
        assert actual['result']['crc32'] == row['crc32']
    assert report['before']['app_sha256'] == report['restored']['app_sha256'] == args.expected_restored_elf
    assert report['before']['slot'] == report['restored']['slot']
    assert report['restored']['voice_idle'] and report['restored']['wake_listening'] and report['restored']['service_selection_unchanged']
    long = report['clips'][1]
    result = dict(schema='official-long-stream-native-result-v1', protocol=protocol,
        protocol_sha256=sha(args.fixtures / 'protocol.json'), original_report_sha256=sha(args.report),
        complete_two_hour_capacity_verified=True, long_samples=MAX_SAMPLES,
        long_input_calls=long['result']['calls'],
        positive_event_after_original_one_hour_limit=any(e['sample_end'] > 16000*3600 for e in long['raw_events']),
        clips=report['clips'], before=report['before'], restored=report['restored'],
        restore_readiness=report['restore_readiness'], recognition_accepted=False, threshold_selected=False,
        equal_far_comparison=False, candidate_promoted=False, reserved_acceptance_accessed=False,
        speaker_or_microphone_used=False, scope=protocol['scope'])
    with args.output.open('x') as output:
        output.write(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: result[k] for k in ('complete_two_hour_capacity_verified', 'long_samples',
        'long_input_calls', 'positive_event_after_original_one_hour_limit')}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    host = commands.add_parser('check-host'); host.add_argument('--output', type=Path, required=True)
    prep = commands.add_parser('prepare')
    for name in ('source-fixtures', 'build-report', 'output'):
        prep.add_argument('--' + name, type=Path, required=True)
    check = commands.add_parser('audit')
    for name in ('fixtures', 'report', 'build-report', 'output'):
        check.add_argument('--' + name, type=Path, required=True)
    check.add_argument('--expected-restored-elf', required=True)
    args = parser.parse_args()
    {'check-host': host_check, 'prepare': prepare, 'audit': audit}[args.command](args)
