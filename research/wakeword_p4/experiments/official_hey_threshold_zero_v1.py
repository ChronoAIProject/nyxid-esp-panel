"""Freeze, host-check or audit threshold API zero-PCM probes; no device I/O.

Hardware replay uses the guarded baseline runner separately. This checks setter,
getter, reset, priming and complete-event transport semantics, never recognition.
"""
import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import wave
import zlib


REPO = Path(__file__).resolve().parents[3]
REQUESTS = (None, .4, .5, .6, .644, .7, .8, .9, .95, .98, .99, .999, .9999, None)
SAMPLES = 16001


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_runner():
    path = REPO / 'research/wakeword_baseline/run.py'
    spec = importlib.util.spec_from_file_location('threshold_reference_runner', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def rows(wav_hash):
    return [dict(path='zero-16001.wav', sha256=wav_hash, samples=SAMPLES,
        target='diagnostic_silence', point=index, requested_decimal=request,
        reference_threshold_f32_hex=None if request is None else struct.pack('>f', request).hex())
        for index, request in enumerate(REQUESTS)]


def prepare(output):
    output.mkdir(parents=True, exist_ok=False)
    path = output / 'zero-16001.wav'
    with wave.open(str(path), 'wb') as wav:
        wav.setparams((1, 2, 16000, SAMPLES, 'NONE', 'not compressed'))
        wav.writeframes(bytes(SAMPLES * 2))
    manifest = output / 'manifest.jsonl'
    manifest.write_text(''.join(json.dumps(row) + '\n' for row in rows(sha(path))))
    source_paths = ['research/wakeword_baseline/main/baseline.c',
        'research/wakeword_baseline/main/CMakeLists.txt', 'research/wakeword_baseline/run.py',
        str(Path(__file__).relative_to(REPO))]
    protocol = dict(schema='official-hey-threshold-zero-v1', requests=REQUESTS,
        sources={name: sha(REPO / name) for name in source_paths},
        control_protocol='official-threshold-v1', sample_rate=16000, samples_per_stream=SAMPLES,
        manifest_sha256=sha(manifest), transfer_baud=115200,
        engines_in_order=['wn9_heywillow_tts', 'wn9_hiesp'], mode='DET_MODE_95',
        model_bytes='Pinned unmodified ESP-SR 2.6.1',
        purpose='Default before/after, full agreed 13-configuration grid, raw prime and complete input accounting.',
        failure_policy='Preserve failures and every actual getter, including merged points; no retries or recognition threshold selection.',
        no_recognition_scoring=True, no_speech_or_acoustic_io=True, reserved_acceptance_accessed=False)
    (output / 'protocol.json').write_text(json.dumps(protocol, indent=2) + '\n')
    print(json.dumps(dict(manifest_sha256=sha(manifest), protocol_sha256=sha(output / 'protocol.json'),
        streams_per_engine=len(REQUESTS), complete_zero_samples_per_stream=SAMPLES)))


def host_check(output):
    runner = load_runner()
    accepted, rejected = [], []
    def reject(label, action):
        try:
            action()
        except (ValueError, RuntimeError):
            rejected.append(label)
        else:
            raise AssertionError('Not rejected: ' + label)
    for row in rows('analytic'):
        command = runner.threshold_command(row, row['point'], True)
        value = runner.request_value(row[runner.CONTROL_FIELD])
        assert command.endswith(' DEFAULT\n') if value is None else command.endswith(' SET ' + row[runner.CONTROL_FIELD] + '\n')
    accepted.append('All 14 explicit DEFAULT/SET commands including binary32 endpoints')
    assert runner.threshold_command(dict(samples=1), 0, False) == 'CLIP 0 1\n'
    accepted.append('Original default wire command')
    reject('Request cannot silently enter legacy mode', lambda: runner.threshold_command(rows('x')[0], 0, False))
    reject('Control mode requires request even for DEFAULT', lambda: runner.threshold_command(dict(samples=1), 0, True))
    for bits in ('7fc00000', '7f800000', 'ff800000', '00000000', '3f800000', '3f00000', '3F000000', False):
        reject('Invalid request ' + str(bits), lambda bits=bits: runner.request_value(bits))
    old = dict(engine='wn9_hiesp', mode='DET_MODE_95', rate=16000)
    new = dict(old, control_protocol=runner.CONTROL_PROTOCOL)
    runner.validate_ready(old, 'wn9_hiesp')
    runner.validate_ready(new, 'wn9_hiesp', require_threshold_control=True)
    reject('Control firmware needs explicit mode', lambda: runner.validate_ready(new, 'wn9_hiesp'))
    reject('Control mode rejects legacy firmware', lambda: runner.validate_ready(old, 'wn9_hiesp', require_threshold_control=True))
    getter = struct.unpack('>f', struct.pack('>f', .644))[0]
    bits = struct.pack('>f', getter).hex()
    row = rows('x')[4]
    control = dict(schema=runner.CONTROL_PROTOCOL, setter_called=True, setter_result=1,
        requested_f32_hex=row[runner.CONTROL_FIELD], default_getter=getter, default_getter_f32_hex=bits,
        getter_after_set=getter, getter_after_set_f32_hex=bits,
        getter_after_prime_f32_hex=bits, prime_returns=[0] * 63)
    prepared = dict(id=4, chunk=512, threshold=getter, control=control)
    runner.validate_control_initialization(prepared, row, 4)
    accepted.append('Raw binary32 getter and all 63 prime returns')
    transformed = copy.deepcopy(prepared)
    transformed['control']['getter_after_set'] = struct.unpack('>f', struct.pack('>f', .3968))[0]
    transformed['control']['getter_after_set_f32_hex'] = struct.pack('>f', .3968).hex()
    runner.validate_control_initialization(transformed, row, 4)
    accepted.append('Observed getter is not forced into setter input range or equated to request')
    for label, key, value in [('Setter failure', 'setter_result', 0),
            ('Wrong request echo', 'requested_f32_hex', '3f000000'),
            ('Incomplete priming', 'prime_returns', [0] * 62),
            ('Unknown priming return', 'prime_returns', [2] * 63),
            ('Getter value/bit disagreement', 'getter_after_set', .5)]:
        changed = copy.deepcopy(prepared); changed['control'][key] = value
        reject(label, lambda changed=changed: runner.validate_control_initialization(changed, row, 4))
    samples = 70 * 512 - 7
    long_row = dict(samples=samples)
    events = [dict(id=0, call=call, sample_end=min(call * 512, samples), chunk_end=call * 512, result=1)
              for call in range(1, 71)]
    result = dict(id=0, complete=True, samples=samples, calls=70, events=70, events_emitted=70,
        offsets_complete=False, no_detect_calls=0, channel_verified_calls=0,
        offsets=[e['sample_end'] for e in events[:64]])
    runner.validate_control_result(result, events, long_row, 0, 512)
    accepted.append('70 raw events exceed legacy 64 prefix and preserve final partial-chunk clocks')
    reject('Dropped raw event', lambda: runner.validate_control_result(result, events[:-1], long_row, 0, 512))
    for label, key, value in [('Wrong consumed calls', 'calls', 69), ('Uncounted result', 'no_detect_calls', 1),
                             ('Incorrect truncation flag', 'offsets_complete', True)]:
        changed = dict(result, **{key: value})
        reject(label, lambda changed=changed: runner.validate_control_result(changed, events, long_row, 0, 512))
    for label, key, value in [('Foreign clip event', 'id', 1), ('Wrong sample clock', 'sample_end', 1),
                             ('Channel verification is not wake', 'result', -1), ('Duplicate call', 'call', 1)]:
        changed = copy.deepcopy(events); changed[1][key] = value
        reject(label, lambda changed=changed: runner.validate_control_result(result, changed, long_row, 0, 512))
    report = dict(schema='official-hey-threshold-host-check-v1',
        runner_sha256=sha(REPO / 'research/wakeword_baseline/run.py'), checker_sha256=sha(Path(__file__)),
        accepted=accepted, rejected=rejected, source_unchanged_by_check=True,
        scope='Analytic host protocol validation only. The 70 events are constructed, not model outputs or native coverage.',
        device_operated=False, model_inference=False, reserved_acceptance_accessed=False)
    assert not output.exists()
    output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


def audit(args):
    runner = load_runner()
    fixtures = args.fixtures
    protocol = json.loads((fixtures / 'protocol.json').read_text())
    for name, digest in protocol['sources'].items():
        assert sha(REPO / name) == digest
    manifest = fixtures / 'manifest.jsonl'
    assert protocol['manifest_sha256'] == sha(manifest)
    expected = rows(sha(fixtures / 'zero-16001.wav'))
    assert list(map(json.loads, manifest.read_text().splitlines())) == expected
    with wave.open(str(fixtures / 'zero-16001.wav'), 'rb') as wav:
        assert (wav.getnchannels(), wav.getsampwidth(), wav.getframerate(), wav.getnframes()) == (1, 2, 16000, SAMPLES)
        assert wav.readframes(SAMPLES) == bytes(SAMPLES * 2)
    build = json.loads(args.build_report.read_text())
    arms = {}
    for engine, path in zip(protocol['engines_in_order'], (args.willow_report, args.hiesp_report)):
        report = json.loads(path.read_text())
        assert report['control_protocol'] == runner.CONTROL_PROTOCOL and report['engine'] == engine
        assert report['manifest_sha256'] == sha(manifest) and len(report['clips']) == len(expected)
        assert report['uart_baud'] == 115200 and report['sample_rate'] == 16000
        assert report['probe_sha256'] == build['products'][engine]['files']['nyx_wake_baseline.bin']['sha256']
        assert '-DBASELINE_THRESHOLD_CONTROL=1' in build['products'][engine]['defines']
        runner.validate_ready(report['firmware_ready'], engine, True, True)
        assert report['before']['app_sha256'] == report['restored']['app_sha256'] == args.expected_restored_elf
        assert report['before']['slot'] == report['restored']['slot']
        assert report['restored']['voice_idle'] and report['restored']['wake_listening'] and report['restored']['service_selection_unchanged']
        points = []
        for index, (row, actual) in enumerate(zip(expected, report['clips'])):
            assert all(actual[key] == value for key, value in row.items())
            runner.validate_control_initialization(actual['initialization'], row, index)
            runner.validate_control_result(actual['result'], actual['raw_events'], row, index, 512)
            assert actual['result']['crc32'] == zlib.crc32(bytes(SAMPLES * 2))
            points.append(dict(point=index, requested_decimal=row['requested_decimal'],
                requested_f32_hex=row['reference_threshold_f32_hex'], initialization=actual['initialization'],
                result=actual['result'], raw_events=actual['raw_events']))
        before_default = points[0]['initialization']['control']['getter_after_prime_f32_hex']
        after_default = points[-1]['initialization']['control']['getter_after_prime_f32_hex']
        arms[engine] = dict(probe_sha256=report['probe_sha256'], original_report_sha256=sha(path),
            points=points, default_before_after_bits_equal=before_default == after_default,
            observed_distinct_post_prime_getters=len({p['initialization']['control']['getter_after_prime_f32_hex'] for p in points}),
            total_prime_events=sum(v > 0 for p in points for v in p['initialization']['control']['prime_returns']),
            total_input_events=sum(p['result']['events'] for p in points),
            restored=report['restored'], before=report['before'], restore_readiness=report['restore_readiness'])
    result = dict(schema='official-hey-threshold-native-zero-v1', protocol=protocol,
        protocol_sha256=sha(fixtures / 'protocol.json'), manifest_sha256=sha(manifest),
        build_report_sha256=sha(args.build_report), arms=arms, zero_streams_completed=28,
        setter_getter_and_raw_accounting_exercised=True, speech_inference=False,
        thresholds_calibrated=False, candidate_promoted=False, reserved_acceptance_accessed=False,
        scope='Native API behavior on exact zeros; includes every requested/actual/default value and raw priming returns. No speech/FAR/true latency/long positive-event or whole-app acceptance.')
    assert not args.output.exists()
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({engine: dict(points=len(arm['points']), defaults_reset=arm['default_before_after_bits_equal'],
        distinct_getters=arm['observed_distinct_post_prime_getters'], prime_events=arm['total_prime_events'],
        input_events=arm['total_input_events']) for engine, arm in arms.items()}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    for command in ('prepare', 'check-host'):
        p = commands.add_parser(command); p.add_argument('--output', type=Path, required=True)
    p = commands.add_parser('audit')
    for name in ('fixtures', 'build-report', 'willow-report', 'hiesp-report', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--expected-restored-elf', required=True)
    args = parser.parse_args()
    if args.command == 'prepare': prepare(args.output)
    elif args.command == 'check-host': host_check(args.output)
    else: audit(args)


if __name__ == '__main__':
    main()
