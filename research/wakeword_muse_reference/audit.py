"""Extract sanitized Muse frames and independently audit the saved zero run.

Uses no detector implementation, serial port, model, microphone or speaker.
"""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import zlib


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def extract(args):
    frames = []
    for line in args.uart_private.read_text().splitlines():
        for kind in ('READY', 'CASE', 'DONE'):
            marker = 'MUSE_REFERENCE_' + kind + ' '
            at = line.find(marker)
            if at >= 0:
                frames.append(dict(kind=kind, value=json.loads(line[at + len(marker):])))
                break
    with args.output.open('x') as output:
        output.write(json.dumps(frames, indent=2) + '\n')
    print(json.dumps(dict(frames=len(frames), sha256=sha(args.output))))


def audit(args):
    frames = json.loads(args.frames.read_text())
    report = json.loads(args.report.read_text())
    build = json.loads(args.build_report.read_text())
    protocol = json.loads(args.protocol.read_text())
    assert sha(args.build_report) == report['build_report_sha256'] == protocol['build_report_sha256']
    assert sha(args.protocol) == report['protocol_sha256']
    assert report['probe_sha256'] == build['firmware_sha256'] == protocol['firmware_sha256']
    assert build['schema'] == 'official-muse-zero-build-v3'
    assert protocol['schema'] == 'official-muse-zero-plan-v3'
    assert build['preparation_target'] == protocol['preparation_target'] == 'esp32p4'
    assert build['components']['espressif/esp-dl']['version'] == '3.3.14'
    assert build['dl_model_abi_bytes'] == 144 and build['dl_model_abi_compile_assertion']
    assert set(build['model_members']) == {'wn10_data', '_MODEL_INFO_'}
    assert report['schema'] == 'muse-zero-guarded-native-v3'
    assert [f['kind'] for f in frames] == ['READY', *(['CASE'] * 5), 'DONE']
    assert frames[0]['value'] == report['firmware_ready']
    assert [f['value'] for f in frames[1:-1]] == report['cases']
    assert frames[-1]['value'] == report['done']
    ready, done = frames[0]['value'], frames[-1]['value']
    assert ready['schema'] == done['schema'] == 'muse-zero-v3'
    assert ready['engine'] == 'wn10_himuse' and ready['mode'] == 'DET_MODE_95'
    assert ready['expanded_model_bytes'] == build['expanded_model_bytes'] == protocol['expanded_model_bytes']
    assert done['cases'] == 5
    chunk = report['cases'][0]['chunk']
    assert type(chunk) is int and 2 <= chunk <= 4096
    lengths = (1, chunk - 1, chunk + 1, 16000, 16001)
    words = report['cases'][0]['words']
    assert 1 <= len(words) <= 8
    for index, word in enumerate(words, 1):
        assert word['index'] == index and word['name']
        assert word['default_bits'] == word['after_prime_bits']
        value = struct.unpack('!f', bytes.fromhex(word['default_bits']))[0]
        assert 0 < value < 1
    input_calls = prime_events = input_events = 0
    for index, case in enumerate(report['cases']):
        samples = lengths[index]
        calls, remainder = divmod(samples, chunk)
        calls += int(remainder != 0)
        assert (case['id'], case['samples'], case['chunk'], case['rate'], case['channels']) == (index, samples, chunk, 16000, 1)
        assert case['calls'] == calls and len(case['input_returns']) == calls
        crc = 0
        for _ in range(samples):
            crc = zlib.crc32(b'\0\0', crc)
        assert case['crc32'] == crc
        assert case['words'] == words and len(case['prime_returns']) == 63
        for values in (case['input_returns'], case['prime_returns']):
            assert all(type(v) is int and -1 <= v <= len(words) for v in values)
        assert 0 <= case['detect_max_us'] <= case['detect_total_us']
        assert case['create_us'] >= 0 and case['prime_us'] >= 0
        assert case['probe_buffer_bytes'] == 2 * chunk + 4 * calls
        for name in ('heap_before_create', 'heap_after_create', 'heap_after_prime', 'heap_after_destroy'):
            assert set(case[name]) == {'internal_free', 'psram_free', 'internal_largest', 'psram_largest'}
            assert all(type(n) is int and n >= 0 for n in case[name].values())
        input_calls += calls
        prime_events += sum(v > 0 for v in case['prime_returns'])
        input_events += sum(v > 0 for v in case['input_returns'])
    summary = report['summary']
    for name, actual in dict(chunk=chunk, cases=5, rate=16000, channels=1,
            input_calls=input_calls, prime_calls=315, prime_positive_events=prime_events,
            input_positive_events=input_events, word_metadata=words).items():
        assert summary[name] == actual, name
    assert summary['zero_positive_events_absent'] is (prime_events == input_events == 0)
    result = dict(schema='muse-zero-saved-frame-audit-v3',
        frames_sha256=sha(args.frames), report_sha256=sha(args.report),
        protocol_sha256=sha(args.protocol), build_report_sha256=sha(args.build_report),
        frames=len(frames), cases=5, input_calls=input_calls, prime_calls=315,
        prime_positive_events=prime_events, input_positive_events=input_events,
        complete_frame_order_sample_call_crc_raw_returns_getters_verified=True,
        recognition_accepted=False, acceptance_accessed=False,
        scope='Independent saved-frame accounting; no second inference, speech/FAR, acoustic or voice acceptance.')
    with args.output.open('x') as output:
        output.write(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    prep = commands.add_parser('extract')
    prep.add_argument('--uart-private', type=Path, required=True)
    prep.add_argument('--output', type=Path, required=True)
    check = commands.add_parser('audit')
    for name in ('frames', 'report', 'protocol', 'build-report', 'output'):
        check.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    extract(args) if args.command == 'extract' else audit(args)
