"""Compare saved detector UART frames, whole PCM and the frozen long-stream plan.

No detector execution, serial, microphone, speaker or reserved-data access.
Use audit_official_hey_grid_frames_v1.py extract to export detector frames only.
"""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import wave
import zlib


MAX_SAMPLES = 16000 * 7200
DEFAULT_GETTER_BITS = '3f24dd2f'  # Retained native DEFAULT Willow certificate.


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def audit(args):
    frames = json.loads(args.frames.read_text())
    report = json.loads(args.report.read_text())
    protocol_path = args.fixtures / 'protocol.json'
    protocol = json.loads(protocol_path.read_text())
    manifest = args.fixtures / 'manifest.jsonl'
    rows = list(map(json.loads, manifest.read_text().splitlines()))
    assert sha(manifest) == protocol['manifest_sha256'] == report['manifest_sha256']
    assert [r['samples'] for r in rows] == [16001, MAX_SAMPLES]
    assert len(report['clips']) == len(rows) == 2
    assert report['probe_sha256'] == protocol['probe']['sha256']
    assert report['engine'] == protocol['engine'] == 'wn9_heywillow_tts'
    assert report['uart_baud'] == protocol['uart_baud'] == 1500000
    assert report['sample_rate'] == protocol['sample_rate'] == 16000
    assert report['control_protocol'] == protocol['control_protocol'] == 'official-threshold-v1'
    assert report['stream_protocol'] == protocol['stream_protocol'] == 'official-long-stream-v1'
    assert report['max_stream_samples'] == protocol['max_stream_samples'] == MAX_SAMPLES
    ready = [f['value'] for f in frames if f['kind'] == 'READY']
    assert len(ready) == 2 and ready[0] == ready[1] == report['firmware_ready']
    assert ready[0]['engine'] == report['engine'] and ready[0]['mode'] == 'DET_MODE_95'
    assert ready[0]['rate'] == 16000 and ready[0]['resource_probe'] is True
    for key in ('control_protocol', 'stream_protocol', 'max_stream_samples'):
        assert ready[0][key] == report[key]
    stream = [f for f in frames if f['kind'] != 'READY']
    cursor = calls = positives = prime_returns = 0
    for index, (row, actual) in enumerate(zip(rows, report['clips'], strict=True)):
        assert all(actual[k] == v for k, v in row.items())
        assert stream[cursor] == dict(kind='CLIP_READY', value=actual['initialization'])
        cursor += 1
        events = []
        while cursor < len(stream) and stream[cursor]['kind'] == 'EVENT':
            events.append(stream[cursor]['value']); cursor += 1
        result = actual['result']
        assert stream[cursor] == dict(kind='RESULT', value=result)
        cursor += 1
        assert events == actual['raw_events']
        init = actual['initialization']; control = init['control']
        assert init['id'] == result['id'] == index and init['chunk'] == 512
        assert row['reference_threshold_f32_hex'] is control['requested_f32_hex'] is None
        assert control['schema'] == 'official-threshold-v1'
        assert control['setter_called'] is False and control['setter_result'] is None
        for number, bits in ((control['default_getter'], control['default_getter_f32_hex']),
                             (control['getter_after_set'], control['getter_after_set_f32_hex']),
                             (init['threshold'], control['getter_after_prime_f32_hex'])):
            assert bits == DEFAULT_GETTER_BITS and struct.pack('>f', number).hex() == bits
        assert len(control['prime_returns']) == 63
        assert all(type(v) is int and v in (-1, 0, 1) for v in control['prime_returns'])
        path = args.fixtures / row['path']
        assert sha(path) == row['sha256']
        digest = hashlib.sha256(); crc = consumed = 0
        with wave.open(str(path), 'rb') as wav:
            assert (wav.getframerate(), wav.getnchannels(), wav.getsampwidth(), wav.getcomptype()) == (16000, 1, 2, 'NONE')
            assert wav.getnframes() == row['samples']
            while block := wav.readframes(512 * 1024):
                digest.update(block); crc = zlib.crc32(block, crc); consumed += len(block) // 2
        assert consumed == row['samples'] and digest.hexdigest() == row['pcm_sha256']
        expected_calls = (consumed + 511) // 512
        assert result['complete'] is True and result['samples'] == consumed
        assert result['calls'] == expected_calls and result['crc32'] == row['crc32'] == crc
        assert result['events'] == result['events_emitted'] == len(events)
        assert result['no_detect_calls'] + result['channel_verified_calls'] + len(events) == expected_calls
        assert min(result['no_detect_calls'], result['channel_verified_calls']) >= 0
        assert result['offsets_complete'] is (len(events) <= 64)
        assert result['offsets'] == [e['sample_end'] for e in events[:64]]
        previous = 0
        for event in events:
            assert event['id'] == index and event['result'] == 1
            assert type(event['call']) is int and previous < event['call'] <= expected_calls
            assert event['chunk_end'] == event['call'] * 512
            assert event['sample_end'] == min(event['chunk_end'], consumed)
            previous = event['call']
        calls += expected_calls; positives += len(events); prime_returns += 63
    assert cursor == len(stream)
    assert report['before']['app_sha256'] == report['restored']['app_sha256'] == args.expected_restored_elf
    assert report['before']['slot'] == report['restored']['slot']
    assert all(report['restored'][k] is True for k in ('voice_idle', 'wake_listening', 'service_selection_unchanged'))
    result = dict(schema='official-long-stream-saved-frame-audit-v1',
        frames_sha256=sha(args.frames), original_report_sha256=sha(args.report),
        protocol_sha256=sha(protocol_path), streams=2, input_calls=calls,
        positive_event_frames=positives, prime_returns=prime_returns,
        long_stream_samples=MAX_SAMPLES,
        late_positive_after_original_one_hour_cap=any(e['sample_end'] > 16000 * 3600 for e in report['clips'][1]['raw_events']),
        all_saved_frames_and_complete_PCM_CRC_samples_calls_getters_verified=True,
        scope='Independent saved-UART/whole-fixture comparison, not second inference, negative FAR exposure, recognition, parity or voice acceptance.')
    with args.output.open('x') as output:
        output.write(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('frames', 'report', 'fixtures', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--expected-restored-elf', required=True)
    audit(parser.parse_args())
