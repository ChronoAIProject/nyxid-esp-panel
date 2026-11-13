"""Prepare or audit zero-PCM P4 initialization checks, never operate a device.

Run the existing guarded wakeword_baseline/run.py separately. No speech, TTS,
microphone, speaker, recognition comparison or reserved acceptance data is used.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import wave
import zlib


SAMPLES = (16000, 16001, 48000, 48001, 16000)
REPO = Path(__file__).resolve().parents[3]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(output):
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    for index, count in enumerate(SAMPLES):
        path = output / f'zero-{index:02d}.wav'
        with wave.open(str(path), 'wb') as wav:
            wav.setparams((1, 2, 16000, count, 'NONE', 'not compressed'))
            wav.writeframes(bytes(count * 2))
        rows.append(dict(path=path.name, sha256=sha(path), samples=count,
            target='diagnostic_silence', scope='initialization_not_recognition'))
    (output / 'manifest.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows))
    (output / 'protocol.json').write_text(json.dumps(dict(
        schema='official-hey-zero-smoke-v1', sample_counts=SAMPLES, rate=16000,
        channels=1, pcm_bits=16, source='exact integer zeros',
        transfer_baud=115200, mode='DET_MODE_95',
        engines=['wn9_hiesp', 'wn9_heywillow_tts'], source_sha256=sha(Path(__file__)),
        scope='Repeated create/prime/detect/destroy and CRC/partial-chunk checks only.',
        no_recognition_or_far_estimate=True, no_audio_io=True), indent=2) + '\n')


def audit(args):
    report = json.loads(args.report.read_text())
    build = json.loads(args.build_report.read_text())
    manifest = args.fixtures / 'manifest.jsonl'
    protocol = json.loads((args.fixtures / 'protocol.json').read_text())
    assert tuple(protocol['sample_counts']) == SAMPLES and protocol['source_sha256'] == sha(Path(__file__))
    rows = list(map(json.loads, manifest.read_text().splitlines()))
    assert len(rows) == len(report['clips']) == len(SAMPLES)
    assert report['manifest_sha256'] == sha(manifest)
    assert report['engine'] == args.engine and report['mode'] == 'DET_MODE_95'
    assert report['sample_rate'] == report['firmware_ready']['rate'] == 16000
    assert report['uart_baud'] == 115200 and report['firmware_ready']['resource_probe'] is True
    assert report['firmware_ready']['engine'] == args.engine
    assert report['probe_sha256'] == build['products'][args.engine]['files']['nyx_wake_baseline.bin']['sha256']
    assert report['restored']['app_sha256'] == args.expected_restored_elf
    assert report['restored']['wake_listening'] is True
    assert report['before']['app_sha256'] == report['restored']['app_sha256']
    assert report['before']['slot'] == report['restored']['slot']
    assert report['restored']['voice_idle'] and report['restored']['service_selection_unchanged']
    clips = []
    for index, (row, recorded, samples) in enumerate(zip(rows, report['clips'], SAMPLES)):
        path = args.fixtures / row['path']
        assert sha(path) == row['sha256']
        assert row['samples'] == samples
        assert all(recorded[key] == value for key, value in row.items())
        with wave.open(str(path), 'rb') as wav:
            assert (wav.getnchannels(), wav.getsampwidth(), wav.getframerate(), wav.getnframes()) == (1, 2, 16000, samples)
            pcm = wav.readframes(samples)
        assert pcm == bytes(2 * samples)
        result = recorded['result']
        prepared = recorded['initialization']
        chunk = prepared['chunk']
        assert 0 < chunk <= 2048 and prepared['id'] == index
        assert result['id'] == index and result['complete'] is True
        assert result['samples'] == samples and result['crc32'] == zlib.crc32(pcm)
        assert result['calls'] == math.ceil(samples / chunk)
        assert result['events'] == 0 and result['offsets'] == []
        assert result['total_us'] > 0 and result['max_us'] > 0
        for phase in ('heap_before_create', 'heap_after_create', 'heap_after_prime'):
            assert set(prepared[phase]) == {'internal_free', 'psram_free', 'internal_largest', 'psram_largest'}
            assert all(isinstance(value, int) and value >= 0 for value in prepared[phase].values())
        clips.append(dict(index=index, samples=samples, pcm_sha256=hashlib.sha256(pcm).hexdigest(),
                          initialization=prepared, result=result))
    calls = sum(clip['result']['calls'] for clip in clips)
    result = dict(schema='official-hey-zero-smoke-result-v1', engine=args.engine,
        report_sha256=sha(args.report), build_report_sha256=sha(args.build_report),
        manifest_sha256=sha(manifest), protocol_sha256=sha(args.fixtures / 'protocol.json'),
        probe_sha256=report['probe_sha256'], firmware_ready=report['firmware_ready'], clips=clips,
        create_us_median=statistics.median(clip['initialization']['create_us'] for clip in clips),
        zero_prime_us_median=statistics.median(clip['initialization']['prime_us'] for clip in clips),
        zero_detect_calls=calls,
        zero_detect_mean_us=sum(clip['result']['total_us'] for clip in clips) / calls,
        zero_detect_max_us=max(clip['result']['max_us'] for clip in clips),
        before=report['before'], restored=report['restored'], restoration_readiness=report['restore_readiness'],
        complete_zero_transfers_and_initializations=True, recognition_accepted=False,
        candidate_scored=False, reserved_acceptance_accessed=False, acoustic_audio_io=False,
        scope='Native P4 zero-PCM initialization/transfer/resources only; not speech compute, recognition, false-wake-rate or whole-app acceptance.')
    assert not args.output.exists()
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({key: result[key] for key in ('engine', 'create_us_median', 'zero_detect_calls',
        'zero_detect_mean_us', 'complete_zero_transfers_and_initializations', 'restored')}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    prep = commands.add_parser('prepare')
    prep.add_argument('--output', type=Path, required=True)
    check = commands.add_parser('audit')
    check.add_argument('--fixtures', type=Path, required=True)
    check.add_argument('--report', type=Path, required=True)
    check.add_argument('--build-report', type=Path, required=True)
    check.add_argument('--expected-restored-elf', required=True)
    check.add_argument('--engine', choices=('wn9_hiesp', 'wn9_heywillow_tts'), required=True)
    check.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'prepare':
        prepare(args.output)
    else:
        audit(args)


if __name__ == '__main__':
    main()
