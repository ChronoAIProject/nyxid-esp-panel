"""Prepare/audit a frozen native >64-event developer transport diagnostic.

Reuses a previously consumed synthetic file; no TTS, playback, microphone,
reserved material or device access. Replay uses the guarded runner separately.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import wave
import zlib


REPO = Path(__file__).resolve().parents[3]
SOURCE_MANIFEST = 'cb536c6e9c2176c51a88d3af6a1867ad00602053b0b4d1f788294acf8cbd8b2e'
REPEATS = 80


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def pcm(path):
    with wave.open(str(path), 'rb') as wav:
        assert (wav.getnchannels(), wav.getsampwidth(), wav.getframerate(), wav.getcomptype()) == (1, 2, 16000, 'NONE')
        return wav.readframes(wav.getnframes())


def prepare(args):
    manifest = args.source_fixtures / 'manifest.jsonl'
    assert sha(manifest) == SOURCE_MANIFEST
    source = [r for r in map(json.loads, manifest.read_text().splitlines())
              if r['voice'] == 'Samantha' and r['target'] == 'hey_willow' and r['form'] == 'bare']
    assert len(source) == 1
    row = source[0]; path = args.source_fixtures / row['path']
    assert sha(path) == row['sha256']
    raw = pcm(path); assert len(raw) == row['samples'] * 2
    assert hashlib.sha256(raw).hexdigest() == row['pcm_sha256']
    args.output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(path, args.output / 'single.wav')
    with wave.open(str(args.output / 'repeated-80.wav'), 'wb') as wav:
        wav.setparams((1, 2, 16000, row['samples'] * REPEATS, 'NONE', 'not compressed'))
        for _ in range(REPEATS): wav.writeframes(raw)
    rows = [dict(path=name, sha256=sha(args.output / name), samples=row['samples'] * repeats,
        target='hey_willow', source_role='consumed synthetic development transport', repeats=repeats,
        reference_threshold_f32_hex=None) for name, repeats in [('single.wav', 1), ('repeated-80.wav', REPEATS)]]
    (args.output / 'manifest.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    protocol = dict(schema='official-hey-event-stream-v1', source_manifest_sha256=SOURCE_MANIFEST,
        source_record=row, repeats=REPEATS, complete_source_pcm_repeated_verbatim=True,
        source_sha256=sha(Path(__file__)), runner_sha256=sha(REPO / 'research/wakeword_baseline/run.py'),
        manifest_sha256=sha(args.output / 'manifest.jsonl'), engine='wn9_heywillow_tts',
        mode='DET_MODE_95', threshold='DEFAULT without setter, both streams', uart_baud=921600,
        lifecycle='Original per-stream create/63-zero-prime/full-PCM/destroy; long stream has no intermediate reset.',
        criterion='Both complete transfers and all raw returns/events authenticated. Native >64 positive frames required to claim that emission branch exercised; preserve lower count as incomplete coverage, never regenerate/extend/tune to rescue.',
        scope='Consumed synthetic positive-event transport diagnostic. Inserted source silence repeats too; not continuous natural background, recognition/FAR/true wake latency or acceptance.',
        speaker_or_microphone_used=False, reserved_acceptance_accessed=False, candidate_scored=False)
    (args.output / 'protocol.json').write_text(json.dumps(protocol, indent=2) + '\n')
    print(json.dumps(dict(source_samples=row['samples'], long_samples=row['samples'] * REPEATS,
        long_seconds=row['samples'] * REPEATS / 16000, manifest_sha256=protocol['manifest_sha256'])))


def audit(args):
    protocol = json.loads((args.fixtures / 'protocol.json').read_text())
    assert protocol['source_sha256'] == sha(Path(__file__))
    assert protocol['runner_sha256'] == sha(REPO / 'research/wakeword_baseline/run.py')
    assert protocol['source_manifest_sha256'] == SOURCE_MANIFEST and protocol['repeats'] == REPEATS
    manifest = args.fixtures / 'manifest.jsonl'; assert sha(manifest) == protocol['manifest_sha256']
    rows = list(map(json.loads, manifest.read_text().splitlines()))
    assert len(rows) == 2 and [r['repeats'] for r in rows] == [1, REPEATS]
    single = pcm(args.fixtures / 'single.wav')
    assert hashlib.sha256(single).hexdigest() == protocol['source_record']['pcm_sha256']
    assert pcm(args.fixtures / 'repeated-80.wav') == single * REPEATS
    runner_path = REPO / 'research/wakeword_baseline/run.py'
    spec = importlib.util.spec_from_file_location('event_stream_reference_runner', runner_path)
    runner = importlib.util.module_from_spec(spec); spec.loader.exec_module(runner)
    report = json.loads(args.report.read_text()); build = json.loads(args.build_report.read_text())
    assert report['engine'] == protocol['engine'] and report['uart_baud'] == protocol['uart_baud']
    assert report['control_protocol'] == runner.CONTROL_PROTOCOL
    assert report['manifest_sha256'] == sha(manifest) and len(report['clips']) == 2
    assert report['probe_sha256'] == build['products'][protocol['engine']]['files']['nyx_wake_baseline.bin']['sha256']
    runner.validate_ready(report['firmware_ready'], protocol['engine'], True, True)
    assert report['before']['app_sha256'] == report['restored']['app_sha256'] == args.expected_restored_elf
    assert report['before']['slot'] == report['restored']['slot']
    assert report['restored']['voice_idle'] and report['restored']['wake_listening'] and report['restored']['service_selection_unchanged']
    for index, (row, actual) in enumerate(zip(rows, report['clips'])):
        assert sha(args.fixtures / row['path']) == row['sha256']
        assert all(actual[key] == value for key, value in row.items())
        assert row['reference_threshold_f32_hex'] is None and len(pcm(args.fixtures / row['path'])) == row['samples'] * 2
        runner.validate_control_initialization(actual['initialization'], row, index)
        runner.validate_control_result(actual['result'], actual['raw_events'], row, index, 512)
        assert actual['result']['crc32'] == zlib.crc32(pcm(args.fixtures / row['path']))
    long = report['clips'][1]
    result = dict(schema='official-hey-event-stream-native-result-v1', protocol=protocol,
        protocol_sha256=sha(args.fixtures / 'protocol.json'), manifest_sha256=sha(manifest),
        original_report_sha256=sha(args.report), build_report_sha256=sha(args.build_report),
        probe_sha256=report['probe_sha256'], clips=report['clips'], before=report['before'],
        restored=report['restored'], restore_readiness=report['restore_readiness'],
        native_over_64_positive_events_verified=len(long['raw_events']) > 64,
        long_raw_event_count=len(long['raw_events']), long_prefix_count=len(long['result']['offsets']),
        long_prefix_complete=long['result']['offsets_complete'], all_transfers_and_raw_accounting_verified=True,
        recognition_accepted=False, equal_far_comparison=False, candidate_promoted=False,
        reserved_acceptance_accessed=False, speaker_or_microphone_used=False, scope=protocol['scope'])
    assert not args.output.exists(); args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({key: result[key] for key in ('native_over_64_positive_events_verified',
        'long_raw_event_count', 'long_prefix_count', 'long_prefix_complete', 'all_transfers_and_raw_accounting_verified')}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    prep = commands.add_parser('prepare')
    prep.add_argument('--source-fixtures', type=Path, required=True)
    prep.add_argument('--output', type=Path, required=True)
    check = commands.add_parser('audit')
    for name in ('fixtures', 'report', 'build-report', 'output'):
        check.add_argument('--' + name, type=Path, required=True)
    check.add_argument('--expected-restored-elf', required=True)
    args = parser.parse_args()
    prepare(args) if args.command == 'prepare' else audit(args)


if __name__ == '__main__': main()
