"""Prepare or audit the frozen native P4 speech diagnostic; never operate a board.

Preparation invokes macOS say with -o (file generation only). Hardware replay
uses the unchanged guarded baseline runner, separately. All waveforms stay out
of git; both voices are already consumed development sources, not acceptance.
"""
import argparse
import hashlib
import json
from math import ceil, gcd
from pathlib import Path
import random
import subprocess
import wave
import zlib


PROTOCOL = Path(__file__).with_suffix('.json')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path):
    return json.loads(path.read_text())


def records(plan):
    rows = [dict(voice=voice, target=phrase['target'], form=form, text=phrase[form])
            for voice in plan['voices'] for phrase in plan['phrases'] for form in plan['forms']]
    random.Random(plan['order_seed']).shuffle(rows)
    assert len(rows) == plan['records']
    return rows


def render(source, lead, tail):
    import numpy as np
    import soundfile as sf
    from scipy.signal import resample_poly
    data, rate = sf.read(source, dtype='int16')
    assert data.ndim == 1 and len(data) > 0 and sf.info(source).subtype == 'PCM_16'
    divisor = gcd(rate, 16000)
    converted = resample_poly(data.astype('float64') / 32768., 16000 // divisor, rate // divisor)
    pcm = np.rint(converted * 32768.).clip(-32768, 32767).astype('<i2').tobytes()
    return bytes(lead * 2) + pcm + bytes(tail * 2), dict(
        native_rate=rate, native_samples=len(data), resampled_samples=len(pcm) // 2,
        native_pcm_sha256=hashlib.sha256(data.astype('<i2').tobytes()).hexdigest())


def prepare(output):
    import numpy as np
    import scipy
    import soundfile as sf
    plan = read_json(PROTOCOL)
    output.mkdir(parents=True, exist_ok=False)
    # Freeze the precise plan and implementation before generating a single file.
    (output / 'protocol.json').write_bytes(PROTOCOL.read_bytes())
    (output / 'prepare-source.py').write_bytes(Path(__file__).read_bytes())
    installed = subprocess.check_output(['/usr/bin/say', '-v', '?'], text=True)
    assert all(any(line.startswith(voice + ' ') for line in installed.splitlines()) for voice in plan['voices'])
    environment = dict(say_sha256=sha(Path('/usr/bin/say')),
        macos=subprocess.check_output(['sw_vers'], text=True), numpy=np.__version__,
        scipy=scipy.__version__, soundfile=sf.__version__,
        installed_voice_inventory_sha256=hashlib.sha256(installed.encode()).hexdigest())
    rows = []
    for index, row in enumerate(records(plan)):
        stem = f'clip-{index:02d}'
        source = output / (stem + '.aiff')
        command = ['/usr/bin/say', '-v', row['voice'], '-o', str(source), row['text']]
        subprocess.run(command, check=True, capture_output=True, timeout=60)
        pcm, native = render(source, plan['render']['lead_samples'], plan['render']['tail_samples'])
        path = output / (stem + '.wav')
        with wave.open(str(path), 'wb') as wav:
            wav.setparams((1, 2, 16000, len(pcm) // 2, 'NONE', 'not compressed'))
            wav.writeframes(pcm)
        rows.append(dict(row, path=path.name, source_path=source.name,
            source_sha256=sha(source), sha256=sha(path), pcm_sha256=hashlib.sha256(pcm).hexdigest(),
            samples=len(pcm) // 2, native=native,
            role=plan['role'], synthesis_default_rate_preserved=True))
    manifest = output / 'manifest.jsonl'
    manifest.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    result = dict(schema=plan['schema'], protocol_sha256=sha(PROTOCOL),
        prepare_source_sha256=sha(Path(__file__)), manifest_sha256=sha(manifest),
        environment=environment, records=len(rows), model_inference_performed=False,
        speaker_or_microphone_used=False, reserved_acceptance_accessed=False)
    (output / 'generation.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


def audit(args):
    plan = read_json(PROTOCOL)
    fixtures = args.fixtures
    generation = read_json(fixtures / 'generation.json')
    manifest = fixtures / 'manifest.jsonl'
    assert generation['protocol_sha256'] == sha(PROTOCOL) == sha(fixtures / 'protocol.json')
    assert generation['prepare_source_sha256'] == sha(Path(__file__)) == sha(fixtures / 'prepare-source.py')
    assert generation['manifest_sha256'] == sha(manifest)
    rows = list(map(json.loads, manifest.read_text().splitlines()))
    assert len(rows) == plan['records']
    for index, (row, expected) in enumerate(zip(rows, records(plan))):
        assert all(row[key] == value for key, value in expected.items())
        assert row['path'] == f'clip-{index:02d}.wav' and row['source_path'] == f'clip-{index:02d}.aiff'
        source, path = fixtures / row['source_path'], fixtures / row['path']
        assert sha(source) == row['source_sha256'] and sha(path) == row['sha256']
        pcm, native = render(source, plan['render']['lead_samples'], plan['render']['tail_samples'])
        assert native == row['native']
        with wave.open(str(path), 'rb') as wav:
            assert (wav.getnchannels(), wav.getsampwidth(), wav.getframerate(), wav.getnframes()) == (1, 2, 16000, row['samples'])
            assert wav.readframes(row['samples']) == pcm
        assert hashlib.sha256(pcm).hexdigest() == row['pcm_sha256']
    build = read_json(args.build_report)
    arms = {}
    for engine, report_path in zip(plan['arm_order'], (args.willow_report, args.hiesp_report)):
        report = read_json(report_path)
        assert report['engine'] == engine and report['mode'] == plan['mode']
        assert report['sample_rate'] == 16000 and report['uart_baud'] == plan['transfer_baud']
        assert report['firmware_ready']['engine'] == engine and report['firmware_ready']['resource_probe']
        assert report['manifest_sha256'] == sha(manifest) and len(report['clips']) == len(rows)
        assert report['probe_sha256'] == build['products'][engine]['files']['nyx_wake_baseline.bin']['sha256']
        assert report['before']['app_sha256'] == report['restored']['app_sha256'] == args.expected_restored_elf
        assert report['before']['slot'] == report['restored']['slot']
        assert report['restored']['voice_idle'] and report['restored']['wake_listening']
        assert report['restored']['service_selection_unchanged']
        clips, counts = [], {}
        for index, (row, actual) in enumerate(zip(rows, report['clips'])):
            assert all(actual[key] == value for key, value in row.items())
            result, init = actual['result'], actual['initialization']
            chunk = init['chunk']
            assert chunk == 512 and init['id'] == result['id'] == index
            assert actual['threshold'] == init['threshold'] == plan['threshold_getter_expected'][engine]
            assert result['complete'] is True and result['samples'] == row['samples']
            pcm, _ = render(fixtures / row['source_path'], plan['render']['lead_samples'], plan['render']['tail_samples'])
            assert result['crc32'] == zlib.crc32(pcm) and result['calls'] == ceil(row['samples'] / chunk)
            offsets = result['offsets']
            assert len(offsets) == result['events'] < 64 and offsets == sorted(set(offsets))
            assert all(0 < at <= row['samples'] and (at % chunk == 0 or at == row['samples']) for at in offsets)
            early = sum(at <= plan['render']['lead_samples'] for at in offsets)
            targeted = row['target'] == plan['targets'][engine]
            key = row['target'] + '/' + row['form']
            count = counts.setdefault(key, dict(streams=0, streams_with_any_event=0,
                streams_with_post_lead_event=0, total_events=0, pre_generation_events=0, intended_target=targeted))
            count['streams'] += 1
            count['streams_with_any_event'] += bool(offsets)
            count['streams_with_post_lead_event'] += len(offsets) > early
            count['total_events'] += len(offsets)
            count['pre_generation_events'] += early
            clips.append(dict(actual, intended_target_for_engine=targeted, pre_generation_events=early))
        calls = sum(c['result']['calls'] for c in clips)
        arms[engine] = dict(report_sha256=sha(report_path), probe_sha256=report['probe_sha256'],
            firmware_ready=report['firmware_ready'], clips=clips, counts=counts,
            speech_container_detect_calls=calls,
            speech_container_detect_mean_us=sum(c['result']['total_us'] for c in clips) / calls,
            speech_container_detect_max_us=max(c['result']['max_us'] for c in clips),
            before=report['before'], restored=report['restored'], restore_readiness=report['restore_readiness'])
    result = dict(schema=plan['schema'] + '-result', protocol_sha256=sha(PROTOCOL),
        auditor_sha256=sha(Path(__file__)), build_report_sha256=sha(args.build_report),
        manifest_sha256=sha(manifest), generation=generation, records=rows, arms=arms,
        complete_native_transfers=True, recognition_accepted=False, equal_far_comparison=False,
        candidate_scored=False, reserved_acceptance_accessed=False, speaker_or_microphone_used=False,
        scope=plan['limitations'])
    assert not args.output.exists()
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({engine: arm['counts'] for engine, arm in arms.items()}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    prep = commands.add_parser('prepare')
    prep.add_argument('--output', type=Path, required=True)
    check = commands.add_parser('audit')
    for name in ('fixtures', 'build-report', 'willow-report', 'hiesp-report', 'output'):
        check.add_argument('--' + name, type=Path, required=True)
    check.add_argument('--expected-restored-elf', required=True)
    args = parser.parse_args()
    if args.command == 'prepare':
        prepare(args.output)
    else:
        audit(args)


if __name__ == '__main__':
    main()
