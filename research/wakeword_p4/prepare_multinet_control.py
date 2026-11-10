#!/usr/bin/env python3
"""Prepare silent MultiNet known-command controls using fixed training voices.

Synthesis always writes WAV files; it never opens a playback device. No source
audio from validation/final partitions is read. Recognition stays unaccepted.
"""
import argparse
from array import array
import csv
import hashlib
import json
import math
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import wave
import zlib

from prepare_multinet_probe import packed_model, sha


def grammar_ids(text, grammar, stock):
    normalized = text.upper().rstrip('.!')
    if grammar['stock_grammar']:
        return sorted({int(row[0]) for row in stock if row[1] == normalized})
    return [n + 1 for n, command in enumerate(grammar['commands']) if command == normalized]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--source-metadata', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    if args.output.exists():
        parser.error('Use a fresh private output directory')
    if (sha(Path(__file__)) != plan['preparation_script_sha256']
            or sha(args.source_metadata) != plan['source_metadata_sha256']
            or sha(Path(plan['say_binary'])) != plan['say_binary_sha256']):
        raise ValueError('Preparation, source metadata or synthesis binary identity differs')
    template = Path(__file__).with_name('multinet_probe')
    for name, expected in plan['template_files'].items():
        if sha(template / name) != expected:
            raise ValueError('Frozen firmware template differs')
    if sha(Path(__file__).with_name('run_multinet_probe.py')) != plan['runner_sha256']:
        raise ValueError('Frozen runner differs')
    # Read only existing metadata to preserve voice roles; never load held-out WAVs.
    roles = {}
    for row in json.loads(args.source_metadata.read_text()):
        if row['engine'] == 'macos-say':
            voice = row['path'].split('/')[2].rsplit('-', 1)[0]
            roles.setdefault(voice, set()).add(row['split'])
    if any(roles.get(voice) != {'training'} for voice in plan['voices']):
        raise ValueError('Control voice is not exclusively assigned to training')
    installed = subprocess.check_output([plan['say_binary'], '-v', '?'], text=True)
    if any(not any(line.split('#')[0].split()[:2] == [voice, plan['voice_languages'][voice]]
                   for line in installed.splitlines()) for voice in plan['voices']):
        raise ValueError('Expected installed English voice is unavailable')
    model = packed_model(args.model, plan['model_files'])
    commands_path = args.model.parent / 'fst/commands_en.txt'
    if sha(commands_path) != plan['initial_commands_sha256']:
        raise ValueError('Pinned initialization commands differ')
    stock = list(csv.reader(commands_path.read_text().splitlines()))
    components = args.model.resolve().parents[3]
    dependency_manifest = 'dependencies:\n  idf: "==5.5.4"\n'
    for name, identity in plan['dependencies'].items():
        component = components / ('espressif__' + name)
        if (sha(component / 'idf_component.yml') != identity['manifest_sha256']
                or (component / '.component_hash').read_text().strip() != identity['component_hash']):
            raise ValueError('Pinned local dependency identity differs')
        dependency_manifest += (f'  espressif/{name}:\n    version: "=={identity["version"]}"\n'
                                f'    override_path: {json.dumps(str(component))}\n')
    arms = plan['command_arms']
    if (len(arms) != 2 or not arms[0]['stock_grammar'] or arms[0]['commands']
            or arms[1]['stock_grammar'] or not 0 < len(arms[1]['commands']) <= 5):
        raise ValueError('Expected original initialization then one replacement grammar')
    args.output.mkdir(parents=True)
    shutil.copytree(template, args.output, dirs_exist_ok=True)
    sources_dir = args.output / 'sources'
    sources_dir.mkdir()
    records, payloads = [], []
    for voice in plan['voices']:
        for phrase in plan['phrases']:
            number = len(records)
            path = sources_dir / f'{number:02d}.wav'
            command = [plan['say_binary'], '-v', voice, '-o', str(path), '--file-format=WAVE',
                       '--data-format=LEI16@16000', '--channels=1', phrase['text']]
            subprocess.run(command, capture_output=True, check=True)
            with wave.open(str(path), 'rb') as wav:
                if (wav.getframerate(), wav.getnchannels(), wav.getsampwidth(), wav.getcomptype()) != (16000, 1, 2, 'NONE'):
                    raise ValueError('Synthesis output format differs')
                signal = wav.readframes(wav.getnframes())
            samples = array('h', signal)
            if sys.byteorder != 'little':
                samples.byteswap()
            if not 1600 < len(samples) < 8 * 16000 or max(abs(v) for v in samples) < 1000:
                raise ValueError('Synthesis WAV is empty, silent or unexpectedly long')
            rendered = bytes(plan['lead_samples'] * 2) + signal + bytes(plan['tail_samples'] * 2)
            aligned = rendered + bytes((-len(rendered)) % 1024)
            expected = [grammar_ids(phrase['text'], arm, stock) for arm in arms]
            if phrase['role'] == 'known_command' and any(len(ids) != 1 for ids in expected):
                raise ValueError('Control phrase is absent/ambiguous in a fixed grammar')
            payloads.append(rendered)
            records.append({'clip': number, 'engine': 'macos-say', 'voice': voice,
                'split': 'training', 'role': phrase['role'], 'text': phrase['text'],
                'label': int(phrase['role'] == 'wake_target'),
                'source_sha256': sha(path), 'source_samples': len(samples),
                'source_peak': max(abs(v) for v in samples),
                'source_rms': math.sqrt(sum(v * v for v in samples) / len(samples)),
                'samples': len(rendered) // 2, 'expected_command_ids': expected,
                'rendered_pcm_sha256': hashlib.sha256(rendered).hexdigest(),
                'detect_pcm_crc32': zlib.crc32(aligned)})
    pcm = struct.pack('<I', len(records)) + b''.join(struct.pack('<I', len(x) // 2) + x for x in payloads)
    (args.output / 'main/idf_component.yml').write_text(dependency_manifest)
    shutil.copyfile(commands_path, args.output / 'main/commands_en.txt')
    for name, raw in (('model', model), ('fixtures', pcm)):
        (args.output / 'main' / (name + '.bin')).write_bytes(struct.pack('<I', len(raw)) + zlib.compress(raw, 9))
    header = '#pragma once\n'
    header += f'constexpr unsigned fixture_count = {len(records)};\n'
    header += f'constexpr unsigned detector_timeout_ms = {plan["detector_timeout_ms"]};\n'
    header += f'constexpr float detector_threshold = {plan["detector_threshold"]}f;\n'
    header += f'constexpr unsigned command_arm_count = {len(arms)};\n'
    header += 'constexpr unsigned command_counts[] = {' + ','.join(str(len(a['commands'])) for a in arms) + '};\n'
    header += 'constexpr const char *command_strings[][5] = {\n'
    for arm in arms:
        header += '{' + ','.join(json.dumps(command) for command in arm['commands']) + '},\n'
    header += '};\n'
    (args.output / 'main/fixture_config.h').write_text(header)
    manifest = {'scope': __doc__, 'plan_sha256': sha(args.plan), 'cases': len(records) * len(arms),
        'sources': records, 'command_arms': arms, 'detector_threshold': plan['detector_threshold'],
        'detector_timeout_ms': plan['detector_timeout_ms'], 'all_sources_training_only': True,
        'heldout_source_hash_disjoint': True, 'audio_speed_changed': False,
        'synthesis_rate_override': False, 'speaker_playback': False,
        'model_raw_bytes': len(model), 'fixtures_raw_bytes': len(pcm),
        'files': {name: sha(args.output / name) for name in ('main/model.bin', 'main/fixtures.bin',
            'main/fixture_config.h', 'main/commands_en.txt')},
        'model_files': plan['model_files'], 'dependencies': plan['dependencies'], 'accepted': False}
    (args.output / 'vectors-manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({'cases': manifest['cases'], 'sources': len(records),
        'manifest_sha256': sha(args.output / 'vectors-manifest.json'), 'accepted': False}, indent=2))


if __name__ == '__main__':
    main()
