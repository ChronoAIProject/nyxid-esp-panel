#!/usr/bin/env python3
"""Combine frozen training PCM with source-defined controls for silent P4 diagnosis.

No new synthesis, score-based source selection, microphone or speaker access.
Every Nyx fixture is reused byte-for-byte; the two upstream controls bracket
it in each grammar. Neither controls nor synthetic clips establish acceptance.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import struct
import zlib

from prepare_multinet_probe import sha


def unpack_fixtures(blob, records):
    """Check exact container coverage and each independently pinned PCM hash."""
    if len(blob) < 5:
        raise ValueError('Truncated fixture container')
    decoder = zlib.decompressobj()
    raw = decoder.decompress(blob[4:]) + decoder.flush()
    if (not decoder.eof or decoder.unused_data or len(raw) != struct.unpack('<I', blob[:4])[0]
            or len(raw) < 4 or struct.unpack('<I', raw[:4])[0] != len(records)):
        raise ValueError('Fixture container size/count differs')
    cursor, payloads = 4, []
    for clip, record in enumerate(records):
        if record['clip'] != clip or cursor + 4 > len(raw):
            raise ValueError('Fixture record order or bounds differs')
        samples = struct.unpack_from('<I', raw, cursor)[0]
        cursor += 4
        end = cursor + samples * 2
        pcm = raw[cursor:end]
        if (samples != record['samples'] or end > len(raw)
                or hashlib.sha256(pcm).hexdigest() != record['rendered_pcm_sha256']):
            raise ValueError('Fixture PCM identity differs')
        payloads.append(pcm)
        cursor = end
    if cursor != len(raw):
        raise ValueError('Fixture container has unaccounted data')
    return payloads


def detect_crc(pcm):
    # Firmware pads the last fractional 512-sample block before detect().
    return zlib.crc32(pcm + bytes((-len(pcm)) % 1024))


def load_project(path, expected):
    if sha(path / 'vectors-manifest.json') != expected['vectors_manifest_sha256']:
        raise ValueError('Frozen source project manifest differs')
    manifest = json.loads((path / 'vectors-manifest.json').read_text())
    for name, identity in manifest['files'].items():
        if sha(path / name) != identity:
            raise ValueError('Frozen source project file differs')
    if sha(path / 'main/idf_component.yml') != expected['dependency_manifest_sha256']:
        raise ValueError('Frozen dependency overrides differ')
    if not manifest['all_sources_training_only'] or manifest['audio_speed_changed']:
        raise ValueError('Expected original-speed training-only fixtures')
    return manifest, unpack_fixtures((path / 'main/fixtures.bin').read_bytes(), manifest['sources'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--nyx-project', type=Path, required=True)
    parser.add_argument('--reference-project', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    root = Path(__file__).parent
    if args.output.exists():
        parser.error('Use a fresh private output directory')
    for name, expected in plan['scripts'].items():
        if sha(root / name) != expected:
            raise ValueError('Frozen preparation, helper, runner or auditor differs')
    for name, expected in plan['template_files'].items():
        if sha(root / 'multinet_probe' / name) != expected:
            raise ValueError('Frozen firmware template differs')
    nyx, nyx_pcm = load_project(args.nyx_project, plan['nyx_project'])
    ref, ref_pcm = load_project(args.reference_project, plan['reference_project'])
    if (not nyx['heldout_source_hash_disjoint'] or len(nyx_pcm) != 24 or len(ref_pcm) != 2
            or nyx['files']['main/model.bin'] != ref['files']['main/model.bin']
            or nyx['files']['main/commands_en.txt'] != ref['files']['main/commands_en.txt']
            or nyx['dependencies'] != ref['dependencies']):
        raise ValueError('Original training selection, model or source controls differ')
    arms, phonemes = plan['command_arms'], plan['command_phonemes']
    if len(arms) != 2 or len(phonemes) != len(arms):
        raise ValueError('Expected two frozen replacement grammars')
    for arm, row in zip(arms, phonemes):
        if (arm['stock_grammar'] or not 1 < len(arm['commands']) <= 5
                or arm['commands'][0] != 'HEY NICKS' or arm['commands'][-1] != 'TELL ME A JOKE'
                or len(row) != len(arm['commands']) or row[:-1] != [None] * (len(row) - 1)
                or row[-1] != 'TfL Mm c qbK'):
            raise ValueError('Target G2P and original control phoneme policy differs')
    records, payloads = [], []
    order = [(ref['sources'][0], ref_pcm[0])] + list(zip(nyx['sources'], nyx_pcm)) + [(ref['sources'][1], ref_pcm[1])]
    for source, pcm in order:
        record = {**source, 'clip': len(records), 'detect_pcm_crc32': detect_crc(pcm)}
        if source.get('role') == 'known_command':
            lead = source['lead_samples'] * 2
            signal = pcm[lead:lead + source['source_samples'] * 2]
            if hashlib.sha256(signal).hexdigest() != plan['control_source_pcm_sha256']:
                raise ValueError('Original source-control PCM identity differs')
            record['expected_command_ids'] = [[len(a['commands'])] for a in arms]
        else:
            record.update(split='training', text=' / '.join(source['texts']),
                          voice='unresolved-source:' + source['source_sha256'][:16])
            record['role'] = 'target' if source['label'] else 'negative'
            record['expected_command_ids'] = [[1] if source['label'] else [] for _ in arms]
        records.append(record)
        payloads.append(pcm)
    calls = sum((r['samples'] + 511) // 512 for r in records) * len(arms)
    if calls != plan['expected_detect_calls'] or calls > 20000 or len(records) * len(arms) > 128:
        raise ValueError('Frozen call count or firmware capacity differs')
    raw = struct.pack('<I', len(records)) + b''.join(
        struct.pack('<I', r['samples']) + pcm for r, pcm in zip(records, payloads))
    header = '#pragma once\n'
    header += f'constexpr unsigned fixture_count = {len(records)};\n'
    header += f'constexpr unsigned detector_timeout_ms = {plan["detector_timeout_ms"]};\n'
    header += f'constexpr float detector_threshold = {plan["detector_threshold"]}f;\n'
    header += f'constexpr unsigned command_arm_count = {len(arms)};\n'
    header += 'constexpr unsigned command_counts[] = {' + ','.join(str(len(a['commands'])) for a in arms) + '};\n'
    header += 'constexpr const char *command_strings[][5] = {\n'
    for arm in arms:
        header += '{' + ','.join(json.dumps(c) for c in arm['commands']) + '},\n'
    header += '};\n#define MN_EXPLICIT_COMMAND_PHONEMES 1\nconstexpr const char *command_phonemes[][5] = {\n'
    for row in phonemes:
        header += '{' + ','.join('nullptr' if p is None else json.dumps(p) for p in row) + '},\n'
    header += '};\n'
    args.output.mkdir(parents=True)
    shutil.copytree(root / 'multinet_probe', args.output, dirs_exist_ok=True)
    for name in ('main/model.bin', 'main/commands_en.txt', 'main/idf_component.yml'):
        shutil.copyfile(args.nyx_project / name, args.output / name)
    (args.output / 'main/fixtures.bin').write_bytes(struct.pack('<I', len(raw)) + zlib.compress(raw, 9))
    (args.output / 'main/fixture_config.h').write_text(header)
    manifest = {**nyx, 'scope': __doc__, 'plan_sha256': sha(args.plan), 'sources': records,
        'cases': len(records) * len(arms), 'command_arms': arms, 'command_phonemes': phonemes,
        'detector_threshold': plan['detector_threshold'], 'detector_timeout_ms': plan['detector_timeout_ms'],
        'expected_detect_calls': calls, 'fixtures_raw_bytes': len(raw),
        'speaker_playback': False, 'synthesis_rate_override': False, 'heldout_source_audio_accessed': False,
        'heldout_source_hash_disjoint': None, 'nyx_heldout_source_hash_disjoint': True,
        'reference_heldout_source_hash_disjoint': None,
        'source_partition_policy': 'All 24 original Nyx training sources unchanged; upstream reference assigned only to training-side harness controls, not independent speakers or quality evaluation.',
        'source_projects': {'nyx': plan['nyx_project'], 'reference': plan['reference_project']},
        'files': {name: sha(args.output / name) for name in nyx['files']}, 'accepted': False}
    (args.output / 'vectors-manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({'cases': manifest['cases'], 'calls': calls,
        'vectors_manifest_sha256': sha(args.output / 'vectors-manifest.json'), 'accepted': False}))


if __name__ == '__main__':
    main()
