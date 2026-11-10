#!/usr/bin/env python3
"""Prepare the pinned upstream original-PCM control, with no audio playback.

Reuse a hash-verified prepared project's model/dependencies and the current
frozen template. Preserve every sample, including the upstream test's partial
final source block. Two prefix placements are correlated transport controls,
never independent recognition/false-wake evaluation or a training addition.
"""
import argparse
from array import array
import hashlib
import json
from pathlib import Path
import re
import shutil
import struct
import zlib

from prepare_multinet_probe import sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--base-project', type=Path, required=True)
    parser.add_argument('--upstream-source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    root = Path(__file__).parent
    if args.output.exists():
        parser.error('Use a fresh private output directory')
    for name, key in ((Path(__file__).name, 'preparation_script_sha256'),
                      ('run_multinet_probe.py', 'runner_sha256'),
                      ('audit_multinet_control.py', 'audit_script_sha256')):
        if sha(root / name) != plan[key]:
            raise ValueError('Frozen preparation, runner or auditor differs')
    for name, expected in plan['template_files'].items():
        if sha(root / 'multinet_probe' / name) != expected:
            raise ValueError('Frozen firmware template differs')
    base = args.base_project
    if sha(base / 'vectors-manifest.json') != plan['base_vectors_manifest_sha256']:
        raise ValueError('Original prepared project manifest differs')
    previous = json.loads((base / 'vectors-manifest.json').read_text())
    for name, expected in previous['files'].items():
        if sha(base / name) != expected:
            raise ValueError('Original prepared project model/fixtures differ')
    if sha(base / 'main/idf_component.yml') != plan['base_dependency_manifest_sha256']:
        raise ValueError('Original pinned dependency overrides differ')
    for name, expected in plan['upstream_files'].items():
        if sha(args.upstream_source / name) != expected:
            raise ValueError('Pinned upstream reference/test source differs')
    header = (args.upstream_source / 'tell_me_a_joke.h').read_text()
    match = re.fullmatch(r'\s*#include <stdio.h>\s*const unsigned char tell_me_a_joke\[\] = \{(.*?)\};\s*', header, re.S)
    if not match or not re.fullmatch(r'[\s,0-9a-fx]+', match[1]):
        raise ValueError('Reference header is not the original literal byte array')
    signal = bytes(int(x, 16) for x in re.findall(r'0x[0-9a-f]{2}', match[1]))
    if (len(signal) != plan['source_pcm_bytes']
            or hashlib.sha256(signal).hexdigest() != plan['source_pcm_sha256']):
        raise ValueError('Decoded original PCM identity differs')
    values = array('h', signal)
    records, payloads = [], []
    total = plan['blocks_per_case'] * 512
    for lead in plan['prefix_samples']:
        tail = total - lead - len(values)
        if tail <= 0:
            raise ValueError('Fixed transport window truncates original reference PCM')
        rendered = bytes(lead * 2) + signal + bytes(tail * 2)
        payloads.append(rendered)
        records.append({'clip': len(records), 'engine': 'esp-sr-upstream-unit-sample',
            'voice': 'unknown-upstream-reference', 'split': 'training_control',
            'role': 'known_command', 'text': 'Tell me a joke.', 'label': 0,
            'source_sha256': plan['source_pcm_sha256'], 'source_samples': len(values),
            'source_peak': max(abs(v) for v in values), 'lead_samples': lead,
            'tail_samples': tail, 'samples': total, 'expected_command_ids': [[1], [3]],
            'rendered_pcm_sha256': hashlib.sha256(rendered).hexdigest(),
            'detect_pcm_crc32': zlib.crc32(rendered)})
    arms = plan['command_arms']
    if arms != previous['command_arms'] or len(records) != 2:
        raise ValueError('Expected unchanged stock/mixed grammars and two prefix placements')
    pcm = struct.pack('<I', len(records)) + b''.join(struct.pack('<I', total) + x for x in payloads)
    args.output.mkdir(parents=True)
    shutil.copytree(root / 'multinet_probe', args.output, dirs_exist_ok=True)
    for name in ('main/idf_component.yml', 'main/commands_en.txt', 'main/model.bin'):
        shutil.copyfile(base / name, args.output / name)
    (args.output / 'main/fixtures.bin').write_bytes(struct.pack('<I', len(pcm)) + zlib.compress(pcm, 9))
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
    manifest = {**previous, 'scope': __doc__, 'plan_sha256': sha(args.plan),
        'cases': len(records) * len(arms), 'sources': records, 'command_arms': arms,
        'detector_timeout_ms': plan['detector_timeout_ms'], 'detector_threshold': plan['detector_threshold'],
        'all_sources_training_only': True,
        'source_partition_policy': 'Upstream unit sample assigned only to training-side harness controls; not independent quality evaluation or a model-fitting addition.',
        'heldout_source_hash_disjoint': None, 'heldout_source_audio_accessed': False,
        'fixtures_raw_bytes': len(pcm),
        'files': {name: sha(args.output / name) for name in previous['files']},
        'source_reference': {'repository': 'espressif/esp-sr', 'commit': plan['upstream_commit'],
                             'files': plan['upstream_files']}, 'accepted': False}
    (args.output / 'vectors-manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({'cases': manifest['cases'], 'calls': len(records) * len(arms) * plan['blocks_per_case'],
        'manifest_sha256': sha(args.output / 'vectors-manifest.json'), 'accepted': False}, indent=2))


if __name__ == '__main__':
    main()
