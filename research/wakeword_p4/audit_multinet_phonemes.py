#!/usr/bin/env python3
"""Check actual input/stock replication for the frozen phoneme counterfactual."""
import argparse
import csv
import json
from pathlib import Path
import re

from audit_multinet_log import parse_log, sha
from run_multinet_probe import validate_result


def active_commands(raw):
    arm, result = None, {}
    for line in raw.decode('utf-8', errors='replace').splitlines():
        if line.startswith('MN_ARM '):
            arm = json.loads(line[len('MN_ARM '):])['arm']
            if arm in result:
                raise ValueError('Duplicate grammar arm')
            result[arm] = []
        match = re.fullmatch(r'Command (\d+): (.*)', line)
        if match:
            if arm is None:
                raise ValueError('Command printed before arm identity')
            result[arm].append((int(match[1]), match[2]))
    return result


def audit(before_result, after_result, before_manifest, after_manifest,
          before_uart, after_uart, reference_commands):
    results = [json.loads(before_result), json.loads(after_result)]
    manifests = [json.loads(before_manifest), json.loads(after_manifest)]
    streams = [before_uart, after_uart]
    for result, manifest, raw, manifest_bytes in zip(results, manifests, streams,
            (before_manifest, after_manifest)):
        if (result['vectors_manifest_sha256'] != sha(manifest_bytes)
                or result['uart_sha256'] != sha(raw) or not result['execution_completed']
                or not result['coverage_and_pcm_crc_verified'] or not result['restored']['original_service_verified']
                or not result['restored']['hi_esp_listening'] or result['restored']['nyx_state'] != 'up'):
            raise ValueError('Completed source/input/restoration identities differ')
        parsed = parse_log(raw)
        validate_result(parsed, manifest)
        if parsed.get('events', []) != result['events']:
            raise ValueError('Retained native events differ from audited result')
    before, after = manifests
    for key in ('sources', 'command_arms', 'cases', 'detector_threshold',
                'detector_timeout_ms', 'dependencies', 'source_reference', 'model_raw_bytes', 'fixtures_raw_bytes'):
        if before[key] != after[key]:
            raise ValueError('Unexpected input/operating-point change: ' + key)
    for name in ('main/model.bin', 'main/fixtures.bin', 'main/commands_en.txt'):
        if before['files'][name] != after['files'][name]:
            raise ValueError('Unexpected model/PCM/reference change: ' + name)
    if sha(reference_commands) != after['files']['main/commands_en.txt']:
        raise ValueError('Pinned original command reference differs')
    matrix = after['command_phonemes']
    if 'command_phonemes' in before:
        raise ValueError('Baseline must use original G2P without explicit phonemes')
    declared = [(arm, n + 1, phone) for arm, row in enumerate(matrix)
                for n, phone in enumerate(row) if phone is not None]
    if len(declared) != 1:
        raise ValueError('Expected exactly one explicit phoneme intervention')
    arm, command_id, new_phone = declared[0]
    text = after['command_arms'][arm]['commands'][command_id - 1]
    stock = list(csv.reader(reference_commands.decode().splitlines()))
    if not any(row[1] == text and row[2] == new_phone for row in stock):
        raise ValueError('Intervention does not match the pinned original phoneme spelling')
    old_commands, new_commands = map(active_commands, streams)
    if old_commands.keys() != new_commands.keys():
        raise ValueError('Actual grammar arms differ')
    changes = []
    for n in old_commands:
        if len(old_commands[n]) != len(new_commands[n]):
            raise ValueError('Actual grammar size changed')
        for old, new in zip(old_commands[n], new_commands[n]):
            if old != new:
                changes.append({'arm': n, 'before_id': old[0], 'after_id': new[0],
                                'before_phonemes': old[1], 'after_phonemes': new[1]})
    if (len(changes) != 1 or changes[0]['arm'] != arm
            or changes[0]['before_id'] != command_id or changes[0]['after_id'] != command_id
            or changes[0]['after_phonemes'] != new_phone):
        raise ValueError('Actual active grammar changes differ from the declared intervention')
    stock_before = [e for e in results[0]['events'] if before['command_arms'][e['arm']]['stock_grammar']]
    stock_after = [e for e in results[1]['events'] if after['command_arms'][e['arm']]['stock_grammar']]
    if not stock_before or stock_before != stock_after:
        raise ValueError('Stock native-event trace did not reproduce exactly')
    return {'scope': 'Actual-P4 phoneme counterfactual on one repeated upstream recording; not Nyx quality acceptance',
        'before_result_sha256': sha(before_result), 'after_result_sha256': sha(after_result),
        'before_manifest_sha256': sha(before_manifest), 'after_manifest_sha256': sha(after_manifest),
        'same_model_and_consumed_pcm_verified': True, 'same_operating_point_and_grammar_texts_verified': True,
        'actual_active_command_changes': changes, 'original_reference_spelling_verified': True,
        'stock_native_event_trace_exactly_reproduced': True,
        'stock_events': stock_after, 'before_arms': results[0]['arms'], 'after_arms': results[1]['arms'],
        'interpretation': 'Compare the retained role counts for the source-defined single-phoneme intervention. A passing known-source control is not independent phrase generalization, continuous FAR or full-app acceptance.',
        'accepted': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('before-result', 'after-result', 'before-manifest', 'after-manifest',
                 'before-uart', 'after-uart', 'reference-commands', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    result = audit(*(getattr(args, name).read_bytes() for name in
        ('before_result', 'after_result', 'before_manifest', 'after_manifest',
         'before_uart', 'after_uart', 'reference_commands')))
    with args.output.open('x') as out:
        out.write(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
