#!/usr/bin/env python3
"""Validate complete PCM/control coverage without interpreting it as acceptance."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path

from audit_multinet_log import parse_log, sha
from run_multinet_probe import validate_result


def audit(report, manifest, uart):
    original = json.loads(report)
    vectors = json.loads(manifest)
    if (original['vectors_manifest_sha256'] != sha(manifest) or original['vectors'] != vectors
            or not vectors['all_sources_training_only'] or vectors['audio_speed_changed']
            or vectors['speaker_playback'] or vectors['synthesis_rate_override']):
        raise ValueError('Frozen manifest, source role or original-speed silent scope differs')
    parsed = parse_log(uart)
    if any(original.get(key) != value for key, value in parsed.items()):
        raise ValueError('Retained UART differs from runner records')
    validate_result(parsed, vectors)
    if not original.get('execution_completed') or 'error' in original:
        raise ValueError('Runner did not complete successfully')
    restored = original.get('restored', {})
    if not (restored.get('original_service_verified') and restored.get('hi_esp_listening')
            and restored.get('nyx_state') == 'up'):
        raise ValueError('Original app/service/listener restoration unverified')
    sources = {s['clip']: s for s in vectors['sources']}
    arms = vectors['command_arms']
    if parsed.get('arm_records') != [{'arm': n, 'stock_grammar': a['stock_grammar']}
                                     for n, a in enumerate(arms)]:
        raise ValueError('Original/replacement grammar identity or order differs')
    expected = {(n, clip) for n in range(len(arms)) for clip in sources}
    decoders = parsed.get('decoder_records', [])
    keys = [(r['arm'], r['clip']) for r in decoders]
    if len(keys) != len(set(keys)) or set(keys) != expected:
        raise ValueError('Missing, duplicate or unexpected final decoder records')
    events = parsed.get('events', [])
    event_keys = [(r['arm'], r['clip'], r['end_sample']) for r in events]
    if len(event_keys) != len(set(event_keys)):
        raise ValueError('Duplicate detection event')
    for row in events + decoders:
        key = row['arm'], row['clip']
        if key not in expected:
            raise ValueError('Event/decoder references an unexpected case')
        last = ((sources[row['clip']]['samples'] + 511) // 512) * 512
        if (not 0 < row['end_sample'] <= last or row['end_sample'] % 512
                or not 0 <= row['num'] <= 5 or len(row['commands']) != row['num']
                or not isinstance(row['raw_string'], str) or not isinstance(row['string'], str)):
            raise ValueError('Decoder result bounds or input position differs')
        for command in row['commands']:
            if (not isinstance(command['id'], int) or command['id'] <= 0
                    or not math.isfinite(command['probability']) or not 0 <= command['probability'] <= 1):
                raise ValueError('Invalid decoded command/probability')
    for row in events:
        if row['state'] != 1 or not row['num']:
            raise ValueError('Detection lacks a top-ranked command')
    for row in decoders:
        if row['end_sample'] != ((sources[row['clip']]['samples'] + 511) // 512) * 512:
            raise ValueError('Final decoder output precedes complete PCM coverage')
    summaries = []
    for row in parsed['clips']:
        arm, clip = row['arm'], row['clip']
        source = sources[clip]
        actual = [e for e in events if (e['arm'], e['clip']) == (arm, clip)]
        counts = Counter(e['commands'][0]['id'] == 1 for e in actual)
        if (row['input_crc32'] != source['detect_pcm_crc32']
                or row['target_events'] != counts[True] or row['other_events'] != counts[False]):
            raise ValueError('Consumed PCM CRC or event accounting differs')
        first = next((e for e in actual if e['commands'][0]['id'] == 1), None)
        if first and (row['first_target_end_sample'] != first['end_sample']
                      or row['first_target_probability'] != first['commands'][0]['probability']):
            raise ValueError('First command1 event differs from retained detection')
        wanted = source['expected_command_ids'][arm]
        summaries.append({'arm': arm, 'clip': clip, 'voice': source['voice'],
            'role': source['role'], 'text': source['text'], 'expected_command_ids': wanted,
            'detected_command_ids': [e['commands'][0]['id'] for e in actual],
            'expected_command_detected': any(e['commands'][0]['id'] in wanted for e in actual),
            'pcm_crc_verified': True})
    rollups = []
    for n, arm in enumerate(arms):
        value = {'arm': n, 'name': arm['name'], 'roles': {}}
        for role in sorted({s['role'] for s in vectors['sources']}):
            subset = [s for s in summaries if s['arm'] == n and s['role'] == role]
            value['roles'][role] = {'trials': len(subset),
                'expected_command_detected': sum(s['expected_command_detected'] for s in subset),
                'any_command_detected': sum(bool(s['detected_command_ids']) for s in subset)}
        value['known_controls_passed'] = all(s['expected_command_detected'] for s in summaries
            if s['arm'] == n and s['role'] == 'known_command')
        rollups.append(value)
    return {'scope': 'Silent standalone P4 known-command harness diagnostic, not wake acceptance',
        'plan_sha256': vectors['plan_sha256'], 'vectors_manifest_sha256': sha(manifest),
        'report_sha256': sha(report), 'uart_sha256': sha(uart),
        'probe_bin_sha256': original['probe_bin_sha256'],
        'execution_completed': True, 'coverage_and_pcm_crc_verified': True,
        'detector_threshold': vectors['detector_threshold'], 'arms': rollups,
        'sources': vectors['sources'], 'cases': summaries, 'events': events,
        'decoder_records': decoders, 'ready': parsed['ready'], 'timing': parsed['timing'],
        'cleanup_uninitialized_command_warning': b'The mn commands is not initialized' in uart,
        'restored': restored,
        'limits': ['Short control sources are not speaker-generalization evidence.',
            'Reset-separated short sources do not establish continuous FAR.',
            'Standalone resources/timing do not establish GUI/voice/OTA capacity.',
            'No acoustic playback or microphone/AFE transfer was tested.'], 'accepted': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--uart', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.report.read_bytes(), args.manifest.read_bytes(), args.uart.read_bytes())
    with args.output.open('x') as out:
        out.write(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'arms': result['arms'], 'restored': result['restored'], 'accepted': False}, indent=2))


if __name__ == '__main__':
    main()
