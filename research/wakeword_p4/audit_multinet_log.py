#!/usr/bin/env python3
"""Reconstruct explicitly reported M/newline/N_CLIP fragments, without hardware.

The original log/report remain untouched. Only that exact marker split can be
joined; JSON payloads are not repaired or invented. Coverage validation must
still match the frozen manifest and every originally parsed record.
"""
import argparse
import hashlib
import json
from pathlib import Path

from run_multinet_probe import UartLines, record_line, validate_result


def sha(data):
    return hashlib.sha256(data).hexdigest()


def parse_log(raw):
    result = {'clips': []}
    lines = UartLines()
    for line in lines.feed(raw):
        record_line(line.decode('utf-8', errors='replace').strip(), result)
    if lines.pending:
        raise ValueError('Retained log ends with an incomplete line')
    return result


def reconstruct(raw):
    lines = raw.splitlines(keepends=True)
    output, repairs = [], []
    n = 0
    while n < len(lines):
        if (lines[n].rstrip(b'\r\n') == b'M' and n + 1 < len(lines)
                and lines[n + 1].startswith(b'N_CLIP {')):
            joined = b'M' + lines[n + 1]
            recovered = parse_log(joined)
            if len(recovered['clips']) != 1:
                raise ValueError('Split marker has no single complete clip payload')
            row = recovered['clips'][0]
            repairs.append({'original_lines_1based': [n + 1, n + 2],
                            'arm': row['arm'], 'clip': row['clip'],
                            'joined_line_sha256': sha(joined)})
            output.append(joined)
            n += 2
        else:
            output.append(lines[n])
            n += 1
    return b''.join(output), repairs


def audit(raw, original, manifest, expected_uart_sha256, expected_repairs):
    if sha(raw) != expected_uart_sha256:
        raise ValueError('Retained UART identity differs')
    report = json.loads(original)
    vectors = json.loads(manifest)
    if (report['vectors_manifest_sha256'] != sha(manifest)
            or report['vectors'] != vectors or not vectors['all_sources_training_only']
            or vectors['audio_speed_changed']):
        raise ValueError('Original manifest identity or training scope differs')
    unmodified = parse_log(raw)
    if any(unmodified[key] != report[key] for key in ('clips', 'ready', 'timing', 'done')):
        raise ValueError('Raw parsed records differ from original failed report')
    try:
        validate_result(unmodified, vectors)
    except ValueError:
        pass
    else:
        raise ValueError('Expected an original coverage failure; no repair is needed')
    if report.get('error') != {'type': 'ValueError',
            'message': 'Missing, duplicate or unexpected MultiNet clip results'}:
        raise ValueError('Original runner failed for another reason')
    repaired, repairs = reconstruct(raw)
    if len(repairs) != expected_repairs:
        raise ValueError('Marker repair count differs from explicit expectation')
    result = parse_log(repaired)
    validate_result(result, vectors)
    restored = report.get('restored', {})
    if not (restored.get('original_service_verified') and restored.get('hi_esp_listening')
            and restored.get('nyx_state') == 'up'):
        raise ValueError('Original app/service/listener restoration was not verified')
    groups = []
    for arm, grammar in enumerate(vectors['command_arms']):
        rows = {r['clip']: r for r in result['clips'] if r['arm'] == arm}
        summary = {'arm': arm, 'name': grammar['name'], 'commands': grammar['commands']}
        for label, name in ((1, 'positive'), (0, 'negative')):
            subset = [rows[s['clip']] for s in vectors['sources'] if s['label'] == label]
            summary[name] = {'trials': len(subset),
                'target_triggered_trials': sum(r['target_events'] > 0 for r in subset),
                'target_events': sum(r['target_events'] for r in subset),
                'other_command_events': sum(r['other_events'] for r in subset),
                'timeouts': sum(r['timeouts'] for r in subset)}
        groups.append(summary)
    return {
        'scope': 'Standalone P4 training-source MultiNet7 diagnostic; no acoustic or whole-app acceptance',
        'plan_sha256': vectors['plan_sha256'],
        'probe_bin_sha256': report['probe_bin_sha256'],
        'restore_bin_sha256': report['restore_bin_sha256'],
        'vectors_manifest_sha256': sha(manifest),
        'retained_uart_sha256': sha(raw), 'original_report_sha256': sha(original),
        'reconstruction': {'rule': 'Join only an entire M line followed by N_CLIP and its unchanged JSON payload',
            'repairs': repairs, 'reconstructed_uart_sha256': sha(repaired),
            'original_parsed_clips': len(unmodified['clips']), 'reconstructed_clips': len(result['clips']),
            'original_runner_error': report['error'],
            'original_runner_validation_passed': False, 'offline_coverage_validation_passed': True,
            'original_files_modified': False, 'hardware_rerun_for_repair': False},
        'execution_completed': True, 'detector_threshold': vectors['detector_threshold'],
        'detector_timeout_ms': vectors['detector_timeout_ms'],
        'sources': vectors['sources'], 'arms': groups,
        'ready': result['ready'], 'timing': result['timing'], 'done': result['done'],
        'clips': result['clips'], 'restored': restored,
        'interpretation': 'Neither frozen grammar detects any of the 12 target sources at .9; not a usable candidate. Missing a known-supported-command positive control prevents concluding the model family cannot work.',
        'limits': ['Training-only synthetic short sources, not speaker generalization.',
            'Zero short-source negative detections do not establish continuous FAR.',
            'One non-target detection has no retained command ID/probability.',
            'Standalone resources and timings do not establish production GUI/voice/OTA capacity.',
            'Final detector destruction logs an already-freed command-state warning; ownership needs review before another firmware experiment.'],
        'accepted': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--uart', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--expected-uart-sha256', required=True)
    parser.add_argument('--expected-repairs', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.uart.read_bytes(), args.report.read_bytes(), args.manifest.read_bytes(),
                   args.expected_uart_sha256, args.expected_repairs)
    with args.output.open('x') as stream:
        stream.write(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'clips': len(result['clips']), 'arms': result['arms'],
                      'restored': result['restored'], 'accepted': result['accepted']}, indent=2))


if __name__ == '__main__':
    main()
