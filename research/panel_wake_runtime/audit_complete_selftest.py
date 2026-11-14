"""Bind a V2 full-app UART verdict to exact PCM/raw bytes and calibration."""
import argparse
import json
from pathlib import Path
import re

from audit_runtime_selftest import audit as audit_verdict
from prepare_complete_selftest import (ARITHMETIC, TAIL, PRIME_REQUESTED, PRIME_ROUNDED,
    event_steps, operating_point, parse_fixture, sha)


def audit(log, data, report, calibration_bytes):
    fixture = parse_fixture(data)
    if (report['schema'] != 'complete-runtime-selftest-v2'
            or report['fixture_sha256'] != sha(data) or report['model_sha256'] != fixture['model_sha256']
            or report['calibration_sha256'] != sha(calibration_bytes)
            or report['pcm_sha256'] != sha(fixture['pcm'])
            or report['expected_outputs_sha256'] != sha(fixture['raw'])
            or report['samples'] != fixture['samples'] or report['expected_outputs'] != fixture['outputs']
            or report['sample_rate'] != 16000 or report['tail_policy'] != TAIL
            or report['final_wave_padding'] != (-fixture['samples'])%160
            or report['final_feature_padding'] != (-fixture['frames'])%3
            or report['reset_reproducible'] is not True):
        raise ValueError('Changed complete fixture report or calibration identity')
    threshold = operating_point(json.loads(calibration_bytes), fixture['model_sha256'])
    prime = report['app_prime']
    if (prime['training_context_frames'] != 225 or prime['requested_prime_samples'] != PRIME_REQUESTED
            or prime['rounded_prime_samples'] != PRIME_ROUNDED or prime['tail_pad_samples'] != 0
            or prime['source_samples'] != fixture['samples']-PRIME_ROUNDED or prime['source_samples'] < 960
            or prime['threshold'] != threshold or prime['smoothing_steps'] != 5 or prime['refractory_steps'] != 67
            or prime['event_arithmetic'] != ARITHMETIC
            or fixture['pcm'][:PRIME_ROUNDED*2] != bytes(PRIME_ROUNDED*2)
            or report['source_pcm_sha256'] != sha(fixture['pcm'][PRIME_ROUNDED*2:])
            or report['runtime']['feature_frames'] != fixture['frames']):
        raise ValueError('Changed application prime or complete PCM policy')
    events = event_steps(fixture['raw'], threshold)
    if prime['host_event_steps'] != events or prime['expected_device_event_steps'] != events:
        raise ValueError('Declared events differ from complete raw bytes')
    markers = list(re.finditer(r'panel\.wake\.research: complete selftest fixture=([0-9a-f]{64}) samples=(\d+) outputs=(\d+)', log))
    if len(markers) != 1 or markers[0].groups() != (sha(data), str(fixture['samples']), str(fixture['outputs'])):
        raise ValueError('UART must identify the exact complete fixture')
    verdict = re.search(r'exact-PCM selftest outputs=', log)
    if verdict is None or markers[0].start() >= verdict.start():
        raise ValueError('Fixture identity must precede its verdict')
    result = audit_verdict(log, report)
    return dict(schema='complete-runtime-selftest-uart-audit-v2', **result,
        samples=fixture['samples'], final_wave_padding=report['final_wave_padding'],
        final_feature_padding=report['final_feature_padding'], calibration_sha256=sha(calibration_bytes),
        fixture_identity_in_uart=True, raw_byte_event_recalculation_exact=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('uart', 'fixture', 'fixture-report', 'calibration', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    log, raw_report = args.uart.read_bytes(), args.fixture_report.read_bytes()
    result = audit(log.decode(errors='replace'), args.fixture.read_bytes(), json.loads(raw_report), args.calibration.read_bytes())
    result.update(uart_sha256=sha(log), fixture_report_sha256=sha(raw_report), auditor_sha256=sha(Path(__file__).read_bytes()))
    with args.output.open('x') as output:
        output.write(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result))
