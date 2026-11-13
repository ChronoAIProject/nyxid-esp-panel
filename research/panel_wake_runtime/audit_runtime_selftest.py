"""Check one full-app exact-PCM verdict against its bound host fixture report."""
import argparse
import hashlib
import json
from pathlib import Path
import re


def audit(log, fixture):
    prime = fixture['app_prime']
    if 'event_arithmetic' in prime:
        if (prime['event_arithmetic'] != 'uint8-sum-binary64-v1'
                or prime.get('host_event_steps') != prime['expected_device_event_steps']):
            raise ValueError('Explicit event arithmetic must agree with host activation steps')
    markers = list(re.finditer(
        r'exact-PCM selftest outputs=(\d+) mismatches=(\d+) passed=(true|false)', log))
    if len(markers) != 1:
        raise ValueError('Expected exactly one self-test verdict')
    marker = markers[0]
    outputs, mismatches, passed = marker.groups()
    if (int(outputs), int(mismatches), passed) != (fixture['expected_outputs'], 0, 'true'):
        raise ValueError('MCU output comparison failed or has incomplete coverage')
    before = log[:marker.start()]
    identities = re.findall(r'panel\.wake\.research: model=([0-9a-f]{64})\b', before)
    if identities != [fixture['model_sha256']]:
        raise ValueError('Expected one matching model initialization before self-test')
    steps = [int(step) for step in re.findall(
        r'panel\.wake\.research: activation [^\r\n]*?step=(\d+)\b', before)]
    if steps != fixture['app_prime']['expected_device_event_steps']:
        raise ValueError('MCU activation steps differ from the declared device arithmetic')
    return {'scope': __doc__, 'model_sha256': fixture['model_sha256'],
            'fixture_sha256': fixture['fixture_sha256'], 'outputs': int(outputs),
            'mismatches': 0, 'activation_steps': steps,
            'limits': 'Checks the firmware-reported comparison and event steps for this fixture only; no acoustic recognition, latency or general MCU equivalence claim.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--uart', type=Path, required=True)
    parser.add_argument('--fixture-report', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Use a fresh output file')
    log = args.uart.read_bytes(); raw = args.fixture_report.read_bytes()
    result = audit(log.decode(errors='replace'), json.loads(raw))
    result.update(uart_sha256=hashlib.sha256(log).hexdigest(),
                  fixture_report_sha256=hashlib.sha256(raw).hexdigest(),
                  auditor_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
