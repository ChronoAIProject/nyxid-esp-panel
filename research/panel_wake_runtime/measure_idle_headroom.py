"""Measure development-panel idle memory and optionally audit raw audio counts."""
import argparse
import hashlib
import json
from pathlib import Path
import re
from statistics import median
import time
import urllib.request


HEAP = ('free', 'spiram_free', 'dma_free', 'dma_largest_free',
        'internal_free', 'internal_largest_free')

TIMING_FIELDS = ('calls', 'pcm_samples', 'wall_us', 'detect_us', 'max_us', 'stack_free',
                 'end_us', 'read_bytes', 'pending_samples', 'rx_bytes', 'rx_overflows')
TIMING_LINE = re.compile(
    rb'panel\.wake\.timing: ' + b' '.join(key.encode() + rb'=(\d+)' for key in TIMING_FIELDS))


def capture_audit(log, backlog_bound, minimum_seconds):
    rows = [dict(zip(TIMING_FIELDS, map(int, values))) for values in TIMING_LINE.findall(log)]
    # Each complete wake window starts at the previous window's end.
    # Compare deltas of uint32 DMA counters, which can wrap on a long uptime.
    balance, intervals = 0, []
    for a, b in zip(rows, rows[1:]):
        rx_bytes = (b['rx_bytes'] - a['rx_bytes']) % 2**32
        overflows = (b['rx_overflows'] - a['rx_overflows']) % 2**32
        detector_bytes = 4 * (b['pcm_samples'] + b['pending_samples'] - a['pending_samples'])
        balance += rx_bytes - b['read_bytes']
        intervals.append({
            'contiguous': b['end_us'] - b['wall_us'] == a['end_us'],
            'rx_bytes': rx_bytes, 'new_overflows': overflows,
            'read_bytes': b['read_bytes'], 'detector_input_bytes': detector_bytes,
            'unaccounted_rx_bytes_since_start': balance,
            'wall_us': b['wall_us'], 'input_us': b['read_bytes'] * 1_000_000 / (16000 * 4),
        })
    wall_us = sum(w['wall_us'] for w in intervals)
    input_us = sum(w['input_us'] for w in intervals)
    passed = (len(intervals) >= 2 and wall_us >= minimum_seconds * 1_000_000
              and abs(wall_us - input_us) <= input_us * .01
              and all(w['contiguous'] and w['rx_bytes'] > 0 and w['new_overflows'] == 0
                      and w['read_bytes'] == w['detector_input_bytes']
                      and abs(w['unaccounted_rx_bytes_since_start']) <= backlog_bound
                      and abs(w['wall_us'] - w['input_us']) <= w['input_us'] * .03 for w in intervals))
    return {'passed': passed, 'backlog_bound_bytes': backlog_bound, 'minimum_seconds': minimum_seconds,
            'wall_seconds': wall_us / 1_000_000, 'input_seconds': input_us / 1_000_000,
            'windows': rows, 'intervals': intervals}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--status-url', required=True)
    p.add_argument('--expected-app-sha256', required=True)
    p.add_argument('--expected-slot', choices=('ota_0', 'ota_1'), required=True)
    p.add_argument('--expected-word', choices=('Hi ESP', 'Hey Nyx'), required=True)
    p.add_argument('--seconds', type=int, default=40)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--uart', type=Path, help='UART log from the same raw-input wake-timing build')
    p.add_argument('--capture-backlog-bytes', type=int, help='Verified hardware DMA and read buffer capacity')
    a = p.parse_args()
    if a.seconds < 30 or a.output.exists():
        p.error('Use at least 30 seconds and a fresh output file')
    if bool(a.uart) != (a.capture_backlog_bytes is not None) or (
            a.capture_backlog_bytes is not None and a.capture_backlog_bytes <= 0):
        p.error('Capture audit requires UART and a positive verified backlog bound')
    uart_start = a.uart.stat().st_size if a.uart else None
    samples, identity = [], None
    report = {'scope': __doc__, 'measure_seconds': a.seconds, 'completed': False,
              'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'app_sha256': a.expected_app_sha256, 'slot': a.expected_slot,
              'wake_word': a.expected_word, 'samples': samples,
              'speaker_playback': False, 'microphone_pcm_access': False, 'firmware_changed': False,
              'limits': 'Status snapshots measure idle heap and listening continuity. The optional capture audit measures idle RX byte conservation and cadence. Neither measures recognition, CPU utilization or calls.'}
    start = time.monotonic()
    try:
        for sample in range(a.seconds + 1):
            time.sleep(max(0, start + sample - time.monotonic()))
            with urllib.request.urlopen(a.status_url, timeout=4) as result:
                state = json.load(result)
            wake = state['voice']['wake']
            if (state['app_sha256'] != a.expected_app_sha256 or state['ota']['running'] != a.expected_slot
                    or state['ota']['busy'] or state['nyx']['state'] != 'up'
                    or state['voice']['busy'] or wake['word'] != a.expected_word or not wake['listening']):
                raise RuntimeError('Expected development app, connected idle listener or OTA slot changed')
            # Keep service identity constant without putting identifiers in the report.
            service = state['voice']['service_id'], state['voice']['agent_id']
            if identity is None:
                identity = service
            elif identity != service:
                raise RuntimeError('Voice service identity changed')
            row = {'elapsed_s': round(time.monotonic() - start, 3), 'uptime_s': state['uptime_s'],
                   'detections': wake['detections'], 'heap_bytes': {key: state['heap'][key] for key in HEAP}}
            if samples and (row['uptime_s'] < samples[-1]['uptime_s']
                            or row['detections'] != samples[-1]['detections']):
                raise RuntimeError('Board restarted or a wake event interrupted the idle window')
            samples.append(row)
        report['heap_summary_bytes'] = {}
        for key in HEAP:
            values = [row['heap_bytes'][key] for row in samples]
            report['heap_summary_bytes'][key] = {'min': min(values), 'median': median(values), 'max': max(values)}
        if a.uart:
            with a.uart.open('rb') as log:
                log.seek(uart_start)
                report['capture'] = capture_audit(log.read(), a.capture_backlog_bytes, a.seconds - 12)
            if not report['capture']['passed']:
                raise RuntimeError('Idle capture byte conservation, overflow, cadence or coverage check failed')
        report['completed'] = True
    except Exception as error:
        report['failure'] = type(error).__name__
        raise
    finally:
        with a.output.open('x') as f:
            f.write(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'completed': True, 'samples': len(samples), 'heap_bytes': report['heap_summary_bytes']}))


if __name__ == '__main__':
    main()
