"""Measure an idle experimental full app without changing its configuration.

Run inside run_full_app.py; that runner owns USB, UART and app restoration.
This gate measures resources/scheduling, not wake recognition or acoustic quality.
"""
import argparse
import json
from pathlib import Path
import re
import time
import urllib.request


def diagnostics(path, capture_backlog_bytes=None):
    lines = path.read_text(errors='replace') if path.exists() else ''
    pattern = (r'I \((\d+)\) panel\.wake\.embedding.*step=(\d+) mean_us=(\d+) p50_us=(\d+) '
               r'p95_us=(\d+) max_us=(\d+) over80ms=(\d+)')
    timings = [dict(zip(('board_ms', 'step', 'mean_us', 'p50_us', 'p95_us', 'max_us', 'over80ms'),
                       map(int, values))) for values in re.findall(pattern, lines)]
    stages = (r'stages step=(\d+) frontend_us=(\d+) backbone_us=(\d+) '
              r'dense_us=(\d+) temporal_us=(\d+) core=(\d+)')
    cadence = [{'steps': b['step'] - a['step'], 'elapsed_ms': b['board_ms'] - a['board_ms'],
                'expected_ms': (b['step'] - a['step']) * 80} for a, b in zip(timings, timings[1:])]
    elapsed_ms = sum(r['elapsed_ms'] for r in cadence)
    expected_ms = sum(r['expected_ms'] for r in cadence)
    window_passed = bool(cadence) and all(
        r['steps'] > 0 and r['elapsed_ms'] <= r['expected_ms'] * 1.03 for r in cadence)
    # A small allowance per window must not admit persistent producer drift.
    aggregate_passed = expected_ms > 0 and elapsed_ms <= expected_ms * 1.01
    capture = [dict(zip(('step', 'rx_bytes', 'rx_overflows'), map(int, values)))
               for values in re.findall(r'capture step=(\d+) rx_bytes=(\d+) rx_overflows=(\d+)', lines)]
    capture_windows = [{'steps': b['step'] - a['step'], 'rx_bytes': b['rx_bytes'] - a['rx_bytes'],
                        'overflows': b['rx_overflows'] - a['rx_overflows']}
                       for a, b in zip(capture, capture[1:])]
    capture_balance = [{'steps': b['step'] - capture[0]['step'],
                        'unaccounted_bytes': b['rx_bytes'] - capture[0]['rx_bytes'] -
                            (b['step'] - capture[0]['step']) * 1280 * 4}
                       for b in capture[1:]]
    capture_queue_passed = bool(capture_windows) and all(
        r['steps'] > 0 and r['rx_bytes'] > 0 and r['overflows'] == 0 for r in capture_windows)
    capture_balance_passed = bool(capture_balance) and capture_backlog_bytes is not None and all(
        abs(r['unaccounted_bytes']) <= capture_backlog_bytes for r in capture_balance)
    return {'timings': timings,
            'stages': [dict(zip(('step', 'frontend_us', 'backbone_us', 'dense_us', 'temporal_us', 'core'),
                               map(int, values))) for values in re.findall(stages, lines)],
            'cadence_windows': cadence,
            'queue_windows': [dict(zip(('consumed', 'highwater', 'max_delay_us', 'mean_delay_us'), map(int, values)))
                              for values in re.findall(r'queue consumed=(\d+) highwater=(\d+) max_delay_us=(\d+) mean_delay_us=(\d+)', lines)],
            'inference_margin_passed': len(timings) >= 2 and all(
                r['mean_us'] < 80000 and r['p95_us'] < 80000 for r in timings),
            'pipeline_window_cadence_passed': window_passed,
            'pipeline_aggregate_cadence_passed': aggregate_passed,
            'pipeline_aggregate': {'elapsed_ms': elapsed_ms, 'expected_ms': expected_ms},
            'pipeline_cadence_passed': window_passed and aggregate_passed,
            'capture_counters': capture,
            'capture_windows': capture_windows,
            'capture_balance': capture_balance,
            'capture_backlog_bound_bytes': capture_backlog_bytes,
            'capture_queue_passed': capture_queue_passed,
            'capture_balance_passed': capture_balance_passed,
            'capture_loss_passed': capture_queue_passed and capture_balance_passed,
            'uart_failure': any(marker in lines for marker in ('Guru Meditation', 'abort()',
                'ordered detector queue overflow', 'ordered detector sequence/PCM integrity failure'))}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--status-url', required=True)
    p.add_argument('--expected-app-sha256', required=True)
    p.add_argument('--uart', type=Path, required=True)
    p.add_argument('--seconds', type=int, default=30)
    p.add_argument('--require-capture-counters', action='store_true')
    p.add_argument('--capture-backlog-bytes', type=int,
                   help='Verified total capacity: DMA + current capture block + optional model FIFO')
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.seconds < 20 or args.output.exists():
        p.error('Measure at least 20 seconds and use a fresh output file')
    if args.require_capture_counters and (args.capture_backlog_bytes is None or args.capture_backlog_bytes <= 0):
        p.error('Hardware loss acceptance requires the verified capture backlog bound')
    report = {'scope': 'Idle whole-app resources/scheduling only; no recognition acceptance',
              'states': [], 'timings': [], 'passed': False}
    started = time.monotonic()
    try:
        while time.monotonic() - started < args.seconds:
            with urllib.request.urlopen(args.status_url, timeout=5) as response:
                value = json.load(response)
            voice = value['voice']
            if (value['app_sha256'] != args.expected_app_sha256
                    or value['nyx']['state'] != 'up' or voice['busy']
                    or not voice['wake']['listening'] or voice['wake']['word'] != 'Hey Nyx'):
                raise AssertionError('App identity, connection or idle wake state changed')
            if report['states'] and value['uptime_s'] < report['states'][-1]['uptime_s']:
                raise AssertionError('App restarted during measurement')
            report['states'].append({'seconds': round(time.monotonic() - started, 3),
                                     'uptime_s': value['uptime_s'], 'heap': value['heap']})
            time.sleep(1)
        report.update(diagnostics(args.uart, args.capture_backlog_bytes))
        if len(report['timings']) < 2:
            raise AssertionError('Fewer than two complete 100-step timing windows')
        if not report['inference_margin_passed']:
            raise AssertionError('Inference does not fit the 80 ms audio cadence with scheduling margin')
        if not report['pipeline_cadence_passed']:
            raise AssertionError('Whole microphone pipeline falls behind the incoming audio cadence')
        if args.require_capture_counters and not report['capture_loss_passed']:
            raise AssertionError('Hardware RX counters missing, queue overflow or audio byte balance exceeds capacity')
        if report['uart_failure']:
            raise AssertionError('Firmware or ordered detector queue failure reported in UART')
        report['passed'] = True
    except Exception as error:
        report['failure'] = str(error)
        raise
    finally:
        # Preserve diagnostics even when the HTTP state fails before the end.
        if not report['timings']:
            report.update(diagnostics(args.uart, args.capture_backlog_bytes))
        report['elapsed_seconds'] = time.monotonic() - started
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps({k: v for k, v in report.items() if k != 'states'}), flush=True)


if __name__ == '__main__':
    main()
