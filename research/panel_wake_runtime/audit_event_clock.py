"""Audit sample/timer markers from a single boot of a raw-input development arm."""
import argparse
import hashlib
import json
from pathlib import Path
import re

BEGIN = ('epoch', 'start_us', 'read_chunk', 'detect_chunk', 'rx_bytes', 'rx_overflows')
EVENT = ('epoch', 'sample_end', 'read_end', 'read_us', 'detect_start_us',
         'detect_end_us', 'rx_bytes', 'rx_overflows')
PREFIX = b'panel.wake.clock: '
LINES = {kind: re.compile(kind + b' ' + b' '.join(k.encode() + rb'=(\d+)' for k in fields))
         for kind, fields in ((b'begin', BEGIN), (b'event', EVENT))}


def audit(log, backlog_bound):
    epochs, event_count = {}, 0
    for line in log.splitlines():
        if PREFIX not in line:
            continue
        payload = line.split(PREFIX, 1)[1].split(b'\x1b', 1)[0].strip()
        kind = payload.split(b' ', 1)[0]
        match = LINES[kind].fullmatch(payload) if kind in LINES else None
        if not match:
            raise ValueError('Malformed or truncated event-clock marker')
        row = dict(zip(BEGIN if kind == b'begin' else EVENT, map(int, match.groups())))
        epoch = row['epoch']
        if kind == b'begin':
            if (epoch < 1 or (epochs and epoch != next(reversed(epochs)) + 1)
                    or not (0 < row['read_chunk'] <= 1024 and 0 < row['detect_chunk'] <= 2048)):
                raise ValueError('Repeated epoch, lost marker or unexpected chunk geometry')
            epochs[epoch] = {'begin': row, 'event': None}
            continue
        if not epochs or next(reversed(epochs)) != epoch or epochs[epoch]['event'] is not None:
            raise ValueError('Missing begin marker, stale epoch or duplicated event')
        begin = epochs[epoch]['begin']
        read, end = row['read_end'], row['sample_end']
        if (not 0 < end <= read or read % begin['read_chunk'] or end % begin['detect_chunk']
                or read - end >= begin['read_chunk']
                or not begin['start_us'] <= row['read_us'] <= row['detect_start_us'] <= row['detect_end_us']):
            raise ValueError('Detector sample position or local timer sequence inconsistent')
        dma_delta = (row['rx_bytes'] - begin['rx_bytes']) % (1 << 32)
        new_overflows = (row['rx_overflows'] - begin['rx_overflows']) % (1 << 32)
        balance = dma_delta - 4 * read
        epochs[epoch]['event'] = {**row, 'new_rx_overflows': new_overflows, 'dma_balance_bytes': balance,
                                  'sample_clock_usable': not new_overflows and abs(balance) <= backlog_bound,
                                  'detector_wall_us': row['detect_end_us'] - row['detect_start_us'],
                                  'read_to_return_us': row['detect_end_us'] - row['read_us']}
        event_count += 1
    if not epochs:
        raise ValueError('No event-clock markers found')
    return {'epoch_count': len(epochs), 'event_count': event_count, 'epochs': list(epochs.values())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--uart', required=True, type=Path)
    parser.add_argument('--backlog-bound-bytes', required=True, type=int)
    parser.add_argument('--require-event', action='store_true')
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if args.backlog_bound_bytes < 1 or args.output.exists():
        parser.error('Use a positive DMA bound and a fresh report')
    log = args.uart.read_bytes()
    result = audit(log, args.backlog_bound_bytes)
    result['uart_sha256'] = hashlib.sha256(log).hexdigest()
    result['latency_limits'] = (
        'Board timer measures detector execution, not acoustic delay. For an acoustic latency '
        'interval independently locate the keyword end in the same raw microphone sample clock. '
        'An asynchronous detector may return a decision for an earlier submitted block.')
    with args.output.open('x') as handle:
        json.dump(result, handle, indent=2)
        handle.write('\n')
    print(json.dumps({k: result[k] for k in ('epoch_count', 'event_count')}))
    if args.require_event and (not result['event_count'] or any(
            row['event'] and not row['event']['sample_clock_usable'] for row in result['epochs'])):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
