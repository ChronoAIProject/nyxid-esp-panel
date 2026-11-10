"""Capture explicitly enabled development-panel UDP diagnostics into private files."""
import argparse
import json
from pathlib import Path
import signal
import socket
import struct
import time

HEADER = struct.Struct('<6IQ2iI')
SIZES = {0: HEADER.size + 1408, 1: HEADER.size + 1408, 2: HEADER.size + 96}


def parse(packet):
    if len(packet) < HEADER.size:
        raise ValueError('Truncated trace header')
    header = HEADER.unpack_from(packet)
    magic, instance, step, part, flags, crc, started, dense, temporal, numerator = header
    if (magic != 0x3154584e or not instance or step < 1 or part not in SIZES
            or len(packet) != SIZES[part] or flags & ~3
            or not -1 <= dense <= 255 or not -1 <= temporal <= 255
            or numerator > 1530 or bool(flags & 1) != (step <= 30)):
        raise ValueError('Invalid trace protocol')
    return header, packet[HEADER.size:]


def records(path):
    with path.open('rb') as stream:
        while prefix := stream.read(2):
            if len(prefix) != 2:
                raise ValueError('Truncated trace framing')
            size = struct.unpack('<H', prefix)[0]
            packet = stream.read(size)
            if len(packet) != size:
                raise ValueError('Truncated trace record')
            yield parse(packet)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bind', default='0.0.0.0')
    p.add_argument('--port', type=int, default=30282)
    p.add_argument('--source-ip', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--seconds', type=float, default=300)
    args = p.parse_args()
    if not 0 < args.seconds <= 600 or not 1024 <= args.port <= 65535:
        p.error('Use a 1..600second capture and a port in1024..65535')
    args.output.mkdir(parents=True, exist_ok=False, mode=0o700)
    args.output.chmod(0o700)
    stopping = False

    def stop(*_):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4 * 1024 * 1024)
    sock.bind((args.bind, args.port))
    sock.settimeout(.5)
    started = time.monotonic()
    count = 0
    instances = set()
    errors = []
    (args.output / 'ready.json').write_text(json.dumps({'ready': True}))
    try:
        with (args.output / 'packets.bin').open('xb') as out:
            while not stopping and time.monotonic() - started < args.seconds:
                try:
                    packet, source = sock.recvfrom(4096)
                except socket.timeout:
                    continue
                if source[0] != args.source_ip:
                    continue
                try:
                    header, _ = parse(packet)
                except ValueError as error:
                    errors.append(str(error))
                    continue
                out.write(struct.pack('<H', len(packet)) + packet)
                count += 1
                instances.add(header[1])
    finally:
        sock.close()
        report = {'seconds': time.monotonic() - started, 'packets': count,
                  'instances': sorted(instances), 'errors': errors,
                  'scope': 'Private diagnostic copy only; contains microphone audio'}
        (args.output / 'capture.json').write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(report), flush=True)
    if errors or not count:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
