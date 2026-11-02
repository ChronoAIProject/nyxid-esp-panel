#!/usr/bin/env python3
"""Verify real WHIP reception and Nyx responsiveness across Camera transitions.

Requires a logged-in nyxid CLI, an ffmpeg executable and a MediaMTX receiver.
Temporarily changes the publisher configuration and restores it on exit.
Artifacts contain camera images; keep the output directory private.
"""
import argparse
import json
from pathlib import Path
import subprocess
import time
import urllib.request


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--ip', required=True)
    p.add_argument('--slug', required=True)
    p.add_argument('--whip', required=True)
    p.add_argument('--rtsp', required=True)
    p.add_argument('--receiver-api', required=True, help='MediaMTX path status URL')
    p.add_argument('--ffmpeg', required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--seconds', type=int, default=120)
    a = p.parse_args()
    a.output_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    results = []

    def record(name, value):
        results.append({'check': name, 'result': value})
        (a.output_dir / 'results.json').write_text(json.dumps(results, indent=2))
        print(name, value, flush=True)

    def get(url, body=None):
        data = None if body is None else json.dumps(body).encode()
        req = urllib.request.Request(url, data=data, headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(req, timeout=10) as r:
            return json.load(r)

    def lan(path, body=None):
        return get('http://' + a.ip + path, body)

    def nyx(path, post=False):
        cmd = ['nyxid', 'proxy', 'request', a.slug, path, '--output', 'json']
        if post:
            cmd += ['-m', 'POST', '-d', '']
        r = subprocess.run(cmd, capture_output=True, timeout=35)
        if r.returncode:
            raise RuntimeError(r.stderr.decode(errors='replace')[:500])
        return json.loads(r.stdout)

    def wait_for(predicate, seconds=30):
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            if predicate():
                return
            time.sleep(.5)
        raise TimeoutError('Expected media state did not appear')

    def decode(label):
        out = a.output_dir / (label + '.jpg')
        cmd = [a.ffmpeg, '-hide_banner', '-loglevel', 'warning', '-rtsp_transport', 'tcp',
               '-i', a.rtsp, '-frames:v', '1', '-y', str(out)]
        r = subprocess.run(cmd, capture_output=True, timeout=25)
        (a.output_dir / (label + '-ffmpeg.log')).write_bytes(r.stderr)
        assert r.returncode == 0 and out.stat().st_size > 0, 'Receiver could not decode H.264'
        record(label, {'jpeg_bytes': out.stat().st_size})

    previous = lan('/config/stream')
    initial = lan('/status')
    record('Initial state', initial)
    try:
        lan('/stream/stop', {})
        wait_for(lambda: lan('/stream/status')['state'] == 'idle')
        lan('/config/stream', {'target_whip_url': a.whip, 'auto_start': False,
                               'fps': 15, 'bitrate': 800000})
        lan('/stream/start', {})
        wait_for(lambda: get(a.receiver_api)['ready'])
        decode('initial-receiver-frame')
        start = time.monotonic()
        transitioned = False
        while time.monotonic() - start < a.seconds:
            elapsed = round(time.monotonic() - start, 2)
            before = time.monotonic()
            state = nyx('/status')
            latency = time.monotonic() - before
            local = lan('/status')
            assert state['nyx']['state'] == 'up'
            assert local['uptime_s'] >= initial['uptime_s'], 'Board rebooted'
            receiver = get(a.receiver_api)
            assert receiver['ready'], 'WHIP receiver lost publisher'
            record('Publishing control request', {'elapsed_s': elapsed, 'nyx_seconds': round(latency, 3),
                   'stream': state['stream']['state'], 'heap': state['heap'],
                   'receiver_bytes': receiver.get('bytesReceived', receiver.get('inboundBytes')),
                   'receiver_errors': receiver.get('inboundFramesInError')})
            if not transitioned and elapsed >= a.seconds / 2:
                nyx('/lcd/screen?target=camera', True)
                wait_for(lambda: lan('/status')['media']['mode'] == 'preview')
                with urllib.request.urlopen('http://' + a.ip + '/snapshot.jpg', timeout=10) as r:
                    (a.output_dir / 'preview.jpg').write_bytes(r.read())
                record('Camera preview while publisher suspended', lan('/status')['media'])
                nyx('/lcd/screen?target=back', True)
                wait_for(lambda: lan('/status')['media']['mode'] == 'publish')
                wait_for(lambda: get(a.receiver_api)['ready'])
                decode('resumed-receiver-frame')
                transitioned = True
            time.sleep(3)
        decode('sustained-receiver-frame')
        record('PASS', {'seconds': round(time.monotonic() - start, 1),
                        'preview_transition': transitioned, 'final': lan('/status')})
    finally:
        lan('/stream/stop', {})
        wait_for(lambda: lan('/stream/status')['state'] == 'idle')
        lan('/config/stream', previous)
        nyx('/lcd/screen?target=home', True)


if __name__ == '__main__':
    main()
