#!/usr/bin/env python3
"""Verify on-demand camera capture through NyxID against a real panel.

Requires a logged-in nyxid CLI and Pillow. Saves private camera images, checks
idle release and concurrent calls, then opens/closes the native Camera once.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import io
import json
from pathlib import Path
import subprocess
import time
import urllib.error
import urllib.request
from PIL import Image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ip', required=True)
    parser.add_argument('--slug', required=True)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--count', type=int, default=8)
    parser.add_argument('--nyx-timeout', type=float, default=15,
                        help='Per-request CLI deadline in seconds (default: 15)')
    args = parser.parse_args()
    if not 2 <= args.count <= 50:
        parser.error('--count must be 2..50')
    if not 0 < args.nyx_timeout <= 60:
        parser.error('--nyx-timeout must be >0 and <=60')
    args.output_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    results = []

    def record(name, value):
        results.append({'check': name, 'result': value})
        (args.output_dir / 'results.json').write_text(json.dumps(results, indent=2))
        print(name, value, flush=True)

    def lan(path, expected=200, data=None):
        try:
            r = urllib.request.urlopen(urllib.request.Request('http://' + args.ip + path, data=data), timeout=10)
        except urllib.error.HTTPError as e:
            r = e
        assert r.code == expected, (path, r.code)
        return r.read()

    def nyx(path, data=None, expect_ok=True):
        cmd = ['nyxid', 'proxy', 'request', args.slug, path, '--stream', '--output', 'json']
        # --stream preserves raw JPEG bytes on stdout; this remains one bounded
        # HTTP request. It does not ask the panel to stream video.
        if data is not None:
            cmd += ['-m', 'POST', '-d', '-']
        started = time.monotonic()
        r = subprocess.run(cmd, input=data, capture_output=True, timeout=args.nyx_timeout)
        if expect_ok:
            assert r.returncode == 0, (path, r.stdout[:160], r.stderr[:160])
        return r, time.monotonic() - started

    def status():
        return json.loads(lan('/status'))

    def wait_mode(mode):
        end = time.monotonic() + 20
        while time.monotonic() < end:
            s = status()
            if s['media']['mode'] == mode:
                return s
            time.sleep(.2)
        raise AssertionError(('camera mode did not settle', mode, s['media']))

    def check_jpeg(raw, name):
        im = Image.open(io.BytesIO(raw))
        im.load()
        assert im.size == (800, 1280), im.size
        p = args.output_dir / name
        p.write_bytes(raw)
        p.chmod(0o600)
        return len(raw)

    initial = wait_mode('idle')
    assert initial['stream']['state'] == 'disabled'
    record('Initial idle status', initial)
    record('Nyx request deadline seconds', args.nyx_timeout)
    for path in ['/camera', '/camera/snapshot.jpg', '/openapi.json']:
        lan(path, 404)
    lan('/snapshot.jpg', 503)
    lan('/stream/start', 410, b'')
    lan('/config/stream', 410, b'{"auto_start":true}')
    schema = json.loads(nyx('/openapi.json')[0].stdout)
    assert schema['paths']['/camera/snapshot.jpg']['get']['operationId'] == 'camera_take_photo'
    profile = json.loads(nyx('/camera')[0].stdout)
    assert (profile['width'], profile['height']) == (800, 1280), profile
    assert profile['mode'] == 'idle' and profile['continuous_streaming'] is False
    record('Authenticated camera routes and retired stream routes', True)

    latencies = []
    baseline = None
    with ThreadPoolExecutor(max_workers=3) as pool:
        for i in range(args.count):
            before = status()
            photo = pool.submit(nyx, '/camera/snapshot.jpg')
            control = pool.submit(nyx, '/status')
            r, elapsed = photo.result()
            control_result, control_s = control.result()
            assert json.loads(control_result.stdout)['nyx']['state'] == 'up'
            size = check_jpeg(r.stdout, f'remote-{i}.jpg')
            after = wait_mode('idle')
            assert after['media']['captures'] == before['media']['captures'] + 1
            assert after['media']['failed_captures'] == before['media']['failed_captures']
            assert after['lcd'] == before['lcd'], 'Remote capture navigated the LCD'
            latencies.append(elapsed)
            record(f'Capture {i + 1}', {'seconds': round(elapsed, 3), 'control_seconds': round(control_s, 3),
                   'board_ms': after['media']['last_capture_ms'], 'bytes': size, 'heap': after['heap']})
            if i == 1:
                baseline = after['heap']
        before = status()
        calls = [pool.submit(nyx, '/camera/snapshot.jpg', expect_ok=False) for _ in range(3)]
        successes = busy = 0
        for i, call in enumerate(calls):
            r, _ = call.result()
            if r.returncode == 0:
                check_jpeg(r.stdout, f'concurrent-{i}.jpg')
                successes += 1
            else:
                err = json.loads(r.stderr or r.stdout)
                assert err['status'] == 503, err
                busy += 1
        after = wait_mode('idle')
        assert after['media']['captures'] == before['media']['captures'] + successes
        assert successes >= 1 and successes + busy == 3
        record('Concurrent captures are serialized or rejected busy', {'success': successes, 'busy': busy})

    nyx('/lcd/screen?target=camera', b'')
    try:
        wait_mode('preview')
        check_jpeg(lan('/snapshot.jpg'), 'native-lan.jpg')
        check_jpeg(nyx('/camera/snapshot.jpg')[0].stdout, 'native-nyx.jpg')
        assert status()['media']['mode'] == 'preview', 'Snapshot closed the native preview'
    finally:
        nyx('/lcd/screen?target=back', b'')
    final = wait_mode('idle')
    assert final['uptime_s'] >= initial['uptime_s'], 'Board rebooted'
    assert final['nyx']['state'] == 'up'
    # Network allocations can fluctuate. Large capture buffers must not persist.
    assert final['heap']['free'] >= baseline['free'] - 128 * 1024, (baseline, final['heap'])
    assert final['heap']['internal_free'] >= baseline['internal_free'] - 16 * 1024
    record('Native preview survives capture and Back releases resources', True)
    record('Remote latency', {'min': round(min(latencies), 3), 'max': round(max(latencies), 3)})
    record('Final idle status', final)


if __name__ == '__main__':
    main()
