#!/usr/bin/env python3
"""Hardware persistence checks. Reboots twice and restores the built-in background.

Requires a logged-in nyxid CLI, Pillow and a board whose current custom default
may be replaced/deleted. Artifacts contain framebuffer screenshots, not secrets.
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
    parser.add_argument('--image', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    results = []

    def record(name, value=True):
        results.append({'check': name, 'result': value})
        (args.output_dir / 'results.json').write_text(json.dumps(results, indent=2))
        print(name, value, flush=True)

    def nyx(path, method='GET', data=None, content_type='image/jpeg', ok=True):
        cmd = ['nyxid', 'proxy', 'request', args.slug, path, '-m', method, '--output', 'json']
        if data is not None:
            cmd += ['-H', 'Content-Type:' + content_type, '-d', '-']
        result = subprocess.run(cmd, input=data, capture_output=True, timeout=45)
        assert (result.returncode == 0) == ok, (path, result.stdout[:200], result.stderr[:200])
        return json.loads(result.stdout) if ok else result

    def lan(path, method='GET', data=None, expected=200):
        req = urllib.request.Request('http://' + args.ip + path, method=method, data=data)
        try:
            result = urllib.request.urlopen(req, timeout=25)
        except urllib.error.HTTPError as error:
            result = error
        assert result.code == expected, (path, result.code)
        return result.read()

    def screenshot(name):
        time.sleep(.6)
        raw = lan('/lcd/snapshot.jpg')
        im = Image.open(io.BytesIO(raw)).convert('RGB')
        assert im.size == (800, 1280)
        (args.output_dir / (name + '.jpg')).write_bytes(raw)
        return im.tobytes()

    def reboot():
        before = json.loads(lan('/status'))['uptime_s']
        nyx('/system/reboot', 'POST')
        deadline = time.monotonic() + 100
        while time.monotonic() < deadline:
            time.sleep(2)
            try:
                state = json.loads(lan('/status'))
                if state['uptime_s'] < before and state['nyx']['state'] == 'up' and state.get('time_valid', True):
                    assert state['lcd']['mode'] == 'home', state['lcd']
                    return state
            except (OSError, ValueError):
                pass
        raise AssertionError('Board failed to boot into Nyx Panel / reconnect')

    state = json.loads(lan('/status'))
    assert state['lcd']['mode'] == 'home'
    record('Boot opens Nyx Panel', state['fw'])
    # The following operations deliberately replace the board's custom default.
    nyx('/lcd/default-image', 'DELETE')
    nyx('/lcd/clear', 'POST')
    factory = screenshot('builtin')
    assert nyx('/lcd/default-image')['source'] == 'builtin'
    record('Built-in fallback')
    for method in ['GET', 'POST', 'DELETE']:
        lan('/lcd/default-image', method, b'' if method == 'POST' else None, expected=404)
    record('Persistent configuration excluded from LAN')

    jpeg = args.image.read_bytes()
    nyx('/lcd/default-image', 'POST', jpeg)
    info = nyx('/lcd/default-image')
    assert info['source'] == 'stored' and info['persistent'] and info['jpeg_bytes'] == len(jpeg), info
    stored = screenshot('stored')
    assert stored != factory
    record('Remote upload persisted', info)

    nyx('/lcd/default-image', 'POST', b'invalid JPEG', ok=False)
    nyx('/lcd/default-image', 'POST', b'x' * 65537, ok=False)
    nyx('/lcd/default-image', 'POST', jpeg, content_type='application/json', ok=False)
    oversized = io.BytesIO()
    Image.new('RGB', (801, 1280), 'red').save(oversized, 'JPEG')
    nyx('/lcd/default-image', 'POST', oversized.getvalue(), ok=False)
    assert nyx('/lcd/default-image') == info
    assert screenshot('after-invalid') == stored
    record('Invalid, oversized and wrong-type uploads preserve old image')

    # Match native Settings' NVS -> flash order while replacing a large blob.
    # A flash -> NVS inversion deadlocks these operations against one another.
    cfg = nyx('/config/lcd')
    with ThreadPoolExecutor(max_workers=2) as pool:
        for _ in range(3):
            saving = pool.submit(nyx, '/lcd/default-image', 'POST', jpeg)
            settings = pool.submit(nyx, '/config/lcd', 'POST', json.dumps(cfg).encode(), 'application/json')
            saving.result()
            settings.result()
    assert screenshot('after-concurrent-nvs') == stored
    record('Concurrent background and Settings NVS writes complete')

    transient = io.BytesIO()
    Image.new('RGB', (800, 1280), '#5A2AF1').save(transient, 'JPEG')
    nyx('/lcd/image?duration_ms=5000', 'POST', transient.getvalue())
    pushed = screenshot('temporary')
    assert pushed != stored
    time.sleep(5)
    assert screenshot('expired') == stored
    record('TTL restores persistent fallback')
    nyx('/lcd/image', 'POST', transient.getvalue())
    pushed = screenshot('sticky-temporary')
    nyx('/lcd/default-image', 'DELETE')
    assert screenshot('reset-under-temporary') == pushed
    nyx('/lcd/default-image', 'POST', jpeg)
    assert screenshot('set-under-temporary') == pushed
    nyx('/lcd/clear', 'POST')
    assert screenshot('cleared') == stored
    record('Set/reset keep temporary image; clear restores fallback')

    reboot()
    assert nyx('/lcd/default-image') == info
    assert screenshot('after-reboot') == stored
    record('Stored image survives reboot; boot opens panel')
    nyx('/lcd/default-image', 'DELETE')
    assert screenshot('reset') == factory
    reboot()
    assert nyx('/lcd/default-image')['source'] == 'builtin'
    assert screenshot('reset-reboot') == factory
    state = json.loads(lan('/status'))
    assert state['stream']['state'] == 'disabled' and state['media']['mode'] == 'idle'
    record('Reset survives reboot; camera remains idle', state['heap'])


if __name__ == '__main__':
    main()
