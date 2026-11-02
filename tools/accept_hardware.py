#!/usr/bin/env python3
"""Exercise display and native Camera against a real board and Nyx service.

Requires a logged-in nyxid CLI and Pillow. Changes the displayed content and
opens/closes Camera three times. Artifacts may contain the camera's surroundings.
"""
import argparse
import io
import json
from pathlib import Path
import subprocess
import time
import urllib.error
import urllib.request

from PIL import Image


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--ip', required=True)
    p.add_argument('--slug', required=True)
    p.add_argument('--image', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--camera', action='store_true')
    a = p.parse_args()
    a.output_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    results = []

    def record(name, value):
        results.append({'check': name, 'result': value})
        (a.output_dir / 'results.json').write_text(json.dumps(results, indent=2))
        print(name, value, flush=True)

    def lan(path, data=None, expected=200):
        req = urllib.request.Request('http://' + a.ip + path, data=data,
                                     headers={'Content-Type': 'image/jpeg'})
        try:
            r = urllib.request.urlopen(req, timeout=30)
        except urllib.error.HTTPError as e:
            r = e
        assert r.code == expected, (path, r.code)
        try:
            body = r.read()
        except ConnectionResetError:
            # An early rejection can close while request bytes are still in
            # flight. The HTTP error status must already have been received.
            if expected < 400:
                raise
            body = b''
        return body

    def nyx(path, data=None, json_body=None, expected_ok=True):
        cmd = ['nyxid', 'proxy', 'request', a.slug, path, '--output', 'json']
        if data is not None or json_body is not None:
            cmd += ['-m', 'POST', '-H', 'Content-Type:' +
                    ('application/json' if json_body is not None else 'image/jpeg'), '-d', '-']
        body = json.dumps(json_body, ensure_ascii=False).encode() if json_body is not None else data
        r = subprocess.run(cmd, input=body, capture_output=True, timeout=45)
        assert (r.returncode == 0) == expected_ok, (path, r.stdout[:160], r.stderr[:160])
        return r.stdout

    def snapshot(name, path='/lcd/snapshot.jpg'):
        raw = lan(path)
        im = Image.open(io.BytesIO(raw))
        im.load()
        out = a.output_dir / name
        out.write_bytes(raw)
        out.chmod(0o600)
        return raw, im.size

    state = json.loads(nyx('/status'))
    assert state['nyx']['state'] == 'up'
    record('Nyx status', {'fw': state['fw'], 'ota': state['ota']})
    lan('/lcd/clear', b'', 202)
    nyx('/lcd/image', a.image.read_bytes())
    nyx('/lcd/text', json_body={'text': 'Hello Lexa · 你好，世界', 'duration_ms': 0})
    time.sleep(1)
    baseline, size = snapshot('display.jpg')
    record('Nyx JPEG/text and LCD JPEG decode', size)
    lan('/lcd/image', b'invalid JPEG', 400)
    lan('/lcd/image', b'x' * 65537, 413)
    unchanged, _ = snapshot('after-rejected-images.jpg')
    assert unchanged == baseline, 'Rejected image altered the display'
    record('Invalid/oversized images preserve current content', True)
    nyx('/lcd/text', json_body={'text': 'Expires after 2 seconds', 'duration_ms': 2000})
    time.sleep(.4)
    temporary, _ = snapshot('temporary-text.jpg')
    assert temporary != baseline
    time.sleep(2)
    expired, _ = snapshot('expired-text.jpg')
    assert expired == baseline, 'Expiry did not restore the original rendered content'
    record('Text expiry restores content', True)
    nyx('/lcd/screen?target=unknown', b'', expected_ok=False)
    record('Unknown native target rejected', True)
    if a.camera:
        try:
            for cycle in range(1, 4):
                nyx('/lcd/screen?target=camera', b'')
                end = time.monotonic() + 25
                while time.monotonic() < end:
                    state = json.loads(lan('/status'))
                    if state['media']['mode'] == 'preview':
                        break
                    time.sleep(.5)
                else:
                    raise AssertionError('Camera did not enter preview')
                time.sleep(1)
                _, size = snapshot(f'camera-{cycle}.jpg', '/snapshot.jpg')
                snapshot(f'camera-screen-{cycle}.jpg')
                nyx('/lcd/screen?target=back', b'')
                end = time.monotonic() + 20
                while time.monotonic() < end:
                    state = json.loads(lan('/status'))
                    if state['media']['mode'] == 'publish':
                        break
                    time.sleep(.5)
                else:
                    raise AssertionError('Camera close did not release preview')
                record(f'Camera open/snapshot/back cycle {cycle}', size)
        finally:
            nyx('/lcd/screen?target=home', b'')
    record('Final status', json.loads(lan('/status')))


if __name__ == '__main__':
    main()
