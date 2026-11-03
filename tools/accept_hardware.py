#!/usr/bin/env python3
"""Exercise display and native Camera against a real board and Nyx service.

Requires a logged-in nyxid CLI and Pillow. Changes the displayed content and
opens/closes Camera three times. Artifacts may contain the camera's surroundings.
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
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--ip', required=True)
    p.add_argument('--slug', required=True)
    p.add_argument('--image', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--camera', action='store_true')
    p.add_argument('--native-pages', action='store_true',
                   help='Exercise Settings/WLAN exit persistence and native status pages')
    p.add_argument('--onboarding', action='store_true',
                   help='Exercise scan/back navigation and request then cancel a real login; never approves it')
    p.add_argument('--wifi-qr-image', type=Path,
                   help='With --onboarding, decode a Wi-Fi QR JPEG and check confirmation without connecting')
    a = p.parse_args()
    if a.wifi_qr_image and not a.onboarding:
        p.error('--wifi-qr-image requires --onboarding')
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

    def navigate(target):
        nyx('/lcd/screen?target=' + target, b'')
        end = time.monotonic() + 20
        while time.monotonic() < end:
            screen = json.loads(lan('/lcd/screen'))
            if screen['target'] == target:
                return screen
            time.sleep(.3)
        raise AssertionError((target, screen))

    state = json.loads(nyx('/status'))
    assert state['nyx']['state'] == 'up'
    initial_uptime = state['uptime_s']
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
    # Observe while Nyx is in flight: its response can arrive after a short
    # message has already expired, even though the board rendered it correctly.
    with ThreadPoolExecutor(max_workers=1) as pool:
        pushed = pool.submit(nyx, '/lcd/text', json_body={
            'text': 'Expires after 10 seconds', 'duration_ms': 10000})
        end = time.monotonic() + 45
        while time.monotonic() < end:
            temporary, _ = snapshot('temporary-text.jpg')
            if temporary != baseline:
                break
            time.sleep(.3)
        else:
            raise AssertionError('Expiring text was never observed')
        pushed.result()
    end = time.monotonic() + 20
    while time.monotonic() < end:
        expired, _ = snapshot('expired-text.jpg')
        if expired == baseline:
            break
        time.sleep(.5)
    else:
        raise AssertionError('Expiry did not restore the original rendered content')
    record('10-second text observed and expiry restores content', True)
    nyx('/lcd/screen?target=unknown', b'', expected_ok=False)
    record('Unknown native target rejected', True)
    if a.native_pages:
        for cycle in range(1, 3):
            targets = ['settings', 'wifi', 'settings', 'wifi', 'info', 'device_bind', 'home']
            for step, target in enumerate(targets, 1):
                nyx('/lcd/screen?target=' + target, b'')
                end = time.monotonic() + 20
                while time.monotonic() < end:
                    screen = json.loads(lan('/lcd/screen'))
                    if screen['target'] == target:
                        break
                    time.sleep(.5)
                else:
                    raise AssertionError((target, screen))
                time.sleep(1)
                _, size = snapshot(f'native-{cycle}-{step}-{target}.jpg')
                state = json.loads(lan('/status'))
                assert state['uptime_s'] >= initial_uptime, 'Board rebooted during native navigation'
                initial_uptime = state['uptime_s']
                record(f'Native page cycle {cycle}: {target}', {'screen': screen, 'size': size})
    if a.onboarding:
        # These routes must stay outside the unauthenticated LAN recovery API.
        lan('/nyx/login', expected=404)
        lan('/wifi/qr', expected=404)
        for origin in ['settings', 'wifi', 'device_bind']:
            navigate(origin)
            navigate('scan')
            snapshot(f'scan-from-{origin}.jpg')
            nyx('/lcd/screen?target=back', b'')
            end = time.monotonic() + 20
            while time.monotonic() < end:
                screen = json.loads(lan('/lcd/screen'))
                if screen['target'] == origin:
                    break
                time.sleep(.3)
            else:
                raise AssertionError(('scan return', origin, screen))
        record('Scan returns to its originating Settings page', True)
        if a.wifi_qr_image:
            navigate('scan')
            result = json.loads(nyx('/_dev/qr/inject_jpeg', a.wifi_qr_image.read_bytes()))
            assert result.get('confirmation_required') is True, result
            candidate = json.loads(nyx('/wifi/qr'))
            assert candidate['state'] == 'confirm' and candidate['ssid'], candidate
            assert 'password' not in candidate and 'psw' not in candidate
            # A second decode must not replace the candidate awaiting consent.
            nyx('/_dev/qr/inject_jpeg', a.wifi_qr_image.read_bytes(), expected_ok=False)
            assert json.loads(nyx('/wifi/qr')) == candidate
            snapshot('wifi-qr-confirmation.jpg')
            navigate('settings')  # Cancels; intentionally never POST /wifi/qr.
            record('Wi-Fi QR requires confirmation and rejects replacement', True)
        try:
            nyx('/nyx/login', b'')
            end = time.monotonic() + 30
            while time.monotonic() < end:
                login = json.loads(nyx('/nyx/login'))
                assert not {'device_code', 'access_token', 'refresh_token'} & login.keys()
                if login['state'] == 'waiting':
                    assert login['user_code'] and login['seconds_remaining'] > 0
                    break
                assert login['state'] == 'requesting', login['state']
                time.sleep(.5)
            else:
                raise AssertionError('No real sign-in code was delivered')
            snapshot('nyx-login-qr.jpg')
            record('Real device login waits for human approval', True)
        finally:
            nyx('/nyx/login/cancel', b'')
            navigate('settings')
        end = time.monotonic() + 35
        while time.monotonic() < end:
            login = json.loads(nyx('/nyx/login'))
            if not login['busy']:
                break
            time.sleep(.5)
        assert login['state'] == 'cancelled' and not login['busy'], login['state']
        assert not login['user_code'] and not login['verification_uri_complete']
        record('Device login cancels and removes the displayed code', True)
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
                    if state['media']['mode'] == 'idle':
                        break
                    time.sleep(.5)
                else:
                    raise AssertionError('Camera close did not release preview')
                record(f'Camera open/snapshot/back cycle {cycle}', size)
        finally:
            # Revisit Settings after Camera was started for the first time.
            # This catches subpage timers that survived a missing app pause.
            navigate('device_bind')
            snapshot('nyx-after-camera-cycles.jpg')
            navigate('settings')
            nyx('/lcd/screen?target=home', b'')
    state = json.loads(lan('/status'))
    assert state['uptime_s'] >= initial_uptime, 'Board rebooted during acceptance'
    record('Final status', state)


if __name__ == '__main__':
    main()
