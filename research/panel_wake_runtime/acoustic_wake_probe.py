"""Frozen synthetic speaker-to-panel wake pilot, with owned-call cleanup.

Measures acoustic wake events and rearming, not conversational acceptance.
No audio injection, microphone recording, threshold fitting or device key use
by the host. Source PCM, playback device and volume stay unchanged.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import signal
import subprocess
import time
import urllib.request


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def default_output():
    value = json.loads(subprocess.check_output(
        ['/usr/sbin/system_profiler', 'SPAudioDataType', '-json'], timeout=15))
    found = [item for group in value['SPAudioDataType'] for item in group.get('_items', [])
             if item.get('coreaudio_default_audio_output_device') == 'spaudio_yes']
    if len(found) != 1:
        raise RuntimeError('Expected one known default playback device')
    item = found[0]
    return {k: item.get(k) for k in ('_name', 'coreaudio_device_transport',
                                    'coreaudio_device_srate', 'coreaudio_device_output')}


def verify_target(value, app, word, slug):
    if (value['app_sha256'] != app or value['nyx']['slug'] != slug
            or value['nyx']['state'] != 'up' or value['ota']['busy']
            or value['voice']['wake']['word'] != word):
        raise ValueError('Explicit development app, service or wake word differs')


def event_window(before, current, previous_poll_start, current_poll_end, onset, endpoint):
    """Conservative HTTP observation interval; not sample-exact model timing."""
    change = current['wake']['detections'] - before['wake']['detections']
    if change not in (0, 1):
        raise ValueError('Wake count reset or unexpected additional event')
    if not change:
        return None
    return {'observed_interval_s': [previous_poll_start, current_poll_end],
            'before_source_onset': current_poll_end < onset,
            'onset_straddled': previous_poll_start < onset <= current_poll_end,
            'delay_from_estimated_keyword_end_ms': [
                (previous_poll_start - endpoint) * 1000,
                (current_poll_end - endpoint) * 1000] if endpoint is not None else None}


def main():
    import soundfile as sf
    from acoustic_voice_smoke import mac_output

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--arm', choices=('hi_esp', 'hey_nyx'), required=True)
    p.add_argument('--source-root', type=Path, required=True)
    p.add_argument('--status-url', required=True)
    p.add_argument('--expected-app-sha256', required=True)
    p.add_argument('--expected-slug', required=True)
    p.add_argument('--admin-module', type=Path, required=True)
    p.add_argument('--admin-base', required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    plan = json.loads(args.plan.read_text())
    if (sha(Path(__file__)) != plan['script_sha256'] or args.output.exists()
            or not args.status_url.startswith('http://') or plan['sample_rate'] != 16000
            or plan['post_playback_seconds'] != 2 or plan['stable_ready_seconds'] != 2):
        p.error('Frozen implementation/protocol and fresh private output required')
    for name, expected in plan['helpers'].items():
        if sha(Path(__file__).with_name(name)) != expected:
            raise ValueError('Playback helper differs from frozen protocol')
    arm = plan['arms'][args.arm]
    word = arm['word']
    if args.expected_app_sha256 != arm['app_elf_sha256']:
        raise ValueError('App differs from the prespecified acoustic arm')
    root = args.source_root.resolve()
    for row in arm['trials']:
        path = (root / row['path']).resolve()
        path.relative_to(root)
        info = sf.info(path)
        if (sha(path) != row['sha256'] or info.samplerate != 16000 or info.channels != 1
                or info.frames != row['samples'] or info.subtype != 'PCM_16'):
            raise ValueError('Explicit unchanged source PCM differs')
    if mac_output() != plan['mac_output'] or default_output() != plan['default_output']:
        raise ValueError('Mac playback level/mute/device differs from the fixed plan')
    spec = importlib.util.spec_from_file_location('wake_pilot_owner', args.admin_module)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.BASE = args.admin_base
    owner = module.LocalAPI()
    args.output.mkdir(parents=True)
    report = {'scope': __doc__, 'plan_sha256': sha(args.plan), 'arm': args.arm,
        'app_elf_sha256': args.expected_app_sha256, 'trials': [], 'complete': False,
        'accepted': False, 'latency_limits': 'HTTP intervals include playback launch, network/poll delay and approximate keyword annotation. Not sample-exact model latency.'}

    def status():
        with urllib.request.urlopen(args.status_url, timeout=5) as response:
            value = json.load(response)
        verify_target(value, args.expected_app_sha256, word, args.expected_slug)
        return value

    def ready():
        deadline, since = time.monotonic() + 45, None
        while time.monotonic() < deadline:
            value = status()
            v = value['voice']
            if not v['busy'] and not v['active'] and v['wake']['listening']:
                since = since or time.monotonic()
                if time.monotonic() - since >= plan['stable_ready_seconds']:
                    return value
            else:
                since = None
            time.sleep(.2)
        raise TimeoutError('Development panel did not return to stable idle listening')

    def interrupted(_signum, _frame):
        raise KeyboardInterrupt('Stopping owned acoustic pilot call')

    signal.signal(signal.SIGTERM, interrupted)
    try:
        for trial in arm['trials']:
            before = ready()
            if mac_output() != plan['mac_output'] or default_output() != plan['default_output']:
                raise RuntimeError('Playback conditions changed before trial')
            v0 = before['voice']
            row = {**trial, 'event': None, 'rearmed': False, 'call_stop_requested': False}
            report['trials'].append(row)
            start = time.monotonic()
            player, previous, observed = None, 0., False
            try:
                player = subprocess.Popen(['/usr/bin/afplay', str(root / trial['path'])])
                deadline = start + trial['samples'] / 16000 + plan['post_playback_seconds']
                while time.monotonic() < deadline:
                    poll_start = time.monotonic() - start
                    value = status()
                    now = time.monotonic() - start
                    v = value['voice']
                    event = event_window(v0, v, previous, now,
                                         trial['source_onset_samples'] / 16000,
                                         trial['keyword_end'] / 16000 if trial['keyword_end'] is not None else None)
                    if event is not None:
                        if not v['wake']['initiated_call'] or v['remote_initiated']:
                            raise RuntimeError('Observed wake does not own this local call')
                        row['event'], observed = event, True
                        break
                    if v['busy'] or v['active']:
                        raise RuntimeError('Unowned call started during acoustic trial')
                    previous = poll_start
                    time.sleep(.15)
                row['playback_returncode_before_cleanup'] = player.poll()
                row['elapsed_s'] = time.monotonic() - start
                row['source_playback_completed'] = player.poll() == 0
                if player.poll() not in (None, 0) or (not observed and player.poll() is None):
                    raise RuntimeError('Mac source playback failed')
            finally:
                if player:
                    if player.poll() is None:
                        player.terminate()
                    player.wait(timeout=5)
                value = status()
                v = value['voice']
                change = v['wake']['detections'] - v0['wake']['detections']
                if change == 1 and v['wake']['initiated_call'] and not v['remote_initiated'] and v['busy']:
                    owner.call('/proxy/s/' + args.expected_slug + '/voice/stop', 'POST', {}, expected=(200, 201, 202))
                    row['call_stop_requested'] = True
                elif v['busy']:
                    raise RuntimeError('Refusing to stop a call not owned by this trial')
                ready()
                row['rearmed'] = True
                row['playback_conditions_unchanged'] = (mac_output() == plan['mac_output']
                    and default_output() == plan['default_output'])
                (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
            print(json.dumps({'arm': args.arm, 'id': trial['id'], 'event': row['event'],
                              'rearmed': row['rearmed']}), flush=True)
            if not row['playback_conditions_unchanged']:
                raise RuntimeError('Playback conditions changed during trial')
        report['complete'] = True
    finally:
        (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
