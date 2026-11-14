#!/usr/bin/env python3
"""Read-only panel heap/uptime monitor; account/network configuration stays local."""
import argparse
from collections import defaultdict, deque
from datetime import datetime, timedelta, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path
import plistlib
import re
import shutil
import signal
import statistics
import subprocess
import sys
import threading
import time
import urllib.request
from urllib.parse import urlsplit

HEAP = ('free', 'spiram_free', 'dma_free', 'dma_largest_free',
        'internal_free', 'internal_largest_free')
LABEL = 'ai.nyx.panel-stability'
MAX_RESPONSE = 65536


class IdentityMismatch(ValueError):
    pass


def utc(epoch):
    return datetime.fromtimestamp(epoch, timezone.utc).isoformat()


def integer(value):
    if type(value) is not int or value < 0:
        raise ValueError('Invalid numeric status')
    return value


def code(value):
    return value if isinstance(value, str) and re.fullmatch(r'[A-Za-z0-9_.:-]{1,48}', value) else 'other'


def sample(status, expected_slug, epoch):
    if not expected_slug or status.get('nyx', {}).get('slug') != expected_slug:
        raise IdentityMismatch('Configured device identity differs')
    voice = status.get('voice', {})
    wake = voice.get('wake', {})
    lcd = status.get('lcd', {})
    image = lcd.get('temporary_image', {})
    # Explicit allowlist: never retain SSID, IP, identities, keys, descriptions,
    # session IDs, messages, captions, images or complete HTTP responses.
    return dict(kind='sample', t=epoch, utc=utc(epoch),
        uptime_s=integer(status['uptime_s']), fw=code(status['fw']),
        reset_reason=integer(status['reset_reason']) if 'reset_reason' in status else None,
        app_sha256=status['app_sha256'] if re.fullmatch(r'[0-9a-f]{64}', status['app_sha256']) else 'unknown',
        heap={k: integer(status['heap'][k]) for k in HEAP},
        phase=dict(voice_active=voice.get('active') is True, voice_busy=voice.get('busy') is True,
            voice_state=code(voice.get('state')), wake_enabled=wake.get('enabled') is True,
            wake_listening=wake.get('listening') is True, wake_state=code(wake.get('state')),
            microphone_muted=voice.get('microphone_muted') is True,
            voice_mode=code(voice.get('input_mode')),
            media_mode=code(status.get('media', {}).get('mode')),
            display_mode=code(lcd.get('mode')), display_depth=integer(lcd.get('depth', 0)),
            background_storage=code(lcd.get('default_image')),
            background_bytes=integer(lcd.get('default_image_bytes', 0)),
            temporary_image_active=image.get('active') is True,
            temporary_image_bytes=integer(image.get('jpeg_bytes', 0)),
            temporary_image_width=integer(image.get('width', 0)),
            temporary_image_height=integer(image.get('height', 0)),
            ota_stage=code(status.get('ota', {}).get('stage'))),
        counters=dict({k: integer(voice.get(k, 0)) for k in
                  ('call_seconds', 'sent_packets', 'received_packets')},
            wake_detections=integer(wake.get('detections', 0)),
            buffered_ms=integer(wake.get('buffered_ms', 0)),
            transmitted_ms=integer(wake.get('transmitted_ms', 0)),
            camera_captures=integer(status.get('media', {}).get('captures', 0)),
            camera_failures=integer(status.get('media', {}).get('failed_captures', 0))),
        network_connected=status.get('net', {}).get('connected') is True)


def analyze(rows):
    """Same-boot/same-phase five-minute medians, then median pairwise slope.

    A retained loss is only a suspicion: caches and changing workloads can also
    retain allocations. No device action is taken on an observation.
    """
    grouped = defaultdict(list)
    for row in rows:
        if row['uptime_s'] >= 300:
            grouped[json.dumps(row['phase'], sort_keys=True)].append(row)
    trends = []
    for phase, records in grouped.items():
        buckets = defaultdict(list)
        for row in records:
            buckets[int(row['t'] // 300)].append(row)
        if len(buckets) < 8:
            continue
        bins = []
        for _, values in sorted(buckets.items()):
            bins.append(dict(t=statistics.median(r['t'] for r in values),
                heap={k: statistics.median(r['heap'][k] for r in values) for k in HEAP}))
        if bins[-1]['t'] - bins[0]['t'] < 1800:
            continue
        for metric, drop_limit, hourly_limit in (
                ('internal_free', 4096, 1024), ('internal_largest_free', 4096, 1024),
                ('dma_free', 4096, 1024), ('dma_largest_free', 4096, 1024),
                ('spiram_free', 65536, 16384)):
            slopes = [(b['heap'][metric] - a['heap'][metric]) * 3600 / (b['t'] - a['t'])
                      for i, a in enumerate(bins) for b in bins[i + 1:]]
            slope = statistics.median(slopes)
            initial = statistics.median(b['heap'][metric] for b in bins[:3])
            final = statistics.median(b['heap'][metric] for b in bins[-3:])
            retained = initial - final
            suspect = (retained >= drop_limit and slope <= -hourly_limit
                       and all(b['heap'][metric] <= initial - drop_limit for b in bins[-3:]))
            trends.append(dict(phase=json.loads(phase), metric=metric, buckets=len(bins),
                observed_seconds=bins[-1]['t'] - bins[0]['t'],
                bytes_per_hour=slope, retained_drop_bytes=retained,
                suspected_retained_loss=suspect))
    return trends


def config_read(path):
    d = json.loads(Path(path).read_text())
    url = urlsplit(d['status_url'])
    if url.scheme not in ('http', 'https') or not url.hostname or url.username or url.password:
        raise ValueError('Expected a status URL without credentials')
    if not isinstance(d['expected_slug'], str) or not d['expected_slug']:
        raise ValueError('Explicit device identity required')
    d['interval_s'] = int(d.get('interval_s', 30))
    if not 15 <= d['interval_s'] <= 3600:
        raise ValueError('Sampling interval out of range')
    d['state_dir'] = str(Path(d['state_dir']).expanduser().resolve())
    return d


class Monitor:
    def __init__(self, config):
        self.config = config
        self.root = Path(config['state_dir'])
        self.root.mkdir(parents=True, exist_ok=True)
        self.root.chmod(0o700)
        self.state_path = self.root / 'status.json'
        self.state = json.loads(self.state_path.read_text()) if self.state_path.exists() else {}
        self.identity = hashlib.sha256(config['expected_slug'].encode()).hexdigest()
        if self.state.get('identity_fingerprint') != self.identity:
            self.state = {}
        self.rows = deque(maxlen=1440)
        self.last = self.state.get('last_sample')
        self.last_rotation_day = None
        # Restore only recent same-device samples. No raw HTTP responses exist.
        now = time.time()
        if self.state.get('identity_configured') and self.last:
            for day in (datetime.now(timezone.utc) - timedelta(days=1), datetime.now(timezone.utc)):
                path = self.root / ('metrics-' + day.strftime('%Y-%m-%d') + '.jsonl')
                if path.exists():
                    with path.open() as history:
                        for line in history:
                            try:
                                row = json.loads(line)
                                if (row.get('kind') == 'sample' and row.get('device_fingerprint') == self.identity
                                        and row.get('boot') == self.last['boot'] and row['t'] > now - 21600):
                                    self.rows.append(row)
                                elif row.get('kind') in ('monitor_error', 'monitor_gap', 'boot_boundary'):
                                    self.rows.clear()
                            except (ValueError, KeyError):
                                continue

    def log(self, row):
        day = datetime.fromtimestamp(row['t'], timezone.utc).strftime('%Y-%m-%d')
        with (self.root / ('metrics-' + day + '.jsonl')).open('a') as f:
            f.write(json.dumps(row, separators=(',', ':')) + '\n')
        if day != self.last_rotation_day:
            today = datetime.fromtimestamp(row['t'], timezone.utc).date()
            for path in self.root.glob('metrics-*.jsonl*'):
                if path.is_symlink():
                    continue
                match = re.fullmatch(r'metrics-(\d{4}-\d{2}-\d{2})\.jsonl(\.gz)?', path.name)
                if not match:
                    continue
                age = (today - datetime.strptime(match[1], '%Y-%m-%d').date()).days
                if age > 30:
                    path.unlink()
                elif age > 2 and path.suffix != '.gz':
                    compressed = path.with_suffix('.jsonl.gz')
                    compressed.write_bytes(gzip.compress(path.read_bytes(), mtime=0))
                    path.unlink()
            self.last_rotation_day = day

    def save(self):
        temp = self.root / 'status.json.tmp'
        temp.write_text(json.dumps(self.state, indent=2) + '\n')
        temp.replace(self.state_path)

    def poll(self):
        now = time.time()
        try:
            with urllib.request.urlopen(self.config['status_url'], timeout=4) as response:
                raw = response.read(MAX_RESPONSE + 1)
            if len(raw) > MAX_RESPONSE:
                raise ValueError('Response exceeds limit')
            row = sample(json.loads(raw), self.config['expected_slug'], now)
            last = self.last
            boot = last['boot'] if last else 0
            if last and (row['uptime_s'] < last['uptime_s'] - 2 or row['app_sha256'] != last['app_sha256']):
                reboot = row['uptime_s'] < last['uptime_s'] - 2
                boot += 1
                self.rows.clear()
                self.log(dict(kind='boot_boundary', t=now, utc=utc(now), reboot_observed=reboot,
                              reset_reason=row['reset_reason']))
                self.state['observed_restarts'] = self.state.get('observed_restarts', 0) + int(reboot)
            if last and now - last['t'] > max(120, self.config['interval_s'] * 3):
                self.rows.clear()
                self.log(dict(kind='monitor_gap', t=now, utc=utc(now), seconds=now - last['t']))
            row['boot'] = boot
            row['device_fingerprint'] = self.identity
            self.rows.append(row)
            while self.rows and self.rows[0]['t'] < now - 21600:
                self.rows.popleft()
            self.log(row)
            trends = analyze(self.rows)
            self.state.update(health='online', identity_configured=True, identity_fingerprint=self.identity,
                last_success_utc=row['utc'],
                last_sample=row, consecutive_errors=0, recent_samples=len(self.rows),
                sampled_minima={k: min(r['heap'][k] for r in self.rows) for k in HEAP},
                trends=trends, suspected_losses=[t for t in trends if t['suspected_retained_loss']],
                interpretation='Sampled values only. Matched-state retained loss is a suspicion, not proof of a leak. '
                               'Long-call/reconnect/hardware acceptance is separate.')
            self.last = row
        except (OSError, ValueError, KeyError, TypeError) as error:
            self.rows.clear()
            kind = type(error).__name__
            self.log(dict(kind='monitor_error', t=now, utc=utc(now), error_type=kind))
            self.state.update(health='identity_mismatch' if isinstance(error, IdentityMismatch) else 'unreachable_or_invalid',
                consecutive_errors=self.state.get('consecutive_errors', 0) + 1, last_error_utc=utc(now),
                last_error_type=kind, recent_samples=0, trends=[], suspected_losses=[])
        self.state['updated_utc'] = utc(now)
        self.save()
        return self.state


def install(config_path, config):
    agent = Path.home() / 'Library/LaunchAgents' / (LABEL + '.plist')
    if agent.exists():
        raise ValueError('Monitor already installed; stop/uninstall it before replacing')
    monitor = Monitor(config)
    if monitor.poll()['health'] != 'online':
        raise ValueError('Device preflight failed; background monitor not installed')
    installed = monitor.root / 'monitor_stability.py'
    shutil.copyfile(Path(__file__).resolve(), installed)
    agent.parent.mkdir(parents=True, exist_ok=True)
    agent.write_bytes(plistlib.dumps(dict(Label=LABEL,
        ProgramArguments=[sys.executable, str(installed), 'run', '--config', str(Path(config_path).resolve())],
        WorkingDirectory=str(monitor.root), RunAtLoad=True, KeepAlive=True, ThrottleInterval=30,
        ProcessType='Background', LowPriorityIO=True, Nice=10,
        StandardOutPath=str(monitor.root / 'service.log'), StandardErrorPath=str(monitor.root / 'service-error.log'))))
    try:
        subprocess.run(['launchctl', 'bootstrap', 'gui/' + str(os.getuid()), str(agent)], check=True)
    except subprocess.CalledProcessError:
        agent.unlink()
        raise
    print(json.dumps(dict(installed=True, label=LABEL, state_file=str(monitor.state_path))))


def main():
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('sample', 'run', 'report', 'install'))
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    config = config_read(args.config)
    if args.action == 'install':
        install(args.config, config)
        return
    monitor = Monitor(config)
    if args.action == 'report':
        print(json.dumps(monitor.state, indent=2))
    elif args.action == 'sample':
        print(json.dumps(monitor.poll(), indent=2))
    else:
        stop = threading.Event()
        for sig in (signal.SIGTERM, signal.SIGINT):
            signal.signal(sig, lambda *_: stop.set())
        while not stop.is_set():
            started = time.monotonic()
            monitor.poll()
            stop.wait(max(0, config['interval_s'] - (time.monotonic() - started)))


if __name__ == '__main__':
    main()
