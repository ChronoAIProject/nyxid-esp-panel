"""Read only the development panel's idle status to measure whole-app memory."""
import argparse
import hashlib
import json
from pathlib import Path
from statistics import median
import time
import urllib.request


HEAP = ('free', 'spiram_free', 'dma_free', 'dma_largest_free',
        'internal_free', 'internal_largest_free')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--status-url', required=True)
    p.add_argument('--expected-app-sha256', required=True)
    p.add_argument('--expected-slot', choices=('ota_0', 'ota_1'), required=True)
    p.add_argument('--expected-word', choices=('Hi ESP', 'Hey Nyx'), required=True)
    p.add_argument('--seconds', type=int, default=40)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.seconds < 30 or a.output.exists():
        p.error('Use at least 30 seconds and a fresh output file')
    samples, identity = [], None
    report = {'scope': __doc__, 'measure_seconds': a.seconds, 'completed': False,
              'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'app_sha256': a.expected_app_sha256, 'slot': a.expected_slot,
              'wake_word': a.expected_word, 'samples': samples,
              'speaker_playback': False, 'microphone_pcm_access': False, 'firmware_changed': False,
              'limits': 'Status snapshots measure idle heap and listening continuity only; they cannot measure CPU, recognition, capture drops, or calls.'}
    start = time.monotonic()
    try:
        for sample in range(a.seconds + 1):
            time.sleep(max(0, start + sample - time.monotonic()))
            with urllib.request.urlopen(a.status_url, timeout=4) as result:
                state = json.load(result)
            wake = state['voice']['wake']
            if (state['app_sha256'] != a.expected_app_sha256 or state['ota']['running'] != a.expected_slot
                    or state['ota']['busy'] or state['nyx']['state'] != 'up'
                    or state['voice']['busy'] or wake['word'] != a.expected_word or not wake['listening']):
                raise RuntimeError('Expected development app, connected idle listener or OTA slot changed')
            # Keep service identity constant without putting identifiers in the report.
            service = state['voice']['service_id'], state['voice']['agent_id']
            if identity is None:
                identity = service
            elif identity != service:
                raise RuntimeError('Voice service identity changed')
            row = {'elapsed_s': round(time.monotonic() - start, 3), 'uptime_s': state['uptime_s'],
                   'detections': wake['detections'], 'heap_bytes': {key: state['heap'][key] for key in HEAP}}
            if samples and (row['uptime_s'] < samples[-1]['uptime_s']
                            or row['detections'] != samples[-1]['detections']):
                raise RuntimeError('Board restarted or a wake event interrupted the idle window')
            samples.append(row)
        report['heap_summary_bytes'] = {}
        for key in HEAP:
            values = [row['heap_bytes'][key] for row in samples]
            report['heap_summary_bytes'][key] = {'min': min(values), 'median': median(values), 'max': max(values)}
        report['completed'] = True
    except Exception as error:
        report['failure'] = type(error).__name__
        raise
    finally:
        with a.output.open('x') as f:
            f.write(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'completed': True, 'samples': len(samples), 'heap_bytes': report['heap_summary_bytes']}))


if __name__ == '__main__':
    main()
