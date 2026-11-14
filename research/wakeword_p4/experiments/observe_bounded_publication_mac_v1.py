"""Observe existing GPU publications; never launch or restart GPU work."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import time


PREFIX = 'research/wakeword_p4/results/4090-hey-broad-bounded-consistency-'
FILES = {name: PREFIX + name + '-v1.json' for name in
         ['formal-result', 'fixed-grid-compile-result']}


def observe(a):
    a.output_dir.mkdir(exist_ok=False)
    journal = a.output_dir / 'observations.jsonl'
    for number in range(a.max_observations):
        row = dict(observation=number + 1, utc=datetime.now(timezone.utc).isoformat(),
                   gpu_process_live_verified=False)
        try:
            fetched = subprocess.run(['git', 'fetch', '--quiet', 'origin',
                                      'research-hye-nyx-voice-gpu'],
                                     capture_output=True, timeout=30)
            row['fetch_exit_code'] = fetched.returncode
            if fetched.returncode == 0:
                ref = subprocess.check_output(['git', 'rev-parse',
                    'origin/research-hye-nyx-voice-gpu'], text=True).strip()
                row['gpu_ref'] = ref
                found = {}
                for name, path in FILES.items():
                    read = subprocess.run(['git', 'show', ref + ':' + path],
                                          capture_output=True, timeout=10)
                    if read.returncode == 0:
                        value = json.loads(read.stdout)
                        found[name] = dict(path=path, bytes=len(read.stdout),
                            sha256=hashlib.sha256(read.stdout).hexdigest())
                        if name == 'formal-result':
                            row['formal_reported_all_passed'] = value['all_passed']
                row['found'] = found
                row['ready_for_metadata_review'] = len(found) == len(FILES)
        except subprocess.TimeoutExpired:
            row['observation_timeout'] = True
        with journal.open('a') as stream:
            stream.write(json.dumps(row) + '\n')
        print(json.dumps(row), flush=True)
        if row.get('ready_for_metadata_review') or row.get('formal_reported_all_passed') is False:
            (a.output_dir / 'publication-observed.json').write_text(json.dumps(row, indent=2) + '\n')
            return
        if number + 1 < a.max_observations:
            time.sleep(a.interval)
    (a.output_dir / 'observation-window-ended.json').write_text(json.dumps(dict(
        observation_window_ended=True, gpu_training_terminal_inferred=False,
        limits='Only this bounded read-only observation ended; never restart training from this result.'), indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--interval', type=float, default=45)
    parser.add_argument('--max-observations', type=int, default=80)
    args = parser.parse_args()
    assert 1 <= args.interval <= 60 and args.max_observations > 0
    observe(args)
