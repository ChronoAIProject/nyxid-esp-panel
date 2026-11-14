"""Wait for existing GPU publications or a new old-event release asset.

This process does not start GPU work, message anyone, download payloads, invoke
models, infer remote process liveness, or change any admission decision.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import time


REPOSITORY = 'ChronoAIProject/nyxid-esp-panel'
RELEASE = 'research-hey-nyx-20261110'
GPU_BRANCH = 'research-hye-nyx-voice-gpu'
PREFIX = 'research/wakeword_p4/results/4090-hey-broad-oneway-consistency-'
RESULTS = {name: PREFIX + name + '-v1.json'
           for name in ['formal-result', 'fixed-grid-compile-result']}


def command(argv, cwd):
    return subprocess.run(argv, cwd=cwd, capture_output=True, timeout=30)


def observe(args):
    baseline_bytes = args.baseline_assets.read_bytes()
    baseline = json.loads(baseline_bytes)
    baseline_names = {asset['name'] for asset in baseline['assets']}
    assert baseline_names and len(baseline_names) == len(baseline['assets'])
    args.output_dir.mkdir(exist_ok=False)
    (args.output_dir / 'baseline-assets.json').write_bytes(baseline_bytes)
    (args.output_dir / 'observer-source.py').write_bytes(Path(__file__).read_bytes())
    journal = args.output_dir / 'observations.jsonl'
    for number in range(args.max_observations):
        row = dict(observation=number + 1, utc=datetime.now(timezone.utc).isoformat(),
                   gpu_process_live_verified=False, baseline_assets=len(baseline_names),
                   observer_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        try:
            fetched = command(['git', 'fetch', '--quiet', 'origin', GPU_BRANCH], args.repository_root)
            row['fetch_exit_code'] = fetched.returncode
            if fetched.returncode == 0:
                ref = subprocess.check_output(['git', 'rev-parse', 'origin/' + GPU_BRANCH],
                                              cwd=args.repository_root, text=True).strip()
                row['gpu_ref'] = ref
                found = {}
                for label, path in RESULTS.items():
                    read = command(['git', 'show', ref + ':' + path], args.repository_root)
                    if read.returncode == 0:
                        value = json.loads(read.stdout)
                        found[label] = dict(path=path, bytes=len(read.stdout),
                                            sha256=hashlib.sha256(read.stdout).hexdigest())
                        if label == 'formal-result':
                            row['formal_reported_all_passed'] = value['all_passed']
                row['published_results'] = found
                row['ready_for_model_metadata_review'] = len(found) == len(RESULTS)
        except (subprocess.TimeoutExpired, json.JSONDecodeError) as error:
            row['git_observation_error'] = type(error).__name__
        try:
            release = command(['gh', 'release', 'view', RELEASE, '--repo', REPOSITORY,
                               '--json', 'assets'], args.repository_root)
            row['release_read_exit_code'] = release.returncode
            if release.returncode == 0:
                assets = json.loads(release.stdout)['assets']
                row['current_assets'] = len(assets)
                new = [asset for asset in assets if asset['name'] not in baseline_names]
                row['new_asset_names'] = [asset['name'] for asset in new]
                matches = [asset for asset in new
                           if 'old' in asset['name'].lower()
                           and any(word in asset['name'].lower()
                                   for word in ['regression', 'event', 'ledger'])]
                row['new_old_event_assets_for_inspection'] = [
                    {key: asset.get(key) for key in ['name', 'size', 'url', 'updatedAt']}
                    for asset in matches]
        except (subprocess.TimeoutExpired, json.JSONDecodeError) as error:
            row['release_observation_error'] = type(error).__name__
        ready = bool(row.get('ready_for_model_metadata_review')
                     or row.get('new_old_event_assets_for_inspection')
                     or row.get('formal_reported_all_passed') is False)
        row['ready_for_handoff_inspection'] = ready
        with journal.open('a') as handle:
            handle.write(json.dumps(row) + '\n')
        print(json.dumps(row), flush=True)
        if ready:
            (args.output_dir / 'publication-observed.json').write_text(json.dumps(row, indent=2) + '\n')
            return
        if number + 1 < args.max_observations:
            time.sleep(args.interval)
    (args.output_dir / 'observation-window-ended.json').write_text(json.dumps(dict(
        observation_window_ended=True, gpu_training_terminal_inferred=False,
        limits='Only this read-only observer ended; no inference about GPU work. '
               'Do not restart training because this observation window expired.'), indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repository-root', type=Path, default=Path.cwd())
    parser.add_argument('--baseline-assets', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--interval', type=float, default=45)
    parser.add_argument('--max-observations', type=int, default=80)
    args = parser.parse_args()
    assert 1 <= args.interval <= 60 and args.max_observations > 0
    observe(args)
