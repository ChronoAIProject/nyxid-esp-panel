"""Find hard negatives on pinned public dialogue and Mandarin training audio."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time
import urllib.request

import numpy as np

from evaluate import Detector, events, smooth
from features import continuous_wav_features
from mine_stream_confusers import SMOOTH, THRESHOLDS, peaks, sha

GPU = '3b52402e09d83df7da233cc427155f1ffc3850ae'
AMI = {
    'training-source': 'ba4449e1a371e5594cb28968d409b85ff7f3be944a8b217f6c666f1a453d3023',
    'extra-training-source': '1c5ad8fa024b122a6006503590d0ab1ffb735d8400b46a6e7d5db1d07136c007',
    'independent-source': 'ddd3c6d4a1d792965cc2fc8d154d3943f927db4bfabc69c38b7a62e35080447b',
}
FLEURS = '12816832eb6efd1e4d68737d28d5e4b405698be97c8f5725b51edc31325407c3'
MODELS = {
    'ordinary': 'c5eeef619166c71e1d8bf6469e16490689e97e45df3dfdeee282d30f0383e3dc',
    'hard': '921081f23ef620c840b2e397891602d52e1e3e5b8e33bfc8c0ccc8e9e18c0383',
}


def ami_rows():
    manifests = {}
    for name, digest in AMI.items():
        data = subprocess.check_output([
            'git', 'show', f'{GPU}:research/wakeword_p4/results/4090-ami-{name}-v1.json'
        ], cwd=Path(__file__).resolve().parent)
        if hashlib.sha256(data).hexdigest() != digest:
            raise ValueError('AMI audit differs from the frozen GPU branch')
        manifests[name] = json.loads(data)
    reserved = manifests.pop('independent-source')['records']
    keys = ('session', 'sha256', 'pcm_sha256')
    seen = {key: {r[key] for r in reserved} for key in keys}
    people = {person for r in reserved for person in r['participants']}
    rows = []
    for manifest in manifests.values():
        if not manifest['both_split_schemes_reserved_participants_excluded']:
            raise ValueError('AMI split guard is missing')
        for row in manifest['records']:
            session = row['session']
            filename = f'{session}.Array1-01.wav'
            base = 'https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus'
            if (row['split'] != 'training' or set(row['participants']) & people
                    or any(row[key] in seen[key] for key in keys)
                    or row['path'] != filename or row['url'] != f'{base}/{session}/audio/{filename}'):
                raise ValueError('AMI source is not a distinct frozen training meeting')
            people.update(row['participants'])
            for key in keys:
                seen[key].add(row[key])
            rows.append(row)
    return rows


def ami_features(row, cache):
    wav = cache/row['path']
    if not wav.exists():
        temp = wav.with_suffix('.part')
        for attempt in range(3):
            try:
                with urllib.request.urlopen(row['url'], timeout=45) as response, temp.open('wb') as f:
                    shutil.copyfileobj(response, f, length=1 << 20)
                if temp.stat().st_size != row['download_content_length'] or sha(temp) != row['sha256']:
                    raise ValueError('AMI download differs from the pinned recording')
                temp.rename(wav)
                break
            except OSError:
                if attempt == 2:
                    raise
                time.sleep(2 ** attempt)
    if sha(wav) != row['sha256']:
        raise ValueError('Cached AMI source differs from the pinned recording')
    raw, padding = continuous_wav_features(wav)
    if (len(raw)+2)*160-padding != row['samples']:
        raise ValueError('AMI feature clock differs from the pinned sample count')
    path = wav.with_suffix('.npy')
    np.save(path, raw, allow_pickle=False)
    return raw, sha(path)


def score(job):
    corpus, row, root, models = job
    if corpus == 'ami':
        raw, digest = ami_features(row, root)
        name, samples = row['session'], row['samples']
    else:
        path = root/row['path']
        digest = sha(path)
        if digest != row['feature_sha256']:
            raise ValueError('FLEURS training features failed checksum')
        raw = np.load(path, mmap_mode='r', allow_pickle=False)
        name = row['id']
        samples = sum(r['samples'] + 3200 for r in row['sources'])
        if raw.shape != (row['frames'], 40) or raw.dtype != np.uint16:
            raise ValueError('Unexpected FLEURS feature geometry')
    metrics = {}
    for arm, path in models.items():
        track = smooth(Detector(path).predict(raw), SMOOTH)
        # Keep only scores available before the end of the original stream.
        # The frontend's final partial-hop pad does not count as exposure.
        n = max(0, (samples // 160 - (3*(SMOOTH-1)+5)) // 3 + 1)
        track = track[:n]
        metrics[arm] = {
            'maximum_score': float(track.max()),
            'events': {str(t): events(track, t) for t in THRESHOLDS},
            'high_score_groups': peaks(track),
        }
    return {'corpus': corpus, 'stream': name, 'samples': samples,
            'feature_sha256': digest, 'models': metrics}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('fleurs-training', 'experiment', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--workers', type=int, default=4)
    a = p.parse_args()
    if a.output.exists() or a.workers < 1:
        p.error('Use a fresh output directory and at least one worker')
    models = {arm: a.experiment/f'model-{arm}'/'hey_nyx.tflite' for arm in MODELS}
    if {arm: sha(path) for arm, path in models.items()} != MODELS:
        raise ValueError('Use the two frozen matched-training exports')
    manifest = a.fleurs_training/'manifest.json'
    if sha(manifest) != FLEURS:
        raise ValueError('Use the original train-only FLEURS feature view')
    fleurs = json.loads(manifest.read_bytes())
    if fleurs['official_split'] != 'train' or fleurs['data_use'] != 'training_only':
        raise ValueError('FLEURS manifest is not training-only')
    ami = ami_rows()
    cache = a.output/'ami'
    cache.mkdir(parents=True)
    jobs = [('ami', r, cache, models) for r in ami]
    jobs += [('fleurs', r, a.fleurs_training, models) for r in fleurs['records']]
    rows = []
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        for row in pool.map(score, jobs):
            rows.append(row)
            if row['corpus'] == 'ami' or len(rows) % 10 == 0:
                print(json.dumps({'scored': len(rows), 'total': len(jobs),
                                  'corpus': row['corpus']}), flush=True)

    totals = {}
    for corpus in ('ami', 'fleurs'):
        subset = [r for r in rows if r['corpus'] == corpus]
        arms = {}
        for arm in models:
            group = [r['models'][arm] for r in subset]
            tally = Counter()
            for row in group:
                tally.update(row['events'])
            arms[arm] = {'events': dict(tally), 'high_score_groups': sum(
                len(r['high_score_groups']) for r in group),
                'streams_with_confusers': sum(bool(r['high_score_groups']) for r in group),
                'max_score': max(r['maximum_score'] for r in group)}
        totals[corpus] = {'streams': len(subset),
                          'hours': sum(r['samples'] for r in subset)/16000/3600,
                          'models': arms}
    summary = {'scope': 'Training-only mining. No reserved audio scored.',
               'gpu_source_audit_commit': GPU, 'ami_audit_sha256': AMI,
               'fleurs_training_manifest_sha256': FLEURS, 'model_sha256': MODELS,
               'thresholds': THRESHOLDS, 'totals': totals}
    report = a.output/'mining.json'
    report.write_text(json.dumps({**summary, 'streams': rows}, indent=2)+'\n')
    summary['mining_report_sha256'] = sha(report)
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
