"""Frozen two-head regression on existing paired/continuous development PCM.

Requires the explicit historical corpus, Hi ESP reports and per-source embedding
cache identities. No new threshold selection or reserved final corpus is read.
"""
from pathlib import Path
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import importlib.util
import json
import sys
import time
import numpy as np
import soundfile as sf

P = None
REPO = Path(__file__).resolve().parents[3]
# Import the measured public implementations, not another feature frontend.
import prepare_head_features as extractor
from evaluate_head import Head
spec = importlib.util.spec_from_file_location('baseline_compare', REPO/'research/wakeword_baseline/compare.py')
compare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(compare)
detector = None
frozen = None


def initialize(plan_path, data_root):
    global detector, frozen, P
    P = Path(data_root)
    frozen = json.loads(Path(plan_path).read_text())
    for name, digest in frozen.get('runtime_inputs', {}).items():
        assert hashlib.sha256((P/name).read_bytes()).hexdigest() == digest
    for name, digest in frozen.get('helpers', {}).items():
        assert hashlib.sha256((REPO/'research/wakeword_p4/embedding'/name).read_bytes()).hexdigest() == digest
    from evaluate_equal_mean import EqualMeanHead
    paths=[]
    for member in frozen['selected']['members']:
        model=P/member['checkpoint']/'head.tflite'
        assert hashlib.sha256(model.read_bytes()).hexdigest()==member['head_sha256']
        paths.append(model)
    detector=EqualMeanHead(paths)
    extractor.initialize(P/'embedding-stream-int8-quiet.tflite',P/'native-fft-window.npy',P/'native-mel-filter.npy')


def worker(job):
    path = Path(job['path'])
    assert hashlib.sha256(path.read_bytes()).hexdigest() == job['sha256']
    pcm, rate = sf.read(path, dtype='int16')
    assert rate == 16000 and pcm.ndim == 1
    assert len(pcm) == job['samples']
    prime = frozen['pcm_protocol']['prime_zero_samples']
    identity = hashlib.sha256((job['sha256']+':'+str(prime)+':'+
        'af0e12dd0cd6f368b0c2c24b2b5f4ff0df85af8175185d8e6bb6ef07e69e233a').encode()).hexdigest()
    cache = P/'regression-backbone-cache'/f'{identity}.npy'
    if cache.exists():
        assert hashlib.sha256(cache.read_bytes()).hexdigest() == job['expected_embedding_sha256']
        embeddings = np.load(cache)
    else:
        embeddings = extractor.extract(np.pad(pcm, (prime, 0)))
        cache.parent.mkdir(exist_ok=True)
        np.save(cache, embeddings)
    if hashlib.sha256(cache.read_bytes()).hexdigest() != job['expected_embedding_sha256']:
        raise ValueError('Computed or cached regression embedding identity differs')
    scores, times = detector.scores(embeddings, frozen['selected']['event_policy']['score_average_steps'])
    samples = np.rint(times*16000).astype(np.int64)-prime
    offsets, priming_events = [], []
    armed, last = True, -1000
    threshold = frozen['selected']['selected']['threshold']
    for index, (score, sample) in enumerate(zip(scores, samples)):
        if score < threshold*.5 and index-last >= 25:
            armed = True
        if armed and score >= threshold and index-last >= 25:
            if sample <= 0:
                priming_events.append(int(sample))
            elif sample <= len(pcm):
                offsets.append(int(sample))
            last, armed = index, False
    evaluated_samples = len(pcm)//1280*1280
    return {'id': job['id'], 'events': len(offsets), 'offsets': offsets,
            'samples': len(pcm), 'evaluated_samples': evaluated_samples,
            'priming_events': priming_events, 'embedding_sha256': hashlib.sha256(cache.read_bytes()).hexdigest()}


def main():
    global P
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--kind', choices=['paired', 'english', 'mandarin'], required=True)
    parser.add_argument('--workers', type=int, default=3)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    P = args.data_root.resolve()
    if args.workers < 1:
        parser.error('Workers must be positive')
    output = args.output
    output.mkdir(exist_ok=False)
    plan_path = args.plan
    frozen = json.loads(plan_path.read_text())
    if args.kind == 'paired':
        manifest = P.parent/'baseline-paired-spelled/manifest-with-controls.jsonl'
        fixtures = list(map(json.loads, manifest.read_text().splitlines()))
        baseline_path = P.parent/'baseline-spelled-full/report.json'
        baseline = json.loads(baseline_path.read_text())
        assert baseline.get('restored') and len(baseline['clips']) == len(fixtures)
        assert baseline['manifest_sha256'] == hashlib.sha256(manifest.read_bytes()).hexdigest()
        for f, row in zip(fixtures, baseline['clips']):
            assert all(f[k] == row[k] for k in ['path', 'sha256', 'samples']) and row['result']['complete']
        jobs = [{'id': str(index), 'path': str(manifest.parent/f['path']),
                 'sha256': f['sha256'], 'samples': f['samples']} for index, f in enumerate(fixtures)]
    else:
        corpus = P.parent/('continuous-test-clean-exact-v2' if args.kind == 'english' else 'mandarin-fleurs/validation')
        manifest = corpus/'manifest.json'
        fixtures = json.loads(manifest.read_text())['records']
        baseline_path = P.parent/('baseline-continuous/report.json' if args.kind == 'english' else 'baseline-mandarin-validation/report.json')
        baseline = json.loads(baseline_path.read_text())
        assert baseline.get('restored')
        baseline_manifest = corpus/'baseline-manifest.jsonl'
        baseline_rows = list(map(json.loads, baseline_manifest.read_text().splitlines()))
        assert baseline['manifest_sha256'] == hashlib.sha256(baseline_manifest.read_bytes()).hexdigest()
        assert len(baseline_rows) == len(baseline['clips']) == len(fixtures)
        for f, ref, row in zip(fixtures, baseline_rows, baseline['clips']):
            assert f['pcm_sha256'] == ref['sha256'] == row['sha256']
            assert row['result']['complete']
        jobs = [{'id': str(index), 'path': str(corpus/f['pcm_path']),
                 'sha256': f['pcm_sha256'], 'samples': round(f['seconds']*16000)} for index, f in enumerate(fixtures)]
    old_rows = list(map(json.loads, (P/f'secondary-dnn64-{args.kind}'/'rows-private.jsonl').read_text().splitlines()))
    assert len(old_rows) == len(jobs)
    for job, old in zip(jobs, old_rows):
        assert job['id'] == old['id'] and job['samples'] == old['samples']
        job['expected_embedding_sha256'] = old['embedding_sha256']
    started = time.monotonic()
    results = []
    with (output/'rows-private.jsonl').open('x') as log, ProcessPoolExecutor(max_workers=args.workers, initializer=initialize, initargs=(plan_path, P)) as pool:
        for number, row in enumerate(pool.map(worker, jobs), 1):
            log.write(json.dumps(row)+'\n');log.flush();results.append(row)
            if number % 20 == 0 or number == len(jobs):
                print(json.dumps({'kind': args.kind, 'completed': number, 'total': len(jobs), 'seconds': time.monotonic()-started}), flush=True)
    result = {'scope': 'Validation-frozen matched hard-negative development regression, exact existing PCM/cached features; no reserved/physical/whole-app acceptance',
              'plan_sha256': hashlib.sha256(plan_path.read_bytes()).hexdigest(),
              'manifest_sha256': hashlib.sha256(manifest.read_bytes()).hexdigest(),
              'baseline_sha256': hashlib.sha256(baseline_path.read_bytes()).hexdigest(),
              'ensemble_identity_sha256': frozen['selected']['model_sha256'], 'threshold': frozen['selected']['selected']['threshold'],
              'source_pcm_hashes_verified': True, 'priming_events': sum(len(r['priming_events']) for r in results),
              'seconds': time.monotonic()-started, 'accepted': False}
    if args.kind == 'paired':
        ours = [{**f, 'result': row} for f, row in zip(fixtures, results)]
        result['hi_esp'] = compare.summarize([r for r in baseline['clips'] if r.get('scope','paired_mac_regression') == 'paired_mac_regression'], 'hi_esp')
        result['hey_nyx'] = compare.summarize([r for r in ours if r.get('scope','paired_mac_regression') == 'paired_mac_regression'], 'hey_nyx')
    else:
        result['hi_esp'] = {'events': sum(r['result']['events'] for r in baseline['clips']),
                            'source_hours': sum(j['samples'] for j in jobs)/16000/3600}
        hours = sum(r['evaluated_samples'] for r in results)/16000/3600
        count = sum(r['events'] for r in results)
        result['hey_nyx'] = {'events': count, 'evaluated_hours': hours, 'false_per_hour': count/hours}
    (output/'report.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
