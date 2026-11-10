"""Compare frozen heads on matched level views of existing validation speech.

This is a development stress diagnostic, not calibration or fresh acceptance.
Keep every original variant-1 room/noise draw and scale the complete signal,
including its noise. No speed, SNR, source, label or threshold changes.
The scale-1 render must reproduce every original feature and score exactly.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import soundfile as sf
from scipy.signal import fftconvolve, lfilter

import prepare_head_features as extractor
from evaluate_equal_mean import EqualMeanHead

CONFIG = None
MODELS = None


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def render(job):
    """Reproduce the unchanged v1 variant-1 renderer before level scaling."""
    if job['kind'] != 'tts' or job['variant'] != 1 or job['split'] != 'validation':
        raise ValueError('Only existing validation variant-1 speech is eligible')
    if digest(job['source']) != job['source_sha256']:
        raise ValueError('Synthetic source identity differs')
    pcm, rate = sf.read(job['source'], dtype='int16')
    if rate != 16000 or pcm.ndim != 1:
        raise ValueError('Expected original mono16kHz source')
    rng = np.random.default_rng(int(job['id'][:8], 16))
    signal = np.pad(pcm.astype(np.float32), (38400, 16000))
    ir = np.zeros(2400, np.float32)
    ir[0] = 1
    taps = rng.integers(80, len(ir), 24)
    ir[taps] += rng.normal(0, .2, len(taps)) * np.exp(-taps / rng.uniform(500, 1400))
    signal = fftconvolve(signal, ir)[:len(signal)]
    level = float(rng.uniform(.04, .20)) * 32767
    speech_rms = max(655, float(np.sqrt(np.mean(pcm.astype(np.float64) ** 2))))
    signal *= level / speech_rms
    noise = lfilter([1], [1, -float(rng.uniform(.5, .96))], rng.normal(size=len(signal)))
    noise /= max(1e-9, float(np.sqrt(np.mean(noise ** 2))))
    signal += noise * level * 10 ** (-float(rng.uniform(8, 30)) / 20)
    return signal


def initialize(config):
    global CONFIG, MODELS
    CONFIG = config
    extractor.initialize(config['backbone'], config['window'], config['mel_filter'])
    MODELS = {name: EqualMeanHead([Path(p) for p in item['heads']])
              for name, item in config['models'].items()}


def measure(job):
    signal = render(job)
    baseline = extractor.extract(signal)
    reference = Path(job['reference_features'])
    if digest(reference) != job['features_sha256'] or not np.array_equal(baseline, np.load(reference)):
        raise ValueError('Scale-1 render differs from original v1 features')
    rows = []
    for factor in CONFIG['factors']:
        embeddings = baseline if factor == 1 else extractor.extract(signal * factor)
        for name, model in MODELS.items():
            scores, times = model.scores(embeddings, 3)
            original = job['reference_scores'][name]
            if factor == 1 and (scores.tolist() != original['scores'] or times.tolist() != original['times']):
                raise ValueError('Scale-1 scores differ from frozen calibration')
            end = (38400 + job['keyword_end']) / 16000 if job['label'] else None
            valid = (times >= end - .1) & (times <= end + .8) if job['label'] else np.ones(len(times), bool)
            hit = valid & (scores >= CONFIG['models'][name]['threshold'])
            rows.append({'id': job['id'], 'label': job['label'], 'engine': job['engine'],
                         'model': name, 'factor': factor, 'detected': bool(hit.any()),
                         'max_score': float(scores[valid].max()) if valid.any() else 0.,
                         'delay_ms': float((times[hit][0] - end) * 1000) if job['label'] and hit.any() else None})
    return rows


def summarize(rows):
    positives = [r for r in rows if r['label']]
    negatives = [r for r in rows if not r['label']]
    engines = {name: [r for r in positives if r['engine'] == name]
               for name in sorted({r['engine'] for r in positives})}
    recalls = {name: sum(r['detected'] for r in group) / len(group) for name, group in engines.items()}
    return {'positive_sources': len(positives), 'positive_hits': sum(r['detected'] for r in positives),
            'macro_recall': float(np.mean(list(recalls.values()))), 'recall_by_engine': recalls,
            'negative_sources': len(negatives), 'false_clips': sum(r['detected'] for r in negatives)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--workers', type=int, default=3)
    p.add_argument('--source-prefix', type=Path,
                   help='Original corpus path prefix, replaced only when --source-root is provided')
    p.add_argument('--source-root', type=Path,
                   help='Downloaded workspace containing the same hash-verified synthetic WAVs')
    args = p.parse_args()
    plan = json.loads(args.plan.read_text())
    if args.workers < 1 or args.output.exists() or plan['factors'] != [1, .05, .1]:
        p.error('Positive workers, fresh output and the prespecified level factors required')
    if (args.source_prefix is None) != (args.source_root is None):
        p.error('Source prefix and replacement root must be specified together')
    base = args.data_root.resolve()
    for name, expected in plan['inputs'].items():
        if digest(base / name) != expected:
            raise ValueError('Frozen input identity differs: ' + name)
    for name, expected in plan['helpers'].items():
        if digest(Path(__file__).with_name(name)) != expected:
            raise ValueError('Evaluation implementation differs from plan')
    jobs = json.loads((base / 'head-features-tts-v1/jobs-private.json').read_text())
    hashes = {split: {j['source_sha256'] for j in jobs if j['split'] == split}
              for split in ('training', 'validation')}
    if hashes['training'] & hashes['validation']:
        raise ValueError('Validation sources overlap training')
    index = {r['id']: r for r in map(json.loads, (base / 'head-features-tts-v1/index.jsonl').read_text().splitlines())}
    selected = [dict(j) for j in jobs if j['split'] == 'validation' and j['variant'] == 1]
    if len(selected) != plan['validation_jobs'] or len({j['id'] for j in selected}) != len(selected):
        raise ValueError('Validation source coverage differs')
    config = {key: str(base / plan[key]) for key in ('backbone', 'window', 'mel_filter')}
    config.update(factors=plan['factors'], models={})
    for name, entry in plan['models'].items():
        calibration = json.loads((base / entry['calibration']).read_text())
        if calibration['selected']['threshold'] != entry['threshold']:
            raise ValueError('Threshold differs from frozen original calibration')
        heads = [base / member['checkpoint'] / 'head.tflite' for member in calibration['members']]
        for head, member in zip(heads, calibration['members']):
            if not head.resolve().is_relative_to(base) or digest(head) != member['head_sha256']:
                raise ValueError('Frozen head identity differs')
        config['models'][name] = {'heads': list(map(str, heads)), 'threshold': entry['threshold']}
        details = json.loads((base / entry['details']).read_text())
        reference = {r['id']: r for r in details['clips']}
        for job in selected:
            job.setdefault('reference_scores', {})[name] = reference[job['id']]
    for job in selected:
        if args.source_root is not None:
            relative = Path(job['source']).relative_to(args.source_prefix)
            rebased = (args.source_root / relative).resolve()
            rebased.relative_to(args.source_root.resolve())
            job['source'] = str(rebased)
        row = index[job['id']]
        if row['label'] != job['label'] or row['source_sha256'] != job['source_sha256']:
            raise ValueError('Validation labels or sources differ')
        path = Path(row['path'])
        job.update(reference_features=str(path if path.is_absolute() else base / path),
                   features_sha256=row['features_sha256'])
    args.output.mkdir(parents=True)
    (args.output / 'plan.json').write_bytes(args.plan.read_bytes())
    started, rows = time.monotonic(), []
    with (args.output / 'details.jsonl').open('x') as log, ProcessPoolExecutor(
            max_workers=args.workers, initializer=initialize, initargs=(config,)) as pool:
        for number, result in enumerate(pool.map(measure, selected), 1):
            rows.extend(result)
            for row in result:
                log.write(json.dumps(row) + '\n')
            if number % 100 == 0 or number == len(selected):
                log.flush()
                print(json.dumps({'completed': number, 'total': len(selected),
                                  'seconds': time.monotonic() - started}), flush=True)
    report = {'scope': __doc__, 'plan_sha256': digest(args.plan), 'all_scale1_features_and_scores_exact': True,
              'sources_disjoint_from_training': True, 'speed_and_snr_unchanged': True,
              'thresholds_unchanged': True, 'accepted': False, 'results': {}}
    for name in plan['models']:
        report['results'][name] = {str(f): summarize([r for r in rows if r['model'] == name and r['factor'] == f])
                                   for f in plan['factors']}
    (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
