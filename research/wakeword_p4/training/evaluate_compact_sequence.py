"""Frozen development-only comparison of binary versus absolute CTC product.

Both scores come from the same INT8 invocation. No training, final data,
waveform, audio device or board is accessed.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import chi2

from compact_sequence_inference import Detector
from compare_phone_distillation import operating_points
from evaluate import events
from detector_policy import smooth
from phrase_ctc_readout import sha

ARMS = ('control', 'phonetic')  # Names retained for the existing gate helper.


def summarize(clips, ambient, threshold):
    positive, negative = ([r for r in clips if r['label'] == label] for label in (1, 0))
    recall = {engine: float(np.mean([r['max_score'] >= threshold for r in positive if r['engine'] == engine]))
              for engine in sorted({r['engine'] for r in positive})}
    false = {engine: float(np.mean([r['max_score'] >= threshold for r in negative if r['engine'] == engine]))
             for engine in sorted({r['engine'] for r in negative})}
    corpus = {}
    for engine, name in sorted({(r['engine'], r['source'].split('/')[0]) for r in negative}):
        group = [r for r in negative if r['engine'] == engine and r['source'].split('/')[0] == name]
        corpus[engine+':'+name] = {'variants': len(group),
            'false_clips': sum(r['max_score'] >= threshold for r in group),
            'false_rate': float(np.mean([r['max_score'] >= threshold for r in group]))}
    hours = sum(r['hours'] for r in ambient)
    count = sum(r['events'][str(threshold)] for r in ambient)
    return {'threshold': threshold, 'positive_variants': len(positive),
        'recall': float(np.mean([r['max_score'] >= threshold for r in positive])),
        'macro_recall': float(np.mean(list(recall.values()))), 'recall_by_engine': recall,
        'hard_negative_variants': len(negative),
        'hard_negative_false_rate': float(np.mean([r['max_score'] >= threshold for r in negative])),
        'max_source_false_rate': max(false.values()), 'negative_corpora': corpus,
        'max_corpus_false_rate': max(r['false_rate'] for r in corpus.values()),
        'ambient_hours': hours, 'ambient_false_events': count,
        'ambient_false_per_hour': count/hours,
        'ambient_false_per_hour_upper_95': float(.5*chi2.ppf(.95, 2*(count+1))/hours)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('plan', 'export', 'features', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Use a fresh evaluation directory')
    plan = json.loads(args.plan.read_bytes())
    if any(sha(Path(__file__).parent/name) != digest for name, digest in plan['source_sha256'].items()):
        raise ValueError('Executed evaluation source differs')
    smoke_path = args.export/'result.json'
    smoke = json.loads(smoke_path.read_bytes())
    if (sha(smoke_path) != plan['export_result_sha256'] or not smoke['advance_to_development_evaluation']
            or sha(args.export/'compact_dual.tflite') != plan['model_sha256']
            or smoke['model_sha256'] != plan['model_sha256']):
        raise ValueError('Export hash or numerical gate differs')
    for name, digest in plan['feature_index_sha256'].items():
        if sha(args.features/name) != digest:
            raise ValueError('Feature index differs: '+name)
    if plan['event_policy'] != {'smooth_steps': 5, 'refractory_steps': 67, 'rearm_fraction': .5}:
        raise ValueError('Unreviewed event policy')
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/'plan.json').write_bytes(args.plan.read_bytes())
    detector = Detector(args.export/'compact_dual.tflite')
    raw = np.memmap(args.features/'features.bin', mode='r', dtype='<u2').reshape(-1, 40)
    rows = json.loads((args.features/'features.json').read_bytes())
    selected = [r for r in rows if r['split'] == 'validation']
    clips, ambient = {arm: [] for arm in ARMS}, {arm: [] for arm in ARMS}
    with (args.output/'clip-scores.jsonl').open('x') as log:
        for index, row in enumerate(selected):
            source = raw[row['offset']:row['offset']+row['frames']]
            detector.reset()
            binary, product = detector.predict(source)
            if not np.isfinite(product).all() or np.any(product < 0) or np.any(product > binary+1e-12):
                raise ValueError('Invalid absolute CTC product')
            scores = [smooth(value, 5) for value in (binary, product)]
            if scores[0].shape != scores[1].shape:
                raise ValueError('Different score timebases')
            times = (3*(np.arange(len(scores[0]))+4)+5)*.01
            end = (row['keyword_frame']+2)*.01 if row['label'] else None
            valid = ((times >= end-.1) & (times <= end+.8) if end is not None
                     else np.ones(len(times), bool))
            record = {key: row[key] for key in ('engine', 'label', 'text', 'variant', 'sha256')}
            record.update(source=row['path'], keyword_end=end, times=times.tolist(), arms={})
            for arm, values in zip(ARMS, scores, strict=True):
                maximum = float(values[valid].max()) if valid.any() else 0.
                clips[arm].append({'source': row['path'], 'engine': row['engine'],
                                  'label': row['label'], 'max_score': maximum})
                record['arms'][arm] = {'max_score': maximum, 'scores': values.tolist()}
            log.write(json.dumps(record)+'\n')
            if index % 120 == 0 or index == len(selected)-1:
                print(json.dumps({'phase': 'validation_clips', 'completed': index+1,
                                  'total': len(selected)}), flush=True)
    all_ambient = json.loads((args.features/'human.json').read_bytes())
    all_ambient += json.loads((args.features/'ambient.json').read_bytes())
    for index, row in enumerate(all_ambient):
        if row['split'] != 'validation':
            continue
        source = np.load(args.features/row['path'], allow_pickle=False, mmap_mode='r')
        detector.reset()
        binary, product = detector.predict(source)
        if not np.isfinite(product).all() or np.any(product < 0) or np.any(product > binary+1e-12):
            raise ValueError('Invalid continuous CTC product')
        for arm, values in zip(ARMS, (binary, product), strict=True):
            values = smooth(values, 5)
            path = args.output/f'ambient-{arm}-{index:03d}.npy'
            np.save(path, values)
            ambient[arm].append({'feature_path': row['path'], 'array': path.name,
                'source_type': row.get('source_type', 'speech' if 'speaker' in row else 'environment'),
                'hours': len(source)*.01/3600,
                'events': {str(t): events(values, t) for t in plan['threshold_grid']}})
        print(json.dumps({'phase': 'validation_ambient', 'completed': len(ambient['control']),
                          'hours': sum(r['hours'] for r in ambient['control'])}), flush=True)
    arms = {arm: {'grid': [summarize(clips[arm], ambient[arm], t) for t in plan['threshold_grid']]}
            for arm in ARMS}
    for arm in ARMS:
        arms[arm]['reference_0_97'] = next(r for r in arms[arm]['grid'] if r['threshold'] == .97)
    result = {'scope': __doc__, 'plan_sha256': sha(args.plan), 'model_sha256': plan['model_sha256'],
        'arm_definitions': {'control': 'Binary output of the same dual INT8 model',
                            'phonetic': 'Binary times absolute CTC P([hey,nyx]); no forced choice'},
        **operating_points(arms, .97), 'arms': arms, 'accepted': False,
        'limits': plan['limits']}
    (args.output/'ambient-details.json').write_text(json.dumps(ambient, indent=2)+'\n')
    (args.output/'result.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({'phase': 'complete', 'advance': result['advance'],
        'reference_0_97': {arm: arms[arm]['reference_0_97'] for arm in ARMS}}), flush=True)


if __name__ == '__main__':
    main()
