"""Frozen same-weight float/INT8 development diagnostic, without refitting.

Fresh INT8 inference must replay the original audited traces exactly. Float
sees the same actual feature bytes, source resets and timestamps.
No score scale, threshold extension, deployment decision or audio access.
"""
import argparse
import json
from pathlib import Path
import shutil

import numpy as np

from evaluate import events, smooth
from evaluate_compact_sequence import summarize
from phrase_ctc_readout import sha
from scalar_precision import Detector


def paired_counts(clips, threshold):
    result = {}
    for label, name in ((1, 'positive'), (0, 'negative')):
        cohort = [row for row in clips if row['label'] == label]
        buckets = {key: 0 for key in ('both_above', 'float_only_above', 'int8_only_above', 'neither_above')}
        for row in cohort:
            f, q = (row['arms'][p]['max_score'] >= threshold for p in ('float', 'int8'))
            key = 'both_above' if f and q else 'float_only_above' if f else 'int8_only_above' if q else 'neither_above'
            buckets[key] += 1
        result[name] = {'views': len(cohort), **buckets}
    return result


def strict_point(grid):
    return max((point for point in grid if point['ambient_false_events'] == 0
                and max(point['max_source_false_rate'], point['max_corpus_false_rate']) <= .01),
               key=lambda point: (point['macro_recall'], -point['threshold']), default=None)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('plan', 'export', 'models', 'features', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Use a fresh precision-diagnostic directory')
    plan = json.loads(args.plan.read_bytes())
    for name, digest in plan['source_sha256'].items():
        require(sha(Path(__file__).parent/name) == digest, 'Executed source differs: '+name)
    require(sha(args.export/'result.json') == plan['export_result_sha256'], 'Export result differs')
    smoke = json.loads((args.export/'result.json').read_bytes())
    require(smoke['advance_to_precision_diagnostic'], 'Numerical export gate failed')
    for name, digest in plan['parent_files_sha256'].items():
        require(sha(args.models/name) == digest, 'Original audited INT8 artifact differs: '+name)
    original_audit = json.loads((args.models/'cached-score-audit.json').read_bytes())
    require(original_audit['all_maxima_timebases_events_grids_and_gate_reproduced'],
            'Original cached INT8 score audit failed')
    for name, digest in plan['feature_index_sha256'].items():
        require(sha(args.features/name) == digest, 'Feature index differs: '+name)
    require(sha(args.features/'features.bin') == plan['feature_data_sha256'], 'Full scalar cache differs')
    for name, digest in plan['continuous_feature_sha256'].items():
        require(sha(args.features/name) == digest, 'Continuous feature bytes differ: '+name)
    require(plan['event_policy'] == {'smooth_steps': 5, 'refractory_steps': 67, 'rearm_fraction': .5},
            'Unreviewed event policy')
    rows = json.loads((args.features/'features.json').read_bytes())
    selected = [row for row in rows if row['split'] == 'validation']
    reserved = {row['sha256'] for row in rows if row['split'] != 'validation'}
    require(all(row['sha256'] not in reserved for row in selected), 'Validation source overlaps another split')
    raw = np.memmap(args.features/'features.bin', mode='r', dtype='<u2').reshape(-1, 40)
    ambient_index = json.loads((args.features/'human.json').read_bytes())
    ambient_index += json.loads((args.features/'ambient.json').read_bytes())
    selected_ambient = [(i, row) for i, row in enumerate(ambient_index) if row['split'] == 'validation']
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/'plan.json').write_bytes(args.plan.read_bytes())
    result = {'scope': __doc__, 'plan_sha256': sha(args.plan), 'arms': {}, 'accepted': False,
              'limits': plan['limits']}
    for arm in plan['model_files_sha256']:
        models = plan['model_files_sha256'][arm]
        model_path = args.export/f'{arm}-float.tflite'
        require(sha(model_path) == models['float'] == smoke['arms'][arm]['float_model_sha256'],
                'Float export differs: '+arm)
        require(sha(args.models/f'model-{arm}/hey_nyx.tflite') == models['int8'], 'INT8 model differs: '+arm)
        original = json.loads((args.models/f'validation-{arm}/validation-details.json').read_bytes())
        calibration = json.loads((args.models/f'validation-{arm}/threshold.json').read_bytes())
        detector = Detector(model_path)
        quantized = Detector(args.models/f'model-{arm}/hey_nyx.tflite')
        clips = {p: [] for p in ('float', 'int8')}
        ambient = {p: [] for p in clips}
        paired, clip_steps, continuous_steps = [], 0, 0
        delta_sum, delta_max, delta_steps = 0., 0., 0
        with (args.output/f'clips-{arm}.jsonl').open('x') as log:
            for number, (row, cached) in enumerate(zip(selected, original['validation'], strict=True)):
                require(cached['source'] == row['path']
                        and all(cached[k] == row[k] for k in ('engine', 'variant', 'label', 'text')),
                        'Original clip identity differs')
                source = raw[row['offset']:row['offset']+row['frames']]
                detector.reset()
                f = smooth(detector.predict(source), 5)
                quantized.reset()
                q = smooth(quantized.predict(source), 5)
                require(np.array_equal(q, cached['scores']), 'Fresh INT8 clip replay differs: '+row['path'])
                times = (3*(np.arange(len(f))+4)+5)*.01
                require(f.shape == q.shape and np.isfinite(f).all()
                        and np.all((f >= 0) & (f <= 1)) and np.array_equal(times, cached['times']),
                        'Float/INT8 clip timebase differs')
                end = (row['keyword_frame']+2)*.01 if row['label'] else None
                mask = ((times >= end-.1) & (times <= end+.8) if end is not None else np.ones(len(f), bool))
                record = {'source': row['path'], 'sha256': row['sha256'],
                    **{k: row[k] for k in ('engine', 'variant', 'label', 'text')},
                    'times': times.tolist(), 'keyword_end': end, 'arms': {}}
                for precision, values in (('float', f), ('int8', q)):
                    maximum = float(values[mask].max()) if mask.any() else 0.
                    if precision == 'int8':
                        require(maximum == cached['max_score'], 'Original INT8 maximum differs')
                    record['arms'][precision] = {'max_score': maximum, 'scores': values.tolist()}
                    clips[precision].append({'engine': row['engine'], 'source': row['path'],
                                            'label': row['label'], 'max_score': maximum})
                log.write(json.dumps(record, separators=(',', ':'))+'\n')
                paired.append({'label': row['label'], 'arms': record['arms']})
                delta = np.abs(f-q)
                delta_sum += float(delta.sum()); delta_max = max(delta_max, float(delta.max())); delta_steps += len(f)
                clip_steps += len(f)
                if number % 120 == 0 or number == len(selected)-1:
                    print(json.dumps({'phase': 'clips', 'arm': arm, 'completed': number+1,
                                      'total': len(selected)}), flush=True)
        ambient_details = []
        for ordinal, ((index, row), cached) in enumerate(zip(selected_ambient, original['ambient'], strict=True)):
            source = np.load(args.features/row['path'], allow_pickle=False, mmap_mode='r')
            detector.reset()
            f = smooth(detector.predict(source), 5)
            original_array = args.models/f'validation-{arm}/ambient-validation-{index:03d}.npy'
            quantized.reset()
            q = smooth(quantized.predict(source), 5)
            require(np.array_equal(q, np.load(original_array, allow_pickle=False)),
                    'Fresh continuous INT8 replay differs: '+row['path'])
            require(f.shape == q.shape and np.isfinite(f).all() and np.all((f >= 0) & (f <= 1)),
                    'Continuous float/INT8 shape differs')
            hours = len(source)*.01/3600
            source_type = row.get('source_type', 'speech' if 'speaker' in row else 'environment')
            require(cached['hours'] == hours and cached['source_type'] == source_type,
                    'Continuous cached source/exposure differs')
            require(sha(args.features/row['path']) == plan['continuous_feature_sha256'][row['path']],
                    'Continuous feature bytes changed during inference')
            entry = {'feature_path': row['path'], 'feature_sha256': plan['continuous_feature_sha256'][row['path']],
                     'hours': hours, 'source_type': source_type, 'arms': {}}
            for precision, values in (('float', f), ('int8', q)):
                name = f'ambient-{arm}-{precision}-{index:03d}.npy'
                if precision == 'float':
                    np.save(args.output/name, values)
                else:
                    shutil.copyfile(original_array, args.output/name)
                    require(sha(args.output/name) == sha(original_array), 'INT8 trace copy differs')
                counts = {str(t): events(values, t) for t in plan['threshold_grid']}
                if precision == 'int8':
                    require(counts == cached['events'], 'Cached INT8 events differ')
                ambient[precision].append({'hours': hours, 'source_type': source_type, 'events': counts})
                entry['arms'][precision] = {'array': name, 'sha256': sha(args.output/name), 'events': counts}
            ambient_details.append(entry)
            delta = np.abs(f-q)
            delta_sum += float(delta.sum()); delta_max = max(delta_max, float(delta.max())); delta_steps += len(f)
            continuous_steps += len(f)
            print(json.dumps({'phase': 'continuous', 'arm': arm, 'completed': ordinal+1,
                              'total': len(selected_ambient)}), flush=True)
        grids = {p: [summarize(clips[p], ambient[p], t) for t in plan['threshold_grid']] for p in clips}
        # The same independent summary already reconstructed these cached
        # INT8 grids; tolerate only floating summation order, never counts.
        for computed, cached in zip(grids['int8'], calibration['grid'], strict=True):
            for key in ('macro_recall', 'ambient_false_events', 'max_source_false_rate', 'max_corpus_false_rate'):
                require(abs(computed[key]-cached[key]) <= 1e-15, 'Original INT8 grid differs: '+key)
        (args.output/f'ambient-{arm}.json').write_text(json.dumps(ambient_details, indent=2)+'\n')
        result['arms'][arm] = {'models_sha256': models, 'grids': grids,
            'strict_gate': {p: strict_point(grid) for p, grid in grids.items()},
            'paired_views_at_reference_0_97': paired_counts(paired, .97),
            'clip_count': len(selected), 'clip_smoothed_steps': clip_steps,
            'continuous_source_count': len(selected_ambient), 'continuous_smoothed_steps': continuous_steps,
            'all_smoothed_steps': delta_steps, 'max_absolute_precision_delta': delta_max,
            'mean_absolute_precision_delta': delta_sum/delta_steps,
            'all_fresh_int8_scores_replay_original_exactly': True, 'cached_int8_grid_reconstructed': True}
        print(json.dumps({'phase': 'arm_complete', 'arm': arm,
                          'float_has_strict_point': result['arms'][arm]['strict_gate']['float'] is not None,
                          'int8_has_strict_point': result['arms'][arm]['strict_gate']['int8'] is not None}), flush=True)
    (args.output/'result.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({'phase': 'complete', 'accepted': False}), flush=True)


if __name__ == '__main__':
    main()
