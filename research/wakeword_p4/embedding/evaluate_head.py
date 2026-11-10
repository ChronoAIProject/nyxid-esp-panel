"""Validation-only INT8 head calibration, with frozen 80 ms event policy."""
from pathlib import Path
import argparse
import hashlib
import json
import time
import numpy as np
from ai_edge_litert.interpreter import Interpreter, OpResolverType
from scipy.stats import chi2
from train_head import MIN_END, HISTORY, load_rows


class Head:
    def __init__(self, path):
        self.interpreter = Interpreter(model_path=str(path), num_threads=1,
                                      experimental_op_resolver_type=OpResolverType.BUILTIN_REF)
        self.interpreter.allocate_tensors()
        self.di = self.interpreter.get_input_details()[0]
        self.do = self.interpreter.get_output_details()[0]
        assert tuple(self.di['shape']) == (1, HISTORY, 96, 1)
        assert self.di['quantization'] == (1/128, 0)
        assert self.di['dtype'] == np.int8

    def scores(self, embeddings, smoothing):
        result = []
        for end in range(MIN_END, len(embeddings)+1):
            x = embeddings[end-HISTORY:end][None, :, :, None]
            self.interpreter.set_tensor(self.di['index'], x)
            self.interpreter.invoke()
            value = self.interpreter.get_tensor(self.do['index'])[0, 0]
            scale, zero = self.do['quantization']
            result.append((int(value)-zero)*scale)
        if len(result) < smoothing:
            return np.empty(0, np.float32), np.empty(0, np.float32)
        scores = np.convolve(result, np.full(smoothing, 1/smoothing), mode='valid')
        ends = np.arange(MIN_END+smoothing-1, len(embeddings)+1)
        assert len(scores) == len(ends)
        return scores, ends*.08


def events(scores, threshold):
    count, armed, last = 0, True, -1000
    for index, value in enumerate(scores):
        if value < threshold*.5 and index-last >= 25:
            armed = True
        if armed and value >= threshold and index-last >= 25:
            count += 1
            last, armed = index, False
    return count


def summarize(clips, continuous, threshold):
    positive = [row for row in clips if row['label']]
    negative = [row for row in clips if not row['label']]
    recalls = {engine: float(np.mean([r['max_score'] >= threshold for r in positive if r['engine'] == engine]))
               for engine in sorted({r['engine'] for r in positive})}
    corpora = {}
    for engine, corpus in sorted({(r['engine'], r['corpus']) for r in negative}):
        group = [r for r in negative if r['engine'] == engine and r['corpus'] == corpus]
        false = sum(r['max_score'] >= threshold for r in group)
        corpora[engine+':'+corpus] = {'variants': len(group), 'false_variants': false, 'false_rate': false/len(group)}
    hours = sum(r['evaluated_hours'] for r in continuous)
    count = sum(r['events'][str(threshold)] for r in continuous)
    delays = []
    for row in positive:
        above = np.asarray(row['scores']) >= threshold
        valid = above & (np.asarray(row['times']) >= row['keyword_end']-.10) & (np.asarray(row['times']) <= row['keyword_end']+.80)
        if valid.any():
            delays.append((np.asarray(row['times'])[valid][0]-row['keyword_end'])*1000)
    return {'threshold': threshold, 'positive_variants': len(positive),
            'recall': float(np.mean([r['max_score'] >= threshold for r in positive])),
            'macro_recall': float(np.mean(list(recalls.values()))), 'recall_by_engine': recalls,
            'negative_variants': len(negative), 'confusing_corpora': corpora,
            'max_corpus_false_rate': max(r['false_rate'] for r in corpora.values()),
            'continuous_hours': hours, 'continuous_events': count, 'continuous_events_per_hour': count/hours,
            'continuous_events_per_hour_upper_95': float(.5*chi2.ppf(.95, 2*(count+1))/hours),
            'positive_latency_median_ms': float(np.median(delays)) if delays else None,
            'positive_latency_p95_ms': float(np.percentile(delays, 95)) if delays else None}


def eligible(row):
    return row['continuous_events_per_hour'] <= .5 and row['max_corpus_false_rate'] <= .01


def measure(checkpoint, rows, plan, feature_inputs):
    output = checkpoint/'validation'
    output.mkdir(exist_ok=False)
    detector = Head(checkpoint/'head.tflite')
    thresholds = plan['event_policy']['threshold_grid']
    clips, continuous = [], []
    started = time.monotonic()
    for number, row in enumerate(rows, 1):
        scores, times = detector.scores(row['array'], plan['event_policy']['score_average_steps'])
        if row['kind'] == 'tts':
            end = row['keyword_end']
            valid = (times >= end-.10) & (times <= end+.80) if row['label'] else np.ones(len(scores), bool)
            clips.append({key: row[key] for key in ('id', 'label', 'engine', 'corpus', 'variant', 'keyword_end')}
                         | {'max_score': float(scores[valid].max()) if valid.any() else 0,
                            'scores': scores.tolist(), 'times': times.tolist()})
        else:
            np.save(output/(row['id']+'-scores.npy'), scores)
            continuous.append({'id': row['id'], 'kind': row['kind'], 'engine': row['engine'],
                               'source_hours': row['seconds']/3600,
                               'evaluated_hours': len(scores)*.08/3600,
                               'events': {str(t): events(scores, t) for t in thresholds}})
        if number % 100 == 0 or number == len(rows):
            print(json.dumps({'phase': 'validation', 'checkpoint': checkpoint.name, 'rows': number,
                              'total': len(rows), 'seconds': time.monotonic()-started}), flush=True)
    grid = [summarize(clips, continuous, t) for t in thresholds]
    strict = [r for r in grid if eligible(r)]
    continuous_only = [r for r in grid if r['continuous_events_per_hour'] <= .5]
    prefer_strict = plan.get('selection_policy', 'strict') == 'strict'
    if plan.get('selection_policy', 'strict') not in ('strict', 'continuous'):
        raise ValueError('Unsupported frozen selection policy')
    pool = (strict if prefer_strict else []) or continuous_only
    selected = max(pool, key=lambda r: (r['macro_recall'], -r['continuous_events_per_hour'], -r['max_corpus_false_rate'])) if pool else min(
        grid, key=lambda r: (r['continuous_events_per_hour'], r['max_corpus_false_rate'], -r['macro_recall']))
    result = {'scope': 'Validation-only checkpoint calibration; no heldout/physical acceptance',
              'model_sha256': hashlib.sha256((checkpoint/'head.tflite').read_bytes()).hexdigest(),
              'backbone_sha256': plan['model_sha256'], 'event_policy': plan['event_policy'],
              'feature_inputs': feature_inputs,
              'selection_policy': plan.get('selection_policy', 'strict'),
              'selection': 'strict' if prefer_strict and strict else 'continuous_diagnostic' if continuous_only else 'no_viable_operating_point',
              'strict_operating_point': eligible(selected), 'selected': selected, 'grid': grid,
              'seconds': time.monotonic()-started}
    (output/'calibration.json').write_text(json.dumps(result, indent=2)+'\n')
    (output/'details-private.json').write_text(json.dumps({'clips': clips, 'continuous': continuous})+'\n')
    print(json.dumps({'phase': 'calibrated', 'checkpoint': checkpoint.name, 'selection': result['selection'], 'selected': selected}), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--plan', type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    training = json.loads((args.training/'complete.json').read_text())
    assert training['plan_sha256'] == hashlib.sha256(args.plan.read_bytes()).hexdigest()
    assert not (args.training/'checkpoint-selection.json').exists()
    rows = load_rows('validation', args.data_root, plan['model_sha256'])
    results = []
    for step in plan['recipe']['checkpoints']:
        checkpoint = args.training/f'step-{step:05d}'
        results.append({'step': step, **measure(checkpoint, rows, plan, training['feature_inputs'])})
    strict = [r for r in results if r['strict_operating_point']]
    continuous = [r for r in results if r['selection'] != 'no_viable_operating_point']
    pool = (strict if plan.get('selection_policy', 'strict') == 'strict' else []) or continuous
    chosen = max(pool, key=lambda r: (r['selected']['macro_recall'], -r['selected']['continuous_events_per_hour'],
                                      -r['selected']['max_corpus_false_rate'], -r['step'])) if pool else None
    summary = {'scope': 'Validation-frozen experimental head; no regression/test/device accuracy acceptance',
               'selection_policy': plan.get('selection_policy', 'strict'),
               'selected_step': chosen['step'] if chosen else None,
               'selected_head_sha256': chosen['model_sha256'] if chosen else None,
               'selected': chosen['selected'] if chosen else None,
               'strict_operating_point': bool(chosen and chosen['strict_operating_point']),
               'results': [{k: v for k, v in r.items() if k != 'grid'} for r in results], 'accepted': False}
    (args.training/'checkpoint-selection.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
