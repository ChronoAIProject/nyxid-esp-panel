"""Audit floating mean boundary decisions on existing INT8 validation traces only."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np
from evaluate import Detector, event_indices


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('model', 'english-validation', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--negative-validation', type=Path, action='append', required=True)
    p.add_argument('--threshold', type=float, action='append', required=True)
    a = p.parse_args()
    if a.output.exists() or any(not np.isfinite(t) or not 0 < t <= 1 for t in a.threshold):
        p.error('Fresh output and finite thresholds in (0,1] required')
    model_sha = sha(a.model)
    base = json.loads((a.english_validation/'report.json').read_text())
    if not base.get('validation_only') or base['model_sha256'] != model_sha:
        raise ValueError('Need matching validation-only English report')
    detector = Detector(a.model)
    scale = detector.output_scale
    window = base['sliding_window']
    if scale != 1/256 or window != 5:
        raise ValueError('This diagnostic is defined for 1/256 output quantum and five-frame mean')
    summaries = defaultdict(lambda: dict(traces=0, samples=0, changed_samples=0,
                                         changed_event_traces=0, original_events=0, canonical_events=0))
    differences, inputs = [], []

    def audit(values, identity, kind, times=None, keyword_end=None):
        values = np.asarray(values, dtype=np.float64)
        units = np.rint(values*window/scale)
        if (not len(values) or not np.isfinite(values).all()
                or np.max(np.abs(values*window/scale-units)) > 1e-9):
            raise ValueError('Trace is not compatible with quantized mean lattice')
        canonical = units*scale/window
        for threshold in a.threshold:
            old = event_indices(values, threshold)
            new = event_indices(canonical, threshold)
            g = summaries[(kind, threshold)]
            g['traces'] += 1
            g['samples'] += len(values)
            g['changed_samples'] += int(np.sum(values != canonical))
            g['changed_event_traces'] += int(old != new)
            g['original_events'] += len(old)
            g['canonical_events'] += len(new)
            proxy = None
            if times is not None:
                hit = lambda steps: any(keyword_end-.1 <= times[i] <= keyword_end+.8 for i in steps)
                old_hit, new_hit = hit(old), hit(new)
                g['original_proxy_hits'] = g.get('original_proxy_hits', 0)+int(old_hit)
                g['canonical_proxy_hits'] = g.get('canonical_proxy_hits', 0)+int(new_hit)
                proxy = dict(original_hit=old_hit, canonical_hit=new_hit)
            if old != new:
                differences.append(dict(identity=identity, kind=kind, threshold=threshold,
                                        original_event_steps=old, canonical_event_steps=new, proxy_window=proxy))

    path = a.english_validation/'clip-details.json'
    inputs.append(dict(path=str(path), sha256=sha(path)))
    for row in json.loads(path.read_text())['validation']:
        audit(row['scores'], dict(source=row['source'], variant=row['variant'], label=row['label']),
              'positive' if row['label'] else 'confusing-negative',
              row['times'] if row['label'] else None, row['keyword_end'])
    for folder in a.negative_validation:
        path = folder/'report.json'
        report = json.loads(path.read_text())
        if (report.get('split') != 'validation' or report.get('independent_final_scored') is not False
                or report.get('development_regression_scored') is not False
                or report['model_sha256'] != model_sha or report['sliding_window'] != window):
            raise ValueError('Need same-model negative validation')
        inputs.append(dict(path=str(path), sha256=sha(path)))
        for row in report['speakers']:
            path = folder/(row['speaker']+'-scores.npy')
            if sha(path) != row['score_sha256']:
                raise ValueError('Changed negative trace')
            values = np.load(path, allow_pickle=False)
            if len(event_indices(values, report['threshold'])) != row['false_events']:
                raise ValueError('Original event recount failed')
            audit(values, row['speaker'], folder.name)
    result = dict(role='Existing INT8 mean floating-boundary diagnostic; no scoring-policy change',
                  model_sha256=model_sha, source_sha256=sha(Path(__file__)), inputs=inputs,
                  output_scale=scale, window=window,
                  canonical_formula='round(smoothed_score * window / output_scale) * output_scale / window',
                  summaries=[dict(kind=k, threshold=t, **v) for (k,t),v in summaries.items()],
                  differences=differences, independent_final_audio_accessed=False,
                  deployed_detector_changed=False,
                  limits='Reconstructs mean lattice from existing stored smoothed INT8 scores after strict lattice checks. Canonical event differences are diagnostic, not a reselected threshold, corrected acceptance result or MCU equivalence proof. Counts all events including out-of-proxy-window positives and separately recounts fixed proxy-window hits. Original reports remain authoritative for their original policy.')
    a.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result['summaries']), flush=True)


if __name__ == '__main__':
    main()
