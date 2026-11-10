"""Calibrate a frozen two-head equal mean on existing validation scores only.

Both heads must use int8 sigmoid probability bytes and three-step smoothing.
There are no fitted ensemble weights or changes to member outputs/data.
"""
from pathlib import Path
import argparse
import hashlib
import json

import numpy as np
from ai_edge_litert.interpreter import Interpreter, OpResolverType
from evaluate_head import Head, summarize, events, eligible


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def equal_mean(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Ensemble score shape/value mismatch')
    for value in (a, b):
        if (np.abs(value * 768 - np.rint(value * 768)) > 1e-9).any():
            raise ValueError('Expected three-step averages of int8 probability bytes')
        if (value < 0).any() or (value > 255 / 256 + 1e-12).any():
            raise ValueError('Probability is outside the frozen int8 sigmoid range')
    # Integer numerator avoids different floating addition orders at thresholds.
    return np.rint((a + b) * 768).astype(np.int64) / 1536


class EqualMeanHead:
    def __init__(self, paths):
        if len(paths) != 2:
            raise ValueError('Exactly two equal-weight members are required')
        self.members = [Head(path) for path in paths]
        if any(h.do['quantization'] != (1 / 256, -128) or h.do['dtype'] != np.int8 for h in self.members):
            raise ValueError('Expected int8 sigmoid probability bytes')

    def scores(self, embeddings, smoothing):
        if smoothing != 3:
            raise ValueError('The exact mean grid is bound to three-step smoothing')
        a, ta = self.members[0].scores(embeddings, smoothing)
        b, tb = self.members[1].scores(embeddings, smoothing)
        if not np.array_equal(ta, tb):
            raise ValueError('Ensemble member timebases differ')
        return equal_mean(a, b), ta


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    plan = json.loads(args.plan.read_text())
    if len(plan['members']) != 2 or plan['event_policy']['score_average_steps'] != 3 or plan['selection_policy'] != 'continuous':
        raise ValueError('Expected the frozen two-member three-step continuous-policy recipe')
    roots, details = [], []
    for member in plan['members']:
        checkpoint = (args.data_root / member['checkpoint']).resolve()
        if not checkpoint.is_relative_to(args.data_root.resolve()):
            raise ValueError('Checkpoint is outside data root')
        for file, key in [('head.tflite', 'head_sha256'), ('validation/calibration.json', 'calibration_sha256'),
                          ('validation/details-private.json', 'details_sha256')]:
            if digest(checkpoint / file) != member[key]:
                raise ValueError('Frozen member input identity mismatch')
        files = sorted((checkpoint / 'validation').glob('*-scores.npy'))
        if {f.name: digest(f) for f in files} != member['continuous_scores']:
            raise ValueError('Frozen continuous score identity mismatch')
        interpreter = Interpreter(model_path=str(checkpoint / 'head.tflite'), experimental_op_resolver_type=OpResolverType.BUILTIN_REF)
        interpreter.allocate_tensors()
        output = interpreter.get_output_details()[0]
        if output['dtype'] != np.int8 or output['quantization'] != (1 / 256, -128):
            raise ValueError('Member output is not a sigmoid probability byte')
        roots.append(checkpoint / 'validation')
        details.append(json.loads((checkpoint / 'validation/details-private.json').read_text()))
    left, right = details
    if len(left['clips']) != len(right['clips']) or len(left['continuous']) != len(right['continuous']):
        raise ValueError('Member validation coverage mismatch')
    clips, continuous = [], []
    thresholds = plan['event_policy']['threshold_grid']
    for a, b in zip(left['clips'], right['clips']):
        if {k: v for k, v in a.items() if k not in ('scores', 'max_score')} != {k: v for k, v in b.items() if k not in ('scores', 'max_score')}:
            raise ValueError('Clip identities, labels or timebases differ')
        scores = equal_mean(a['scores'], b['scores'])
        times = np.asarray(a['times'])
        valid = ((times >= a['keyword_end'] - .1) & (times <= a['keyword_end'] + .8)) if a['label'] else np.ones(len(times), bool)
        clips.append({**a, 'scores': scores.tolist(), 'max_score': float(scores[valid].max()) if valid.any() else 0.})
    for a, b in zip(left['continuous'], right['continuous']):
        if {k: v for k, v in a.items() if k != 'events'} != {k: v for k, v in b.items() if k != 'events'}:
            raise ValueError('Continuous validation identities or coverage differ')
        name = a['id'] + '-scores.npy'
        scores = equal_mean(np.load(roots[0] / name), np.load(roots[1] / name))
        continuous.append({**a, 'events': {str(t): events(scores, t) for t in thresholds}})
    grid = [summarize(clips, continuous, threshold) for threshold in thresholds]
    pool = [row for row in grid if row['continuous_events_per_hour'] <= .5]
    selected = max(pool, key=lambda r: (r['macro_recall'], -r['continuous_events_per_hour'], -r['max_corpus_false_rate'])) if pool else None
    result = {'scope': 'Validation-only frozen equal mean; no development/reserved/physical acceptance',
              'plan_sha256': digest(args.plan), 'members': plan['members'],
              'selected': selected, 'strict_operating_point': bool(selected and eligible(selected)),
              'grid': grid, 'accepted': False}
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'calibration.json').write_text(json.dumps(result, indent=2) + '\n')
    (args.output / 'details-private.json').write_text(json.dumps({'clips': clips, 'continuous': continuous}) + '\n')
    print(json.dumps({'selected': selected, 'strict_operating_point': result['strict_operating_point']}))


if __name__ == '__main__':
    main()
