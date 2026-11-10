"""Auditable ambient stream sampling using original exposure, excluding repeats."""
from collections import defaultdict
import numpy as np


def probabilities(rows):
    seconds = np.asarray([r.get('source_seconds', r['seconds']) for r in rows], dtype=float)
    if not len(seconds) or not np.isfinite(seconds).all() or (seconds <= 0).any():
        raise ValueError('Ambient sources require positive finite original exposure')
    return seconds/seconds.sum()


def coverage(rows):
    weights = probabilities(rows)
    groups = defaultdict(lambda: dict(rows=0, original_hours=0., row_probability=0., duration_probability=0.))
    for r, w in zip(rows, weights):
        name = ('human' if 'speaker' in r or r.get('kind') == 'human' else 'ambient')+':'+str(r.get('category', 'unspecified'))
        g = groups[name]
        g['rows'] += 1
        g['original_hours'] += r.get('source_seconds', r['seconds'])/3600
        g['row_probability'] += 1/len(rows)
        g['duration_probability'] += float(w)
    return dict(groups)


def stratified_probabilities(rows):
    """Equal source-category mass; original-duration weighting inside each stratum."""
    duration = probabilities(rows)
    keys = [('human' if 'speaker' in r or r.get('kind') == 'human' else 'ambient',
             str(r.get('category', 'unspecified'))) for r in rows]
    totals = defaultdict(float)
    for key, weight in zip(keys, duration):
        totals[key] += float(weight)
    return np.asarray([w / totals[k] / len(totals) for k, w in zip(keys, duration)])
