"""Training-only hierarchical sampling; waveform hashes are not human speakers."""
from collections import defaultdict

import numpy as np


def hierarchical_probabilities(rows, train_ids):
    """Equal engines/classes/sources/renders, then equal crops within a render.

    Construct the hierarchy only from this fold's training IDs. All four intent
    classes must exist in every retained engine. No inferred pronunciation,
    model score, reserved row or source outside the fold can set a weight.
    """
    ids = np.asarray(train_ids)
    if (ids.ndim != 1 or not np.issubdtype(ids.dtype, np.integer) or not len(ids)
            or len(np.unique(ids)) != len(ids) or ids.min() < 0 or ids.max() >= len(rows)):
        raise ValueError('Expected distinct valid training IDs')
    tree = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))
    identities = {}
    for position, index in enumerate(ids):
        row = rows[index]
        engine, label, source, variant = (row[k] for k in ('engine', 'class', 'sha256', 'variant'))
        if (not isinstance(engine, str) or not engine or type(label) is not int
                or label not in range(4) or not isinstance(source, str) or len(source) != 64
                or any(c not in '0123456789abcdef' for c in source)
                or type(variant) is not int or variant < 0
                or row.get('split', 'training') != 'training'):
            raise ValueError('Invalid training source identity')
        identity = (engine, label)
        if identities.setdefault(source, identity) != identity:
            raise ValueError('A waveform hash has conflicting engine or intent')
        tree[engine][label][source][variant].append(position)
    probability = np.zeros(len(ids), np.float64)
    for classes in tree.values():
        if set(classes) != set(range(4)):
            raise ValueError('Every training engine must cover all four classes')
        for sources in classes.values():
            for variants in sources.values():
                for positions in variants.values():
                    probability[positions] = 1 / (len(tree)*4*len(sources)*len(variants)*len(positions))
    if not np.isclose(probability.sum(), 1, rtol=0, atol=1e-14):
        raise ValueError('Hierarchy probabilities do not sum to one')
    # Normalize floating reduction roundoff only; no probability clipping.
    return probability/probability.sum()


def distribution(rows, train_ids, probabilities):
    """Describe marginal mass and source concentration without quality scores."""
    ids = np.asarray(train_ids)
    p = np.full(len(ids), 1/len(ids)) if probabilities is None else np.asarray(probabilities)
    if p.shape != ids.shape or not np.isfinite(p).all() or np.any(p <= 0) or not np.isclose(p.sum(), 1):
        raise ValueError('Invalid sampling probabilities')
    groups = defaultdict(list)
    for index, probability in zip(ids, p, strict=True):
        row = rows[index]
        groups[f"{row['engine']}:{row['class']}"].append((row, float(probability)))
    result = {}
    for group, values in sorted(groups.items()):
        mass = sum(p for _, p in values)
        source_mass = defaultdict(float)
        for row, value in values:
            source_mass[row['sha256']] += value
        result[group] = {'crops': len(values), 'source_hashes': len(source_mass),
            'probability_mass': mass,
            'minimum_source_conditional_mass': min(source_mass.values())/mass,
            'maximum_source_conditional_mass': max(source_mass.values())/mass,
            'effective_source_hashes': mass**2/sum(v*v for v in source_mass.values())}
    return result
