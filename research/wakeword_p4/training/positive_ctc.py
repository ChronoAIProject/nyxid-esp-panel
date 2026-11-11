"""Opt-in compact keyword targets for existing, complete ordinary positives."""
from collections import Counter
import hashlib
import json

import numpy as np


def eligible(row, end, frames):
    """Use the selected crop, without resampling or relabeling truncated crops."""
    if row['split'] != 'training' or row['label'] != 1:
        raise ValueError('Ordinary positive CTC requires training positive rows')
    return end-frames <= row['lead_frame'] and end >= row['keyword_frame']+3


def mark(labels, selections, frames):
    """Append eligibility before shuffle; pair replacements never get this loss."""
    y = np.asarray(labels)
    if (y.ndim != 2 or y.shape[1] != 3 or len(selections) > len(y)
            or not np.isfinite(y).all() or np.any(y[:len(selections), 0] != 1)
            or np.any(y[:len(selections), 2] != 0)):
        raise ValueError('Positive markers require positive, nonambient initial slots')
    flags = np.zeros((len(y), 1), dtype=y.dtype)
    for slot, (row, end) in enumerate(selections):
        complete = eligible(row, end, frames)
        if y[slot, 1] == 0:
            flags[slot] = complete
    return np.concatenate((y, flags), axis=1)


def mask(labels):
    import tensorflow as tf
    tf.debugging.assert_rank(labels, 2)
    tf.debugging.assert_equal(tf.shape(labels)[1], 4)
    flags = labels[:, 3]
    tf.debugging.assert_all_finite(flags, 'Nonfinite positive markers')
    tf.debugging.assert_equal(tf.reduce_all((flags == 0) | (flags == 1)), True)
    selected = flags == 1
    tf.debugging.assert_equal(tf.boolean_mask(labels[:, 0], selected), 1.)
    tf.debugging.assert_equal(tf.boolean_mask(labels[:, 1:3], selected), 0.)
    return selected


def keyword_loss(logits):
    """Mean CTC NLL per target token; an empty eligible cohort contributes zero."""
    import tensorflow as tf
    tf.debugging.assert_rank(logits, 3)
    tf.debugging.assert_greater_equal(tf.shape(logits)[1], 2)
    tf.debugging.assert_greater(tf.shape(logits)[2], 2)
    count = tf.shape(logits)[0]

    def nonempty():
        # Match the paired head's padded label width as well as its length
        # semantics; mixed dense widths trigger a TF 2.21 graph optimizer bug.
        labels = tf.pad(tf.tile([[1, 2]], [count, 1]), [[0, 0], [0, tf.shape(logits)[1]-2]])
        losses = tf.nn.ctc_loss(labels, logits,
            tf.fill([count], 2), tf.fill([count], tf.shape(logits)[1]),
            logits_time_major=False, blank_index=0)
        return tf.reduce_mean(losses)/2

    return tf.cond(count > 0, nonempty, lambda: tf.zeros((), dtype=logits.dtype))


class Exposure:
    """Bind actual unpaired draws/crops to an independently auditable trace."""
    def __init__(self, records, path):
        self.indices = {id(row): index for index, row in enumerate(records)}
        self.path = path
        self.stream = path.open('xb')
        self.digest = hashlib.sha256()
        self.counts = Counter()
        self.sources = set()
        self.renders = set()
        self.steps = 0

    def record(self, step, labels, selections, samples):
        draws = []
        for slot, (row, end) in enumerate(selections):
            if labels[slot, 1] != 0:
                continue
            selected = bool(labels[slot, 3])
            key = (row['engine'], row['text'], selected)
            self.counts[key] += 1
            if selected:
                self.sources.add(row['sha256'])
                self.renders.add((row['sha256'], row['variant']))
            draws.append([self.indices[id(row)], end, int(selected),
                          hashlib.sha256(samples[slot].astype('<u2', copy=False).tobytes()).hexdigest()])
        data = (json.dumps({'step': step, 'draws': draws}, separators=(',', ':'))+'\n').encode()
        self.stream.write(data)
        self.digest.update(data)
        self.steps += 1

    def metadata(self):
        self.stream.flush()
        return {'steps': self.steps, 'examples': sum(self.counts.values()),
                'eligible_examples': sum(n for (_, _, selected), n in self.counts.items() if selected),
                'eligible_distinct_sources': len(self.sources), 'eligible_distinct_renders': len(self.renders),
                'counts': [{'engine': e, 'text': text, 'eligible': selected, 'examples': n}
                           for (e, text, selected), n in sorted(self.counts.items())],
                'trace_file': self.path.name, 'trace_sha256': self.digest.hexdigest(),
                'trace_encoding': 'One JSON line per training step; draws [feature_index, crop_end, eligible, uint16_crop_sha256] in unpaired positive slot order, before augmentation/shuffle.'}
