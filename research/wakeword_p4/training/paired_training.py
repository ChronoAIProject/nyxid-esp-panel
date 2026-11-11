"""Training-only same-voice wake/confuser pairing and losses."""
from collections import defaultdict
import hashlib
from pathlib import Path

import numpy as np

WAKE = frozenset(('hey nyx', 'hey nix', 'hey nicks'))
CONFUSERS = ('hey mix', 'hey next', 'hey nick')
PAIRS_PER_BATCH = 8


def phrase(row):
    return row['text'].strip().lower().rstrip('.!?')


def voice_key(row):
    # Qwen uses a repeated voice description, not a shared speaker identity.
    if row['engine'] == 'qwen' or not row['speakers'] or len(set(row['speakers'])) != 1:
        return None
    return (row['engine'], row['path'].split('/')[0], row['speakers'][0],
            row.get('preset'), row.get('style'), row.get('rate'), row.get('length_scale'), row['variant'])


def bank(rows, bounds, frames):
    reserved = {r['sha256'] for r in rows if r['split'] != 'training'}
    positive, negative = defaultdict(list), defaultdict(list)
    for row in rows:
        if row['split'] != 'training':
            continue
        group = voice_key(row)
        if group is None:
            continue
        label, text = row['label'], phrase(row)
        if label == 1 and text in WAKE:
            lo = max(frames, row['keyword_frame'] + 3)
            hi = min(row['frames'], row['keyword_frame'] + 27, row['lead_frame'] + frames - 1)
            band = (lo, hi) if lo <= hi else None
            choices = positive
        elif label == 0 and text in CONFUSERS:
            band = bounds.get((row['path'], row['variant']))
            choices = negative
        else:
            continue
        if band is not None:
            path = Path(row['path'])
            if row['sha256'] in reserved or path.is_absolute() or '..' in path.parts:
                raise ValueError('Invalid pair source provenance')
            choices[group].append((row, band))
    result = defaultdict(list)
    for group, negatives in negative.items():
        positives = positive.get(group)
        if positives:
            for n in negatives:
                partners = [p for p in positives if p[0]['sha256'] != n[0]['sha256']]
                if partners:
                    result[(group[0], phrase(n[0]))].append((n, partners))
    if not result or any({k[1] for k in result if k[0] == engine} != set(CONFUSERS)
                         for engine in {k[0] for k in result}):
        raise ValueError('Pair bank requires all confusers for each retained engine')
    return dict(sorted(result.items()))


class Sampler:
    def __init__(self, rows, bounds, frames):
        self.bank = bank(rows, bounds, frames)
        self.strata = list(self.bank)
        self.frames = frames
        self.rng = np.random.default_rng(381)
        self.digest = hashlib.sha256()
        self.draws = 0

    def apply(self, batch, sequence):
        classes = np.zeros(len(batch), dtype=np.int32)
        for slot in range(PAIRS_PER_BATCH):
            group = self.bank[self.strata[int(self.rng.integers(len(self.strata)))]]
            (negative, nband), positives = group[int(self.rng.integers(len(group)))]
            positive, pband = positives[int(self.rng.integers(len(positives)))]
            classes[slot] = 1
            classes[slot + 32] = CONFUSERS.index(phrase(negative)) + 2
            phase = int(self.rng.integers(256))
            for row, band, index in ((positive, pband, slot), (negative, nband, slot + 32)):
                lo, hi = band
                end = lo + phase * (hi - lo + 1) // 256
                batch[index] = sequence(row)[end - self.frames:end]
                self.digest.update(f"{row['sha256']}:{row['variant']}:{end}\n".encode())
            self.draws += 1
        return classes

    def metadata(self):
        return {'draws': self.draws, 'schedule_sha256': self.digest.hexdigest(),
                'eligible_negative_views_by_stratum': {':'.join(k): len(v) for k, v in self.bank.items()}}


def training_loss(gain):
    import tensorflow as tf

    def paired_loss(y_true, y_pred):
        labels, pair = y_true[:, :1], tf.cast(y_true[:, 1], tf.int32)
        bce = tf.keras.losses.binary_crossentropy(labels, y_pred)
        if not gain:
            return bce
        p = tf.clip_by_value(y_pred[:, 0], 1e-6, 1 - 1e-6)
        odds = tf.math.log(p) - tf.math.log1p(-p)
        sign = 1 - 2 * labels[:, 0]
        # Each nonzero ID contributes exactly one negative minus positive
        # log odds. Ignore bucket zero, the ordinary BCE examples.
        gap = tf.math.unsorted_segment_sum(odds * sign, pair, PAIRS_PER_BATCH + 1)
        penalty = tf.nn.softplus(4 + tf.gather(gap, pair))
        positive_pair = tf.cast(pair > 0, p.dtype) * labels[:, 0]
        return bce + gain * positive_pair * penalty

    return paired_loss
