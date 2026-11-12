"""Training-only same-source scalar-logit phrase ranking; no new model layers."""
import tensorflow as tf

PHRASES = ('hey_nyx', 'hey_nick', 'hey_next', 'hey_mix')
ARMS = ('quiet', 'noise25', 'combined', 'noise15', 'rir')

def verified_phrase_groups(records):
    """Return explicit indices [10 groups,4 phrases], authenticating metadata."""
    if len(records) != 40:
        raise ValueError('Expected40 paired records')
    groups = {}
    for i, r in enumerate(records):
        key = (int(r['voice']), float(r['length_multiplier']), r['arm'])
        if r['arm'] not in ARMS or r['target'] not in PHRASES:
            raise ValueError('Unknown rendering or phrase')
        group = groups.setdefault(key, {})
        if r['target'] in group:
            raise ValueError('Duplicate phrase')
        group[r['target']] = i
    voices = {k[0] for k in groups}
    speeds = {k[1] for k in groups}
    if len(voices) != 2 or len(speeds) != 1 or set(groups) != {(v, next(iter(speeds)), a) for v in voices for a in ARMS}:
        raise ValueError('Expected two IDs, one speed, five renderings')
    if any(set(g) != set(PHRASES) for g in groups.values()):
        raise ValueError('Incomplete four-phrase group')
    return [[groups[k][p] for p in PHRASES] for k in sorted(groups)]

def phrase_ranking(logits, group_indices):
    """Mean logsumexp minus target logit over explicit metadata-checked groups."""
    logits = tf.reshape(logits, [40])
    indices = tf.convert_to_tensor(group_indices, dtype=tf.int32)
    tf.debugging.assert_equal(tf.shape(indices), [10, 4])
    tf.debugging.assert_equal(tf.sort(tf.reshape(indices, [-1])), tf.range(40))
    tf.debugging.assert_all_finite(logits, 'Nonfinite phrase logits')
    grouped = tf.gather(logits, indices)
    return tf.reduce_mean(tf.reduce_logsumexp(grouped, axis=1)-grouped[:, 0])
