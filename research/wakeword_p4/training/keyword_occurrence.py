"""CTC probability of a contiguous [hey, nyx] occurrence anywhere in a window.

Arbitrary collapsed symbols may precede/follow the occurrence. Blank zero and
consecutive CTC repeats retain their usual meaning. This sums an event, not
a forced-choice phrase posterior or calibrated human wake confidence.
"""
import numpy as np

from compact_sequence_inference import Detector as ExactDetector


def log_probability(logits):
    values = np.asarray(logits, np.float64)
    if values.ndim != 3 or values.shape[-1] != 6 or not np.isfinite(values).all():
        raise ValueError('Expected finite [batch,time,6] logits')
    shifted = values-values.max(-1, keepdims=True)
    logp = shifted-np.log(np.exp(shifted).sum(-1, keepdims=True))
    none = np.zeros(len(values))
    hey = np.full(len(values), -np.inf)
    found = np.full(len(values), -np.inf)
    for frame in logp.transpose(1, 0, 2):
        # Distinct keyword tokens need only three states: no pending hey,
        # collapsed suffix hey, and absorbing match. Repeated hey or blanks
        # preserve the pending suffix; mix/next/nick break it.
        next_none = np.logaddexp(none+np.logaddexp.reduce(frame[:, [0, 2, 3, 4, 5]], axis=-1),
                                hey+np.logaddexp.reduce(frame[:, 3:], axis=-1))
        next_hey = np.logaddexp(none+frame[:, 1], hey+np.logaddexp(frame[:, 0], frame[:, 1]))
        found = np.logaddexp(found, hey+frame[:, 2])
        none, hey = next_none, next_hey
    return np.minimum(found, 0.)  # Bound only accumulated floating roundoff.


class OccurrenceDetector(ExactDetector):
    """Preserve the original INT8 input/state path; change only the CTC event."""
    def predict(self, raw):
        binary, combined = [], []
        for at in range(0, len(raw)-2, 768):
            p, logits = self.raw_outputs(raw[at:at+768])
            binary.extend(p)
            combined.extend(p*np.exp(log_probability(logits)))
        return np.asarray(binary, np.float32), np.asarray(combined, np.float64)
