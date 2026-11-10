"""Training-only time/frequency masking using an independent RNG."""
import numpy as np


def mask_spectrograms(x, rng, time_width=0, frequency_width=0):
    if not time_width and not frequency_width:
        return x
    if x.ndim != 3 or not 0 <= time_width <= x.shape[1] or not 0 <= frequency_width <= x.shape[2]:
        raise ValueError("Mask widths must fit a [batch,time,frequency] array")
    out = x.copy()
    for row in out:
        for axis, maximum in ((0, time_width), (1, frequency_width)):
            if not maximum:
                continue
            width = int(rng.integers(maximum + 1))
            if not width:
                continue
            start = int(rng.integers(row.shape[axis] - width + 1))
            if axis == 0:
                row[start:start + width, :] = 0
            else:
                row[:, start:start + width] = 0
    return out
