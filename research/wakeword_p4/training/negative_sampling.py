"""Draw extra training negative windows without advancing the baseline RNG."""
import numpy as np


def candidate_ends(rng, mining_rng, low, high, count):
    if count < 1 or low >= high:
        raise ValueError('Invalid negative window range/count')
    baseline = int(rng.integers(low, high))
    if count == 1:
        return np.asarray([baseline], dtype=np.int64)
    return np.concatenate(([baseline], mining_rng.integers(low, high, size=count-1)))
