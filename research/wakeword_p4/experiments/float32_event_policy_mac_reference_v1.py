"""Opt-in float reference: preserve original event state machine, not UINT8 rounding.

Independent Mac reference for a prospectively locked raw-FP32 diagnostic. This
does not replace any historical evaluator or activate recognition collection.
"""
import math
import numbers

import numpy as np

EVENT_ARITHMETIC = "float32-fsum-binary64-v1"


def event_steps(raw, threshold):
    if isinstance(threshold, bool) or not isinstance(threshold, numbers.Real):
        raise ValueError("threshold must be finite binary64 within (0,1]")
    threshold = float(threshold)
    if not math.isfinite(threshold) or not 0 < threshold <= 1:
        raise ValueError("threshold must be finite binary64 within (0,1]")
    if not isinstance(raw, np.ndarray) or raw.dtype != np.float32 or raw.ndim != 1:
        raise ValueError("original one-dimensional float32 probabilities required; no conversion")
    if not np.isfinite(raw).all() or not np.all((raw >= 0) & (raw <= 1)):
        raise ValueError("probabilities must be finite within [0,1]")
    events, last, armed = [], -1000, True
    # Each float32 -> Python float conversion is exact. fsum avoids rolling
    # subtraction drift; no probability is rounded to an integer byte/grid.
    values = raw.tolist()
    for offset in range(4, len(values)):
        step = offset + 1
        mean = math.fsum(values[offset - 4:offset + 1]) / 5.0
        if mean < threshold * .5 and step - last >= 67:
            armed = True
        if armed and mean >= threshold and step - last >= 67:
            events.append(step)
            last = step
            armed = False
    return events


def event_samples(raw, threshold, samples):
    if type(samples) is not int or samples <= 0:
        raise ValueError("positive original sample count required")
    return [(step, min((3 * step + 2) * 160, samples))
            for step in event_steps(raw, threshold)]
