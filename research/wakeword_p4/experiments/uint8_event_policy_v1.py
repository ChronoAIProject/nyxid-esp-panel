"""Future opt-in event arithmetic; raw model bytes, never float-score rounding.

Returns one-based raw inference steps (first full window ends at step5).
This module does not modify or activate any historical evaluator.
"""
import math
import numbers

EVENT_ARITHMETIC = 'uint8-sum-binary64-v1'


def event_steps(raw, threshold):
    if isinstance(threshold, bool) or not isinstance(threshold, numbers.Real):
        raise ValueError('threshold must be finite binary64 within (0,1]')
    threshold = float(threshold)
    if not math.isfinite(threshold) or not 0 < threshold <= 1:
        raise ValueError('threshold must be finite binary64 within (0,1]')
    values = list(raw)
    if any(isinstance(x, bool) or not isinstance(x, numbers.Integral)
           or not 0 <= x <= 255 for x in values):
        raise ValueError('raw scores must be UINT8 integers; float rounding forbidden')
    events, window, total, last, armed = [], [], 0, -1000, True
    for step, value in enumerate(values, 1):
        window.append(int(value)); total += int(value)
        if len(window) > 5:
            total -= window.pop(0)
        if len(window) < 5:
            continue
        mean = total / 1280.0
        if mean < threshold * .5 and step - last >= 67:
            armed = True
        if armed and mean >= threshold and step - last >= 67:
            events.append(step); last = step; armed = False
    return events
