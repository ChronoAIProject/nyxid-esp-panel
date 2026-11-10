"""Shared, calibration-bound score averaging for streaming wake measurements."""
import numpy as np

DEFAULT_SMOOTHING_STEPS = 5
SUPPORTED_SMOOTHING_STEPS = (3, 5, 7)


def calibrated_smoothing_steps(calibration):
    # Old calibrations used five steps without recording the field.
    steps = calibration.get("smoothing_steps", DEFAULT_SMOOTHING_STEPS)
    if type(steps) is not int or steps not in SUPPORTED_SMOOTHING_STEPS:
        raise ValueError("Unsupported calibration smoothing_steps")
    return steps


def smooth(scores, steps=DEFAULT_SMOOTHING_STEPS):
    calibrated_smoothing_steps({"smoothing_steps": steps})
    return (np.convolve(scores, np.ones(steps)/steps, mode="valid")
            if len(scores) >= steps else np.array([]))
