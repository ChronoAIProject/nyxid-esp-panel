"""Opt-in complete whole-input feature scoring; historical scorers unchanged.

Matches the broad225 INT8 probe: final-only zero feature padding to stride3,
one continuous detector traversal, original one-based output sample clocks.
This is feature batching, not waveform padding or phase-reference adjustment.
"""
import numpy as np

POLICY = 'whole-features-final-zero-pad-stride3-v1'


def predict_complete_features(detector, raw, samples):
    """Return raw UINT8 outputs, support clocks and explicit tail metadata.

    Caller resets the frontend/detector once at whole-input start. This helper
    neither resets a detector nor splits the real input into separate streams.
    `raw` must contain all frames from final-hop-padded 16kHz whole PCM.
    """
    raw = np.asarray(raw)
    if raw.ndim != 2 or raw.shape[1] != 40 or raw.dtype != np.uint16:
        raise ValueError('Expected original uint16 [frames,40] frontend output')
    if type(samples) is not int or samples <= 0:
        raise ValueError('Expected positive original PCM sample count')
    expected_frames = max(0, (samples+159)//160-2)
    if len(raw) != expected_frames:
        raise ValueError('Feature count does not match complete original PCM')
    padding = (-len(raw)) % 3
    padded = np.pad(raw, ((0, padding), (0, 0))) if padding else raw
    outputs = detector.predict(padded)
    calls = (len(raw)+2)//3
    if (not isinstance(outputs, np.ndarray) or outputs.dtype != np.uint8
            or outputs.shape != (calls,)):
        raise ValueError('Detector omitted, duplicated or changed raw output steps')
    clocks = np.minimum((3*np.arange(1, calls+1, dtype=np.int64)+2)*160, samples)
    metadata = dict(policy=POLICY, original_samples=samples,
        original_feature_frames=len(raw), final_feature_padding=padding,
        input_feature_frames=len(padded), raw_steps=calls,
        clock='one-based s: min((3*s+2)*160, original_samples); no phase shift')
    return outputs, clocks, metadata
