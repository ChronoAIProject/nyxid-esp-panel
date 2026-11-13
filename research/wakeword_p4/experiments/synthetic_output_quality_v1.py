"""Mechanical native-output/PCM audit; never judges phonetic truth or scores models."""
import hashlib
import math
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly
import soundfile as sf


POLICY = dict(duration_seconds=[.2, 5.], minimum_rms=.001,
              maximum_saturated_samples=0, transport_rate=16000,
              pcm_quantization_error_bound=1 / 32768 + 1e-7)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def audit_output(root, record):
    """Check all saved native samples survive transport without trimming/gain.

    Native completeness is relative to the saved engine return, not semantic
    completeness. A generator must save the entire return before processing.
    Failures are recorded, never repaired by gain, cropping or regeneration.
    """
    root = Path(root).resolve()
    paths = []
    for name, digest in [('raw_path', 'raw_sha256'), ('path', 'sha256')]:
        path = (root / record[name]).resolve()
        if not path.is_relative_to(root):
            raise ValueError('Output path escapes source directory')
        if sha(path) != record[digest]:
            raise ValueError('Changed saved output: ' + record[name])
        paths.append(path)
    native = np.load(paths[0], allow_pickle=False)
    info = sf.info(paths[1])
    pcm, rate = sf.read(paths[1], dtype='int16')
    if native.ndim != 1 or native.dtype != np.float32 or not len(native):
        raise ValueError('Native output must be a nonempty mono float32 vector')
    if pcm.ndim != 1:
        raise ValueError('Transport must be mono')
    native_rate = record['native_rate']
    if not isinstance(native_rate, int) or native_rate <= 0:
        raise ValueError('Invalid native sample rate')
    divisor = math.gcd(native_rate, 16000)
    finite = bool(np.isfinite(native).all())
    full = resample_poly(native, 16000 // divisor, native_rate // divisor) if finite else None
    expected_samples = (len(native) * 16000 + native_rate - 1) // native_rate
    same_length = len(pcm) == expected_samples
    error = (float(np.max(np.abs(pcm.astype(np.float64) / 32768 - full)))
             if finite and same_length else None)
    saturated = int(np.count_nonzero((pcm == -32768) | (pcm == 32767)))
    rms = float(np.sqrt(np.mean((pcm.astype(np.float64) / 32768) ** 2))) if len(pcm) else 0.
    checks = dict(
        mono_pcm16_16k=info.channels == 1 and info.subtype == 'PCM_16' and rate == 16000,
        native_finite=finite,
        native_sample_ledger=len(native) == record['native_samples'],
        native_no_clipping=finite and bool(np.all(np.abs(native) < 1)),
        resampled_no_clipping=finite and bool(np.all(np.abs(full) < 1)),
        full_resampled_length=same_length,
        full_resampled_values=error is not None and error <= POLICY['pcm_quantization_error_bound'],
        pcm_sample_ledger=len(pcm) == record['samples'],
        no_saturated_pcm16=saturated == 0,
        nonempty_nonzero=bool(len(pcm) and np.any(pcm)),
        duration_in_spec=.2 <= len(pcm) / 16000 <= 5.,
        rms_in_spec=rms >= .001,
        no_declared_trimming=record['trimmed'] is False,
        generation_reference_is_full_output=record['generated_reference_end_sample'] == expected_samples,
    )
    return dict(checks=checks, passed=all(checks.values()),
                native_samples=len(native), transport_samples=len(pcm),
                maximum_transport_error=error, rms=rms,
                saturated_samples=saturated)
