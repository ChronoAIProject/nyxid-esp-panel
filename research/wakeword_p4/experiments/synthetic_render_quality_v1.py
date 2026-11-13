"""Audit complete quiet/noise/RIR presentations without invoking a detector."""
import numpy as np

RIR_DELAYS = [0, 160, 480, 960]
RIR_AMPLITUDES = [1., .25, -.12, .06]


def audit_render(source, serialized, row):
    if source.dtype != np.int16 or source.ndim != 1 or not len(source):
        raise ValueError('Require complete mono PCM16 source')
    if serialized.dtype != np.int16 or serialized.ndim != 1:
        raise ValueError('Require mono PCM16 presentation')
    variant = row['variant']
    if variant not in (0, 1, 2):
        raise ValueError('Unknown rendering condition')
    dry = source.astype(np.float64) / 32768
    # Direct convolution independently checks the old FFT implementation.
    ir = np.zeros(961)
    ir[RIR_DELAYS] = RIR_AMPLITUDES
    speech = np.convolve(dry, ir, mode='full') if variant == 2 else dry
    lead, tail = 28800, 16000
    expected = np.pad(speech, (lead, tail))
    snr = None if variant == 0 else 25 if variant == 1 else 15
    if variant:
        noise = np.random.default_rng(row['seed']).normal(0, 1, len(expected))
        noise *= np.sqrt(np.mean(dry ** 2)) * 10 ** (-snr / 20) / np.sqrt(np.mean(noise ** 2))
        expected += noise
    peak = float(np.max(np.abs(expected)))
    gain = min(1., .95 / max(peak, 1e-8)) if variant else 1.
    expected *= gain
    error = (float(np.max(np.abs(serialized.astype(np.float64) / 32768 - expected)))
             if len(serialized) == len(expected) else None)
    checks = dict(
        source_sample_ledger=len(source) == row['source_samples'],
        presentation_sample_ledger=len(serialized) == row['samples'],
        full_source_and_rir_tail_length=len(serialized) == len(expected),
        full_source_noise_rir_values=error is not None and error <= 1 / 32768 + 1e-7,
        full_native_boundary=row['native_generation_end'] == lead + len(source),
        full_rir_boundary=row['keyword_end'] == lead + len(speech),
        full_rir_tail=row['rir_tail_samples'] == (960 if variant == 2 else 0),
        padding=row['lead_samples'] == lead and row['tail_samples'] == tail,
        snr=row['snr_db'] == snr,
        rir_enabled=row['synthetic_rir'] == (variant == 2),
        declared_gain=abs(row['gain_after_mix'] - gain) <= 1e-12,
        finite_no_clipping=bool(np.isfinite(expected).all() and np.all(np.abs(expected) < 1)),
        no_saturated_pcm16=not bool(np.any((serialized == -32768) | (serialized == 32767))),
    )
    return dict(checks=checks, passed=all(checks.values()),
                expected_samples=len(expected), maximum_quantization_error=error,
                reconstructed_gain=gain)
