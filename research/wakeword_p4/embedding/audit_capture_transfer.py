"""Characterize one known playback/capture pair without fitting a wake model.

Only aggregate alignment, band power and coherence are exported. Neither the
microphone recording nor a fitted room/device response becomes training data.
Keep original PCM rate/speed; correlation alignment is diagnostic only.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--capture', type=Path, required=True)
    p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    plan = json.loads(args.plan.read_text())
    if args.output.exists():
        p.error('Fresh output required')
    if digest(args.source) != plan['source_sha256'] or digest(args.capture) != plan['capture_sha256']:
        raise ValueError('Explicit acoustic pair differs')
    if digest(Path(__file__)) != plan['script_sha256']:
        raise ValueError('Diagnostic implementation differs')
    source, rate = sf.read(args.source, dtype='float64')
    captured, captured_rate = sf.read(args.capture, dtype='float64')
    if (rate != 16000 or captured_rate != rate or source.ndim != 1 or captured.ndim != 1
            or not np.isfinite(source).all() or not np.isfinite(captured).all()):
        raise ValueError('Expected unchanged mono16kHz PCM')
    correlation = signal.correlate(captured, source, mode='full', method='fft')
    lags = signal.correlation_lags(len(captured), len(source), mode='full')
    # Restrict to complete playback overlap instead of selecting tiny edges.
    valid = (lags >= 0) & (lags <= len(captured) - len(source))
    if not valid.any():
        raise ValueError('Capture must contain the complete source playback')
    choices = np.flatnonzero(valid)
    lag = int(lags[choices[np.argmax(np.abs(correlation[valid]))]])
    observed = captured[lag:lag + len(source)]
    source_energy, observed_energy = float(source @ source), float(observed @ observed)
    if min(source_energy, observed_energy) <= 0:
        raise ValueError('Acoustic input is silent')
    nperseg = plan['welch_samples']
    overlap = plan['overlap_samples']
    hz, coherence = signal.coherence(source, observed, fs=rate, nperseg=nperseg, noverlap=overlap)
    _, source_power = signal.welch(source, fs=rate, nperseg=nperseg, noverlap=overlap)
    _, observed_power = signal.welch(observed, fs=rate, nperseg=nperseg, noverlap=overlap)
    bands = []
    for low, high in plan['bands_hz']:
        selected = (hz >= low) & (hz < high)
        a, b = float(source_power[selected].sum()), float(observed_power[selected].sum())
        if min(a, b) <= 0:
            raise ValueError('Insufficient source or capture energy in a diagnostic band')
        bands.append({'hz': [low, high], 'relative_gain_db': float(10 * np.log10(b / a)),
                      'coherence_mean': float(coherence[selected].mean()),
                      'source_power_fraction': float(a / source_power.sum()),
                      'capture_power_fraction': float(b / observed_power.sum())})
    report = {'scope': __doc__, 'plan_sha256': digest(args.plan),
              'source_sha256': plan['source_sha256'], 'capture_sha256': plan['capture_sha256'],
              'sample_rate': rate, 'source_samples': len(source), 'capture_samples': len(captured),
              'alignment_lag_samples': lag, 'full_playback_overlap': True,
              'waveform_correlation': float(source @ observed / np.sqrt(source_energy * observed_energy)),
              'scalar_least_squares_gain': float(source @ observed / source_energy),
              'rms_ratio': float(np.sqrt(observed_energy / source_energy)), 'bands': bands,
              'speed_and_sample_rate_unchanged': True, 'training_data_created': False,
              'accepted': False,
              'limits': ['One known synthetic playback at one room/placement; no population inference.',
                         'Band power ratios include room response and noise; they are not an isolated microphone response.',
                         'No equalizer/threshold is selected from these measurements; no fitted impulse response is exported.']}
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
