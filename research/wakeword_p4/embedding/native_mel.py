"""Blockwise FFT frontend for the inspected pretrained embedding mel graph.

Coefficients are supplied separately, with provenance/checksums in the run
manifest. No model weights, private PCM or generated arrays are stored here.
"""
import numpy as np


class NativeMel:
    def __init__(self, window, mel_filter):
        self.window = np.asarray(window, np.float32)
        self.mel_filter = np.asarray(mel_filter, np.float32)
        if self.window.shape != (512,) or self.mel_filter.shape != (257, 32):
            raise ValueError('Expected the source 512-sample window and 257x32 matrix')
        if not np.isfinite(self.window).all() or not np.isfinite(self.mel_filter).all():
            raise ValueError('Nonfinite frontend coefficients')

    def block(self, pcm):
        """Return eight new frames from 480 old + 1280 new 16 kHz samples.

        Flooring is per streaming block, rather than per recording. Framing
        follows the source graph; calls consume 1280 newly acquired samples.
        """
        pcm = np.asarray(pcm)
        if pcm.shape != (1760,):
            raise ValueError('Expected exactly 1760 mono PCM samples')
        frames = np.lib.stride_tricks.sliding_window_view(pcm, 512)[::160]
        spectrum = np.fft.rfft(frames*self.window, axis=-1)
        power = np.abs(spectrum)**2
        log_power = 10*np.log10(np.maximum(power@self.mel_filter, 1e-10))
        log_power = np.maximum(log_power, log_power.max()-80)
        return (log_power/10+2).astype(np.float32)
