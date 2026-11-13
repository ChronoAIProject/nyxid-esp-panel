"""Adversarial transport checks with analytic fixtures, not synthetic speech evidence."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly
import soundfile as sf

from synthetic_output_quality_v1 import audit_output, sha


class OutputQualityTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.native = (.2 * np.sin(2 * np.pi * 440 * np.arange(22050) / 22050)).astype(np.float32)
        self.pcm = resample_poly(self.native, 320, 441)
        self.row = dict(raw_path='raw.npy', path='pcm.wav', native_rate=22050,
                        native_samples=22050, samples=16000, trimmed=False,
                        generated_reference_end_sample=16000)

    def write(self, native=None, pcm=None):
        np.save(self.root / 'raw.npy', self.native if native is None else native)
        sf.write(self.root / 'pcm.wav', self.pcm if pcm is None else pcm, 16000, subtype='PCM_16')
        self.row.update(raw_sha256=sha(self.root / 'raw.npy'), sha256=sha(self.root / 'pcm.wav'))
        return audit_output(self.root, self.row)

    def test_complete_transport_passes(self):
        self.assertTrue(self.write()['passed'])

    def test_trim_with_updated_pcm_ledger_still_fails(self):
        self.row['samples'] = 15900
        result = self.write(pcm=self.pcm[:-100])
        self.assertFalse(result['checks']['full_resampled_length'])
        self.assertFalse(result['passed'])

    def test_gain_change_is_rejected(self):
        self.assertFalse(self.write(pcm=self.pcm * .5)['checks']['full_resampled_values'])

    def test_silence_and_clipping_fail(self):
        for value in (0., 1.2):
            with self.subTest(value=value):
                self.assertFalse(self.write(native=np.full(22050, value, np.float32),
                                            pcm=np.full(16000, value, np.float32))['passed'])

    def test_nonfinite_native_fails(self):
        bad = self.native.copy()
        bad[100] = np.nan
        self.assertFalse(self.write(native=bad)['passed'])

    def test_changed_native_hash_raises(self):
        self.write()
        np.save(self.root / 'raw.npy', self.native * .9)
        with self.assertRaisesRegex(ValueError, 'Changed saved output'):
            audit_output(self.root, self.row)


if __name__ == '__main__':
    unittest.main()
