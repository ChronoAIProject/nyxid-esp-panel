"""Digital support/geometry is exact, without guessing phonetic boundaries."""
import hashlib
from pathlib import Path
import tempfile
import unittest
import wave

import numpy as np

from inventory_complete_signal import fits_support, read_support


class CompleteSignalTest(unittest.TestCase):
    def test_boundary_and_full_rir_tail(self):
        self.assertTrue(fits_support(0, 24160, 0, 0))
        self.assertFalse(fits_support(0, 24161, 0, 0))
        self.assertTrue(fits_support(0, 22240, 0, 12))
        self.assertFalse(fits_support(0, 22241, 0, 12))
        self.assertTrue(fits_support(0, 21280, 960, 12))
        self.assertFalse(fits_support(0, 21281, 960, 12))
        self.assertFalse(fits_support(None, None, 0, 0))

    def test_closed_form_matches_all_five_window_conditions(self):
        rng = np.random.default_rng(97733)
        for _ in range(100):
            first = int(rng.integers(0, 2000))
            end = first+int(rng.integers(16000, 30000))
            tail = int(rng.choice([0, 960]))
            brute = any(all((e+d-150)*160 <= 32000+first and
                            (e+d+1)*160 >= 32000+end+tail for d in range(0, 13, 3))
                        for e in range(150, 500))
            self.assertEqual(fits_support(first, end, tail, 12), brute)

    def test_wav_bytes_and_zero_support_without_trimming(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'test.wav'
            pcm = np.zeros(30000, dtype='<i2');pcm[4000], pcm[17000] = -1, 1
            with wave.open(str(path), 'wb') as stream:
                stream.setparams((1, 2, 16000, len(pcm), 'NONE', 'not compressed'))
                stream.writeframes(pcm.tobytes())
            original = path.read_bytes()
            record = {'samples': len(pcm), 'sha256': hashlib.sha256(original).hexdigest()}
            self.assertEqual(read_support(path, record)[:2], (4000, 17001))
            self.assertEqual(path.read_bytes(), original)
            self.assertFalse(fits_support(0, len(pcm), 960, 12))
            self.assertTrue(fits_support(4000, 17001, 960, 12))
            with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                read_support(path, {**record, 'sha256': '0'*64})
            with self.assertRaisesRegex(ValueError, 'header mismatch'):
                read_support(path, {**record, 'samples': 1})


if __name__ == '__main__':
    unittest.main()
