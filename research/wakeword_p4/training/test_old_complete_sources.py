import hashlib
import unittest

import numpy as np

from prepare_old_complete_sources import group_sources, render


class OldCompleteSourcesTest(unittest.TestCase):
    def test_alias_grouping_and_conflicting_labels(self):
        base = {'sha256': 'abc', 'engine': 'macos-say', 'label': 1, 'samples': 5,
                'support_geometry': {'context150_offset12_rir960': {'full_container': True}}}
        rows = [{**base, 'path': 'b'}, {**base, 'path': 'a'},
                {**base, 'path': 'excluded', 'support_geometry': {'context150_offset12_rir960': {'full_container': False}}}]
        groups = group_sources(rows)
        self.assertEqual([r['path'] for r in groups[0]['aliases']], ['a', 'b'])
        self.assertEqual(len(groups), 1)
        with self.assertRaisesRegex(ValueError, 'Conflicting'):
            group_sources(rows+[{**base, 'path': 'conflict', 'label': 0}])

    def test_dry_stream_preserves_sample_order_and_native_zeros(self):
        original = np.array([0, .1, -.3, 0, .4, 0], dtype=np.float64)
        before = original.copy()
        stream, info = render(original, 'original.wav', 0)
        self.assertEqual(len(stream), 32000+len(original)+16000)
        np.testing.assert_array_equal(original, before)
        self.assertTrue(np.all(stream[:32000] == 0) and np.all(stream[32006:] == 0))
        expected = original*.08/np.sqrt(np.mean(original**2))
        np.testing.assert_allclose(stream[32000:32006], expected, rtol=1e-15, atol=0)
        self.assertEqual(info['rir_tail_samples'], 0)
        self.assertIsNone(info['snr_db'])

    def test_full_rir_and_noise_match_independent_direct_convolution(self):
        # Different implementation: np.convolve rather than FFT, retained full tail.
        pcm = np.array([0, .07, -.11, .03, 0], dtype=np.float64)
        path = 'workspace/engine/voice/source.wav'
        rendered, info = render(pcm, path, 2)
        ir = np.zeros(961);ir[[0, 160, 480, 960]] = [1, .25, -.12, .06]
        speech = np.convolve(pcm, ir, 'full')
        signal = speech*.08/np.sqrt(np.mean(speech**2))
        expected = np.pad(signal, (32000, 16000))
        seed = int(hashlib.sha256((path+':2:phone-v1').encode()).hexdigest()[:16], 16)
        noise = np.random.default_rng(seed).normal(0, 1, len(expected))
        expected += noise*(.08*10**(-15/20)/np.sqrt(np.mean(noise**2)))
        expected *= min(1., .95/max(float(np.max(np.abs(expected))), 1e-8))
        self.assertEqual(info['signal_samples'], len(pcm)+960)
        self.assertEqual(info['rir_tail_samples'], 960)
        np.testing.assert_allclose(rendered, expected, rtol=2e-10, atol=1e-15)
        np.testing.assert_array_equal(rendered, render(pcm, path, 2)[0])
        self.assertLessEqual(float(np.max(np.abs(rendered))), .95)


if __name__ == '__main__':
    unittest.main()
