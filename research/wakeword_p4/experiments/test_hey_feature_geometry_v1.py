"""Generated token-boundary geometry only; no speech, model or scoring."""
import unittest
from build_hey_training_features_v1 import geometry


class GeometryChecks(unittest.TestCase):
    def fixture(self):
        row=dict(id='analytic',role='training',source_sha256='saved-wav',source_samples=48000,
                 lead_samples=28800,rir_tail_samples=960,keyword_end=28800+48000+960)
        timing=dict(id='analytic',role='training',source_waveform_sha256='saved-wav',
                    allocation_matches_saved_waveform_length=True,
                    generated_prefix_end_16k_sample=12800)
        return row,timing

    def test_command_end_is_not_prefix(self):
        row,timing=self.fixture();result=geometry(row,timing)
        self.assertEqual(result['generated_prefix_end_sample'],42560)
        self.assertLess(result['generated_prefix_end_sample'],row['keyword_end'])
        self.assertTrue(result['all_positive_jitters_have_prefix_support'])
        for jitter in range(3,28):
            self.assertLessEqual(result['keyword_frame']+jitter-150,result['lead_frame'])

    def test_long_prefix_rejected_by_support(self):
        row,timing=self.fixture();timing['generated_prefix_end_16k_sample']=26000
        self.assertFalse(geometry(row,timing)['all_positive_jitters_have_prefix_support'])

    def test_mismatched_source_or_role_rejected(self):
        row,timing=self.fixture();timing['source_waveform_sha256']='other'
        with self.assertRaises(AssertionError):geometry(row,timing)
        row,timing=self.fixture();timing['role']='acceptance'
        with self.assertRaises(AssertionError):geometry(row,timing)


if __name__=='__main__':unittest.main()
