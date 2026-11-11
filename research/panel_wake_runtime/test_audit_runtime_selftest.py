import unittest

from audit_runtime_selftest import audit


class VerdictTest(unittest.TestCase):
    def setUp(self):
        self.fixture = {'model_sha256': 'a'*64, 'fixture_sha256': 'b'*64,
                        'expected_outputs': 179,
                        'app_prime': {'expected_device_event_steps': [144]}}
        self.log = ('panel.wake.research: model='+'a'*64+' threshold=0.917969\n'
                    'panel.wake.research: activation average=0.99 step=144 mean_us=1\n'
                    'exact-PCM selftest outputs=179 mismatches=0 passed=true\n')

    def test_bound_verdict_and_pre_verdict_steps(self):
        result = audit(self.log+'panel.wake.research: activation average=1 step=190\n', self.fixture)
        self.assertEqual(result['activation_steps'], [144])
        self.assertEqual(result['outputs'], 179)

    def test_rejects_missing_duplicate_failed_or_partial_verdict(self):
        for log in ('', self.log+self.log,
                    self.log.replace('mismatches=0', 'mismatches=1'),
                    self.log.replace('passed=true', 'passed=false'),
                    self.log.replace('outputs=179', 'outputs=178')):
            with self.subTest(log=log), self.assertRaises(ValueError):
                audit(log, self.fixture)

    def test_rejects_wrong_model_or_activation_sequence(self):
        for log in (self.log.replace('a'*64, 'c'*64),
                    self.log.replace('step=144', 'step=143'),
                    self.log.replace('activation average=0.99 step=144', 'no activation')):
            with self.subTest(log=log), self.assertRaises(ValueError):
                audit(log, self.fixture)


if __name__ == '__main__':
    unittest.main()
