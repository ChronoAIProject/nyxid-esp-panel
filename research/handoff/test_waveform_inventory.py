"""Prevent cross-frontend source rebasing from leaking source partitions."""
import unittest

from waveform_inventory import source_member, validate_partitions


class WaveformInventoryTests(unittest.TestCase):
    def row(self, split='training', sha='one', speaker='voice', engine='tts'):
        return {'split': split, 'sha256': sha, 'speakers': [speaker], 'engine': engine}

    def test_rebase_and_prefix_escape(self):
        self.assertEqual(source_member('/old/corpus/a.wav', '/old'), 'workspace/corpus/a.wav')
        for value in ('/other/a.wav', '/old/../private.wav'):
            with self.assertRaises(ValueError):
                source_member(value, '/old')

    def test_wav_identity_cannot_cross_split_under_another_voice(self):
        with self.assertRaises(ValueError):
            validate_partitions([self.row(), self.row('validation', speaker='another')])

    def test_voice_identity_cannot_cross_split_with_another_wav(self):
        with self.assertRaises(ValueError):
            validate_partitions([self.row(), self.row('validation', sha='two')])

    def test_engine_qualified_tokens_and_preserved_splits(self):
        validate_partitions([self.row(), self.row('validation', sha='two', engine='other')])
        with self.assertRaises(ValueError):
            validate_partitions([self.row('testing')])


if __name__ == '__main__':
    unittest.main()
