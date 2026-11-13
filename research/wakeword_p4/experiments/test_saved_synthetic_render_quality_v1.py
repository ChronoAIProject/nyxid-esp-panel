"""Adversarial integration checks on immutable consumed rendering fixtures."""
import json
from pathlib import Path
import unittest
import soundfile as sf
from synthetic_render_quality_v1 import audit_render


class RenderQualityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        base = Path('.local/wakeword')
        root = base / 'slr83-development-replay-v2'
        cls.row = next(r for r in json.loads((root / 'manifest.json').read_text())['records']
                       if r['variant'] == 2)
        originals = [json.loads(l) for l in (base / 'slr83-development-v1/manifest.jsonl').read_text().splitlines()]
        original = next(r for r in originals if r['sha256'] == cls.row['source_sha256'])
        cls.dry, _ = sf.read(base / 'slr83-development-v1' / original['path'], dtype='int16')
        cls.pcm, _ = sf.read(root / cls.row['path'], dtype='int16')

    def test_original_passes(self):
        self.assertTrue(audit_render(self.dry, self.pcm, self.row)['passed'])

    def test_cropped_output_with_updated_ledger_fails(self):
        row = dict(self.row, samples=len(self.pcm) - 960)
        self.assertFalse(audit_render(self.dry, self.pcm[:-960], row)['passed'])

    def test_erased_rir_tail_with_same_length_fails(self):
        pcm = self.pcm.copy()
        pcm[self.row['native_generation_end']:self.row['keyword_end']] = 0
        self.assertFalse(audit_render(self.dry, pcm, self.row)['checks']['full_source_noise_rir_values'])

    def test_shifted_boundary_fails(self):
        row = dict(self.row, keyword_end=self.row['keyword_end'] - 1)
        self.assertFalse(audit_render(self.dry, self.pcm, row)['checks']['full_rir_boundary'])

    def test_changed_seed_fails(self):
        row = dict(self.row, seed=self.row['seed'] + 1)
        self.assertFalse(audit_render(self.dry, self.pcm, row)['checks']['full_source_noise_rir_values'])

    def test_changed_gain_metadata_fails(self):
        row = dict(self.row, gain_after_mix=self.row['gain_after_mix'] * .9)
        self.assertFalse(audit_render(self.dry, self.pcm, row)['checks']['declared_gain'])


if __name__ == '__main__': unittest.main()
