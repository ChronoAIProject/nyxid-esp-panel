"""Prevent a portable synthetic baseline from exporting unrelated local state."""
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
import wave
import zlib

from export_assets import sha
from export_paired_baseline import export, verified_rows


class PairedBaselineExportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.data = self.root / 'data'
        self.data.mkdir()
        self.wav = self.data / 'synthetic.wav'
        pcm = b'\x00\x01' * 1024
        with wave.open(str(self.wav), 'wb') as wav:
            wav.setparams((1, 2, 16000, 0, 'NONE', 'not compressed'))
            wav.writeframes(pcm)
        row = {'path': str(self.wav), 'voice': 'synthetic-test', 'target': 'none',
               'text': 'Ordinary synthetic test.', 'variant': 0, 'samples': 1024,
               'keyword_end': None, 'sha256': sha(self.wav.read_bytes()),
               'scope': 'paired_mac_regression'}
        control = {**row, 'path': str(self.root / 'unopened-control.wav'),
                   'scope': 'excluded-control'}
        self.fixtures = [row, control]
        result = {'id': 0, 'complete': True, 'samples': 1024, 'crc32': zlib.crc32(pcm),
                  'events': 1, 'calls': 2, 'total_us': 100, 'max_us': 60, 'offsets': [768]}
        self.baseline = {'restored': True, 'sample_rate': 16000, 'engine': 'wn9_hiesp',
                         'mode': 'DET_MODE_95', 'probe_sha256': 'test-only',
                         'private_device_state': 'MUST_STAY_LOCAL',
                         'clips': [{**r, 'threshold': .63, 'result': dict(result),
                                    'private_uart': 'MUST_STAY_LOCAL'} for r in self.fixtures]}
        self.plan = {'scope_counts': {'paired_mac_regression': 1, 'excluded-control': 1},
                     'selected_scope': 'paired_mac_regression', 'target_counts': {'none': 1},
                     'voices': ['synthetic-test'], 'variants': [0], 'limits': ['Unit fixture only.']}
        self.manifest_path = self.data / 'manifest.jsonl'
        self.baseline_path = self.data / 'baseline.json'
        self.plan_path = self.data / 'plan.json'
        self.refresh_bindings()

    def refresh_bindings(self):
        self.manifest_path.write_text('\n'.join(map(json.dumps, self.fixtures)) + '\n')
        self.plan['manifest_sha256'] = sha(self.manifest_path.read_bytes())
        self.baseline['manifest_sha256'] = self.plan['manifest_sha256']
        self.baseline_path.write_text(json.dumps(self.baseline))
        self.plan['baseline_sha256'] = sha(self.baseline_path.read_bytes())
        self.plan_path.write_text(json.dumps(self.plan))

    def test_exact_pcm_numeric_results_relative_paths_and_excluded_state(self):
        entry = export(self.manifest_path, self.baseline_path, self.plan_path,
                       self.root / 'export', 'test-paired')
        with tarfile.open(self.root / 'export' / entry['name']) as archive:
            self.assertEqual(sum(m.name.endswith('.wav') for m in archive.getmembers()), 1)
            baseline_name = next(n for n in archive.getnames() if n.endswith('/baseline.json'))
            raw = archive.extractfile(baseline_name).read()
            self.assertNotIn(b'MUST_STAY_LOCAL', raw)
            self.assertNotIn(str(self.root).encode(), raw)
            row = json.loads(raw)['clips'][0]
            self.assertEqual(row['result']['offsets'], [768])
            self.assertEqual(row['result']['crc32'], self.baseline['clips'][0]['result']['crc32'])
            wav_name = next(n for n in archive.getnames() if n.endswith('.wav'))
            self.assertEqual(archive.extractfile(wav_name).read(), self.wav.read_bytes())
        self.assertFalse((self.root / 'unopened-control.wav').exists())

    def test_corrupt_transfer_crc_or_wav_never_exports(self):
        self.baseline['clips'][0]['result']['crc32'] ^= 1
        self.refresh_bindings()
        with self.assertRaisesRegex(ValueError, 'CRC'):
            verified_rows(self.manifest_path, self.baseline_path, self.plan)
        self.baseline['clips'][0]['result']['crc32'] ^= 1
        self.refresh_bindings()
        self.wav.write_bytes(self.wav.read_bytes() + b'corrupt')
        with self.assertRaisesRegex(ValueError, 'checksum'):
            verified_rows(self.manifest_path, self.baseline_path, self.plan)

    def test_path_escape_and_unbound_inputs_are_rejected(self):
        self.baseline_path.write_text(self.baseline_path.read_text() + '\n')
        with self.assertRaisesRegex(ValueError, 'export plan'):
            verified_rows(self.manifest_path, self.baseline_path, self.plan)
        self.fixtures[0]['path'] = '../outside.wav'
        self.baseline['clips'][0]['path'] = '../outside.wav'
        self.refresh_bindings()
        with self.assertRaises(ValueError):
            verified_rows(self.manifest_path, self.baseline_path, self.plan)


if __name__ == '__main__':
    unittest.main()
