"""CTC timing conventions and training/held-out source boundary checks."""
import json
from pathlib import Path
import tempfile
import unittest
import wave

from audit_ctc_endpoints import feature_geometry, load_sources, sha, standalone, token_spans


class FrameTests(unittest.TestCase):
    def test_wav2vec_geometry(self):
        self.assertEqual(feature_geometry([10, 3, 3, 3, 3, 2, 2], [5, 2, 2, 2, 2, 2, 2]), (320, 400))
        for kernels, strides in (([], []), ([3], []), ([3], [0])):
            with self.assertRaises(ValueError):
                feature_geometry(kernels, strides)

    def test_ctc_repeats_separated_by_blank_and_frame_support(self):
        spans = token_spans([0, 4, 4, 0, 4, 0, 8, 8, 0], [1.] * 9,
                            {0: '<pad>', 4: 'k', 8: 's'}, 0, 320, 400, 3200)
        self.assertEqual([s['token'] for s in spans], ['k', 'k', 's'])
        self.assertEqual([s['feature_input_support_samples'] for s in spans],
                         [[320, 1040], [1280, 1680], [1920, 2640]])

    def test_silence_has_no_invented_endpoint(self):
        self.assertEqual(token_spans([0] * 5, [1.] * 5, {0: '<pad>'}, 0, 320, 400, 2000), [])

    def test_only_complete_standalone_aliases(self):
        for text in ('Hey Nicks.', 'Hey, Nyx!', 'HEY NIX?'):
            self.assertTrue(standalone(text))
        for text in ('Hey Nick.', 'Hey Next.', 'Hey Nyx, check the volume.'):
            self.assertFalse(standalone(text))


class SourceBoundaryTests(unittest.TestCase):
    def fixture(self, root, heldout=False):
        features = root / 'features'
        sources = root / 'dataset'
        directory = root / 'head-features-tts-v1'
        for path in (features, sources, directory):
            path.mkdir()
        audio = sources / 'wake.wav'
        with wave.open(str(audio), 'wb') as wav:
            wav.setparams((1, 2, 16000, 1600, 'NONE', 'not compressed'))
            wav.writeframes(b'\0\0' * 1600)
        metadata = [{'variant': 0, 'path': 'wake.wav', 'split': 'training',
                     'label': 1, 'sha256': sha(audio), 'engine': 'fixture',
                     'keyword_end': 1000, 'text': 'Hey Nyx.'}]
        # The loader resolves metadata-parent/dataset/path, as the real corpus
        # uses symlink-backed dataset paths. Here an explicit symlink is enough.
        (features / 'dataset').symlink_to(sources, target_is_directory=True)
        metadata_path = features / 'features.json'
        metadata_path.write_text(json.dumps(metadata))
        job = {'id': 'source-job', 'source': str(audio), 'source_sha256': sha(audio),
               'split': 'training', 'label': 1, 'kind': 'tts', 'variant': 0,
               'engine': 'fixture', 'keyword_end': 1000}
        jobs_path = directory / 'jobs-private.json'
        jobs_path.write_text(json.dumps([job]))
        index = directory / 'index.jsonl'
        index.write_text(json.dumps({'split': 'validation' if heldout else 'training',
                                     'source_sha256': sha(audio)}) + '\n')
        complete = directory / 'complete.json'
        complete.write_text('{}')
        source = {'source_sha256': sha(audio), 'variant0_id': job['id'], 'engine': 'fixture', 'label': 1}
        parent = {'source_text_metadata': 'features/features.json',
                  'source_text_metadata_sha256': sha(metadata_path),
                  'source_jobs_sha256': sha(jobs_path), 'sources': [source],
                  'feature_inputs': {'head-features-tts-v1': {
                      'index_sha256': sha(index), 'complete_sha256': sha(complete)}}}
        parent_path = root / 'parent.json'
        parent_path.write_text(json.dumps(parent))
        identity = {k: source[k] for k in ('source_sha256', 'variant0_id', 'engine')}
        identity.update(texts=['Hey Nyx.'], keyword_end=1000, samples=1600)
        plan = {'parent_plan_sha256': sha(parent_path), 'sources': [identity], 'omitted_command_sources': []}
        return parent_path, plan, audio

    def test_valid_source_keeps_original_audio_identity(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            parent, plan, audio = self.fixture(root)
            selected = load_sources(root, parent, plan)
            self.assertEqual(selected, [(plan['sources'][0], audio.resolve())])
            self.assertEqual(sha(audio), plan['sources'][0]['source_sha256'])

    def test_same_audio_under_a_heldout_role_is_rejected(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            parent, plan, _ = self.fixture(root, heldout=True)
            with self.assertRaisesRegex(ValueError, 'identity or split'):
                load_sources(root, parent, plan)

    def test_audio_replacement_is_rejected_before_model_loading(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            parent, plan, audio = self.fixture(root)
            audio.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'waveform differs'):
                load_sources(root, parent, plan)


if __name__ == '__main__':
    unittest.main()
