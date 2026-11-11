"""Reject cross-split payload aliases even when the NPY containers differ."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from background_inputs import inventory


class BackgroundInputsTest(unittest.TestCase):
    def test_hash_actual_payload_and_leave_testing_unopened(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            np.save(root/'train.npy', np.zeros((335, 40), np.uint16))
            np.save(root/'val.npy', np.ones((336, 40), np.uint16))
            rows = [{'split': 'training', 'path': 'train.npy', 'speaker': 1},
                    {'split': 'validation', 'path': 'val.npy', 'speaker': 2},
                    {'split': 'testing', 'path': 'not-present.npy', 'speaker': 3}]
            (root/'human.json').write_text(json.dumps(rows))
            (root/'ambient.json').write_text('[]')
            before = inventory(root)
            self.assertEqual(before['counts'], {'training': 1, 'validation': 1})
            np.save(root/'train.npy', np.full((335, 40), 5, np.uint16))
            after = inventory(root)
            self.assertNotEqual(before['rows'][0]['file_sha256'], after['rows'][0]['file_sha256'])
            self.assertNotEqual(before['rows'][0]['data_sha256'], after['rows'][0]['data_sha256'])
            # Fortran-order container bytes differ while canonical feature bytes agree.
            np.save(root/'val.npy', np.asfortranarray(np.full((335, 40), 5, np.uint16)))
            with self.assertRaisesRegex(ValueError, 'Cross-split background bytes: data_sha256'):
                inventory(root)

    def test_metadata_overlap_with_unopened_test_split_is_rejected(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            np.save(root/'train.npy', np.zeros((335, 40), np.uint16))
            (root/'human.json').write_text(json.dumps([
                {'split': 'training', 'path': 'train.npy', 'speaker': 1},
                {'split': 'testing', 'path': 'unopened.npy', 'speaker': 1}]))
            (root/'ambient.json').write_text('[]')
            with self.assertRaisesRegex(ValueError, 'Cross-split metadata identity: speaker'):
                inventory(root)


if __name__ == '__main__':
    unittest.main()
