"""Retained supervision trace must match actual feature crops and counters."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'wakeword_p4/training'))
from background_ctc import mark_ambient
from positive_ctc import Exposure, mark
from positive_exposure import audit


class ExposureTest(unittest.TestCase):
    def test_reconstructs_and_rejects_rehashed_trace_tampering(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            data = np.arange(700*40, dtype=np.uint16).reshape(700, 40)
            data.tofile(root/'features.bin')
            row = {'split': 'training', 'label': 1, 'lead_frame': 400,
                   'keyword_frame': 451, 'engine': 'piper', 'text': 'Hey Nyx, check the volume.',
                   'sha256': 'b'*64, 'variant': 0, 'offset': 0, 'frames': 700}
            (root/'features.json').write_text(json.dumps([row]))
            selections = [(row, 454 if i % 2 else 453) for i in range(32)]
            labels = np.zeros((128, 2), np.float32)
            labels[:32, 0] = 1
            labels[:8, 1] = np.arange(1, 17, 2)
            labels[32:40, 1] = np.arange(2, 17, 2)
            marked = mark(mark_ambient(labels, 48), selections, 335)
            samples = np.zeros((128, 335, 40), np.uint16)
            for i, (_, end) in enumerate(selections):
                samples[i] = data[end-335:end]
            exposure = Exposure([row], root/'positive-exposure.jsonl')
            exposure.record(1, marked, selections, samples)
            meta = exposure.metadata()
            exposure.stream.close()
            training = {'phone_distillation': {'ordinary_positive': {
                'gain': 1, 'steps': 1, 'examples': 12, 'target': [1, 2], 'exposure': meta}}}
            (root/'training.json').write_text(json.dumps(training))
            self.assertEqual(audit(root, root, 1)['eligible_examples'], 12)
            trace = json.loads((root/'positive-exposure.jsonl').read_bytes())
            for field, value in ((0, -1), (1, 701), (2, 1), (3, '0'*64)):
                changed = json.loads(json.dumps(trace))
                changed['draws'][0][field] = value
                content = (json.dumps(changed)+'\n').encode()
                (root/'positive-exposure.jsonl').write_bytes(content)
                meta['trace_sha256'] = hashlib.sha256(content).hexdigest()
                (root/'training.json').write_text(json.dumps(training))
                with self.subTest(field=field), self.assertRaises(ValueError):
                    audit(root, root, 1)


if __name__ == '__main__':
    unittest.main()
