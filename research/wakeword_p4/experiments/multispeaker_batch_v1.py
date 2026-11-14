"""Paired canonical cell sampler, independent of TensorFlow or model scores."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'training'))
from ambient_sampling import stratified_probabilities

BASE = Path('.local/wakeword/multispeaker-coverage-features-v1')
CONTEXT = 309


def cell(row):
    return (row['engine'], row['label'], row['text'], row['source_pair_key']['scale'], row['variant'])


def digest(array):
    return hashlib.sha256(np.asarray(array).tobytes()).hexdigest()


class Sampler:
    def __init__(self, arm, seed=726):
        assert arm in ['narrow', 'broad']
        self.arm = arm
        self.root = BASE / arm
        self.rng = np.random.default_rng(seed)
        self.raw = np.memmap(self.root / 'features.bin', mode='r', dtype='<u2').reshape(-1, 40)
        self.rows = json.loads((self.root / 'features.json').read_text())
        assert len(self.rows) == 12960 and all(r['split'] == 'training' and arm in r['arms'] for r in self.rows)
        self.rows.sort(key=lambda r: (cell(r), r['source_id']))
        self.pools = {}
        for label in [0, 1]:
            engines = sorted({r['engine'] for r in self.rows if r['label'] == label})
            assert len(engines) == 2
            self.pools[label] = [(engine, [r for r in self.rows if r['label'] == label and r['engine'] == engine]) for engine in engines]
        self.by_index = {r['shared_index']: r for r in self.rows}
        self.ambient = []
        for name in ['human.json', 'ambient.json']:
            rows = json.loads((self.root / name).read_text())
            assert all(r['split'] == 'training' for r in rows)
            self.ambient.extend(rows)
        self.ambient_arrays = [np.load(self.root / r['path'], mmap_mode='r', allow_pickle=False) for r in self.ambient]
        assert all(a.ndim == 2 and a.shape[1] == 40 and len(a) >= CONTEXT for a in self.ambient_arrays)
        self.ambient_probabilities = stratified_probabilities(self.ambient)

    def batch(self):
        descriptors = []
        for label, count in [(1, 32), (0, 48)]:
            for _ in range(count):
                _, pool = self.pools[label][int(self.rng.integers(2))]
                row = pool[int(self.rng.integers(len(pool)))]
                if label:
                    end = row['keyword_frame'] + int(self.rng.integers(3, 28))
                    assert end - CONTEXT <= row['lead_frame'] and end <= row['frames']
                else:
                    lo = max(CONTEXT, row['lead_frame'] - 25)
                    # One common uniform draw per paired row, even when native
                    # lengths differ; avoid RNG desynchronization from bounds.
                    end = lo + int(self.rng.random() * (row['frames'] + 1 - lo))
                assert CONTEXT <= end <= row['frames']
                descriptors.append(dict(kind='synthetic', index=row['shared_index'], end=end,
                                        label=label, cell=list(cell(row))))
        for _ in range(48):
            index = int(self.rng.choice(len(self.ambient), p=self.ambient_probabilities))
            frames = len(self.ambient_arrays[index])
            end = CONTEXT + int(self.rng.random() * (frames + 1 - CONTEXT))
            descriptors.append(dict(kind='ambient', index=index, end=end, label=0))
        descriptors = [descriptors[int(i)] for i in self.rng.permutation(128)]
        return self.materialize(descriptors), descriptors

    def materialize(self, descriptors):
        windows = []
        labels = []
        for d in descriptors:
            end = d['end']
            assert end >= CONTEXT
            if d['kind'] == 'synthetic':
                row = self.by_index[d['index']]
                assert d['label'] == row['label'] and tuple(d['cell']) == cell(row)
                assert end <= row['frames']
                offset = row['offset']
                window = self.raw[offset + end - CONTEXT:offset + end]
            else:
                assert d['kind'] == 'ambient' and d['label'] == 0
                window = self.ambient_arrays[d['index']][end - CONTEXT:end]
            assert window.shape == (309, 40)
            windows.append(window)
            labels.append(d['label'])
        x = np.clip(np.asarray(windows, dtype=np.float32) / np.float32(25.6), 0, 26)
        y = np.asarray(labels, dtype=np.float32)[:, None]
        weights = np.where(y[:, 0] == 1, 1., 20.).astype(np.float32)
        assert x.shape == (128, 309, 40) and y.sum() == 32
        return x, y, weights
