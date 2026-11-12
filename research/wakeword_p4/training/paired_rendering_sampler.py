"""Separate local-RNG pairing; never changes inherited main sampler draws."""
import hashlib
import json
from pathlib import Path
import numpy as np

class PairedRenderingSampler:
    def __init__(self, root, authenticated_raw):
        self.root = Path(root)
        self.report_sha256 = hashlib.sha256((self.root/'report.json').read_bytes()).hexdigest()
        assert self.report_sha256 == '7b798653f73e2dcd3b01fd5b7092c947a26b9c8ec2525ff295de2e6478042e44'
        rows = json.loads((self.root/'report.json').read_text())['records']
        self.lookup = {(int(r['voice']), r['length_multiplier'], r['target'], r['arm']): r for r in rows}
        assert len(self.lookup) == len(rows) == len(authenticated_raw) == 40320
        self.raw = authenticated_raw
        self.voices = sorted({int(r['voice']) for r in rows})
        assert self.voices == sorted(np.random.default_rng(726).permutation(904)[:672].tolist())
        self.phrases = ['hey_nyx', 'hey_nick', 'hey_next', 'hey_mix']
        self.arms = ['quiet', 'noise25', 'combined', 'noise15', 'rir']
        self.rng = np.random.default_rng(227726)
        self.trace = []
        self.draws = 0
        self.last_trace = None

    def batch(self):
        speed = float(self.rng.choice([.9, 1., 1.1]))
        voices = [int(v) for v in self.rng.choice(self.voices, size=2, replace=False)]
        rows = [self.lookup[v, speed, p, a] for v in voices for p in self.phrases for a in self.arms]
        low = max(r['feature_crop_end'] for r in rows)
        high = min(350, *(r['raw_shape'][0]-1 for r in rows))
        ends = np.arange(((low+2)//3)*3, high+1, 3, dtype=np.int64)
        assert len(ends)
        end = int(self.rng.choice(ends))
        windows = np.stack([self.raw[r['pack_index']][end-149:end+1] for r in rows])
        assert windows.shape == (40, 150, 40) and windows.dtype == np.uint16
        labels = np.array([r['target'] == 'hey_nyx' for r in rows], dtype=np.float32)[:, None]
        assert labels.sum() == 10
        trace = dict(block=self.draws, voices=voices, speed=speed, endpoint_inclusive=end,
            pack_indices=[r['pack_index'] for r in rows],
            window_sha256=[hashlib.sha256(x.tobytes()).hexdigest() for x in windows])
        self.last_trace = trace
        if self.draws < 50:
            self.trace.append(trace)
        self.draws += 1
        return np.clip(windows.astype(np.float32)/25.6, 0, 26), labels, np.where(labels[:, 0] > 0, 1., 20.).astype(np.float32)
