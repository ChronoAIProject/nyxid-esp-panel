"""Replace four broad positive/negative pairs, preserve parent random schedule."""
import json
from pathlib import Path
import numpy as np
from causal_source_balanced_batch_v1 import Sampler as Parent
from causal_sequence_supervision_v1 import masks

CACHE = Path('.local/wakeword/chatterbox-training-cache-v1')


class Sampler(Parent):
    def __init__(self, seed=726):
        super().__init__(seed)
        self.chatter_rows = json.loads((CACHE / 'records.json').read_text())
        assert len(self.chatter_rows) == 84
        self.chatter_features = np.memmap(CACHE / 'features.bin', dtype='<u2', mode='r').reshape(-1, 40)
        self.chatter_rng = np.random.default_rng(seed + 3)
        self.chatter_voices = sorted({r['voice'] for r in self.chatter_rows})
        assert len(self.chatter_voices) == 4
        self.chatter_pools = {(voice, label): [r for r in self.chatter_rows if r['voice'] == voice and r['label'] == label]
                              for voice in self.chatter_voices for label in [0, 1]}
        assert all(self.chatter_pools.values())

    def batch(self):
        (x, p, n), descriptors = super().batch()
        positives = [i for i, row in enumerate(descriptors) if row['kind'] == 'existing_broad_bare_positive']
        negatives = [i for i, row in enumerate(descriptors) if row['kind'] == 'source_balanced_bare_negative']
        assert len(positives) == len(negatives) == 8
        chosen = self.chatter_rng.choice(8, size=4, replace=False)
        for pair in chosen:
            voice = self.chatter_voices[int(self.chatter_rng.integers(4))]
            for index, label in [(positives[pair], 1), (negatives[pair], 0)]:
                pool = self.chatter_pools[voice, label]
                row = pool[int(self.chatter_rng.integers(len(pool)))]
                raw = self.chatter_features[row['offset']:row['offset'] + row['frames']]
                _, pw, nw = masks(row['frames'], label, row['lead'], row['endpoint'])
                x[index].fill(0); p[index].fill(0); n[index].fill(0)
                x[index, :row['frames']] = np.clip(raw.astype(np.float32) / np.float32(25.6), 0, 26)
                p[index, :len(pw)] = pw; n[index, :len(nw)] = nw
                descriptors[index] = dict(kind='chatterbox_training', index=row['source']['index'],
                    voice=voice, label=label, replaced=descriptors[index], endpoint=row['endpoint'])
        return (x, p, n), descriptors
