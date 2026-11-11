"""Original-intent compact sequence targets for a paired training-only loss."""
import hashlib
import json
from pathlib import Path

import numpy as np

from phone_distillation import Sampler as PhoneSampler, STEPS
from phrase_ctc_readout import TOKENS, load_bank, sha, targets


class Sampler(PhoneSampler):
    def __init__(self, bank, features, plan_path):
        plan = json.loads(plan_path.read_bytes())
        for name, digest in plan['source_sha256'].items():
            if sha(Path(__file__).parent/name) != digest:
                raise ValueError('Compact phrase study source differs from plan')
        self.features, labels, _ = load_bank(bank, features, plan['compact_bank'])
        packed = np.zeros((len(labels), STEPS), np.int32)
        packed[:, :2] = targets(labels)
        self.targets = np.eye(len(TOKENS), dtype=np.float32)[packed]
        self.bank_sha256 = sha(bank/'bank.json')
        self.plan_sha256 = sha(plan_path)
        self.sequence_metadata = {
            'sequence_sha256': hashlib.sha256(packed.astype('<i4').tobytes()).hexdigest(),
            'encoding': 'Original synthesis-intent [hey, nyx|mix|next|nick]; blank0 padding, no teacher labels.',
            'tokens': TOKENS}
        self.rng, self.order = np.random.default_rng(431), iter(())
        self.draws, self.digest, self.count = 0, hashlib.sha256(), len(labels)//2

    def metadata(self):
        return {**super().metadata(), 'vocabulary': len(TOKENS),
                'compact_phrase_plan_sha256': self.plan_sha256}
