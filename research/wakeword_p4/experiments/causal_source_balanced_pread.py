"""Original source-balanced training schedule with positional feature reads on Linux."""
from contextlib import ExitStack
import gzip
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
from causal_source_balanced_batch_v1 import Sampler as OriginalSampler, digest


class FeatureRows:
    """Read a dense range of feature rows using the original mapping's file offset."""
    def __init__(self, mapping, stack):
        if not isinstance(mapping, np.memmap) or mapping.ndim != 2 or not mapping.flags.c_contiguous:
            raise ValueError('Expected a contiguous two-dimensional feature mapping')
        self.file = stack.enter_context(open(mapping.filename, 'rb', buffering=0))
        os.posix_fadvise(self.file.fileno(), 0, 0, os.POSIX_FADV_RANDOM)
        self.dtype = mapping.dtype
        self.rows, self.columns = mapping.shape
        self.row_bytes = self.columns*self.dtype.itemsize
        self.offset = mapping.offset
        if os.fstat(self.file.fileno()).st_size != self.offset + mapping.nbytes:
            raise ValueError('Expected whole-file matrix or a complete NPY payload')

    def __len__(self):
        return self.rows

    def __getitem__(self, key):
        start, stop, step = key.indices(self.rows)
        if step != 1:
            raise ValueError('Feature sampler requires contiguous slices')
        count = max(0, stop - start)
        size = count*self.row_bytes
        data = os.pread(self.file.fileno(), size, self.offset + start*self.row_bytes)
        if len(data) != size:
            raise OSError('Incomplete positional feature read')
        return np.frombuffer(data, dtype=self.dtype).reshape(count, self.columns)


class Sampler(OriginalSampler):
    """Use within a with block to release every source handle when a job ends."""
    def __init__(self, seed=726):
        super().__init__(seed)
        self._files = ExitStack()
        try:
            self.binaries = {key: FeatureRows(bank, self._files) for key, bank in self.binaries.items()}
            self.ambient_arrays = [FeatureRows(bank, self._files) for bank in self.ambient_arrays]
            self.nearword_values = FeatureRows(self.nearword_values, self._files)
            self.extra_values = FeatureRows(self.extra_values, self._files)
            self.extra_positive_values = FeatureRows(self.extra_positive_values, self._files)
        except BaseException:
            self._files.close()
            raise

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self._files.close()


def main():
    results = Path('research/wakeword_p4/results')
    report_path = results/'4090-broadcast-residual-training-original-v1.json'
    report = json.loads(report_path.read_bytes())
    ledger = Path('.local/wakeword/broadcast-residual-training-v1/batches.jsonl.gz')
    with ledger.open('rb') as stream:
        if hashlib.file_digest(stream, 'sha256').hexdigest() != report['files'][ledger.name]:
            raise ValueError('Original sampler ledger differs from the completed run')
    start = time.monotonic()
    with gzip.open(ledger, 'rt') as stream, Sampler(726) as sampler:
        for step in range(1, 65):
            expected = json.loads(stream.readline())
            (x, p, n), descriptors = sampler.batch()
            assert expected['step'] == step and expected['descriptors'] == descriptors
            assert x.shape == (128, 846, 40) and p.shape == n.shape == (128, 282)
            for data, key in [(x, 'features_sha256'), (p, 'positive_sha256'), (n, 'negative_sha256')]:
                assert digest(data) == expected[key], (step, key)
    print(json.dumps(dict(steps=64, rows=64*128, exact_descriptors_and_three_digests=True,
                          reference_ledger_sha256=report['files'][ledger.name], elapsed_seconds=time.monotonic()-start,
                          reader_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                          limits='First 64 sequential seed726 batches; no model, optimizer or GPU execution.')))


if __name__ == '__main__':
    main()
