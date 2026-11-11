"""Bind actual training/development background arrays without opening test data."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np

from audit_compact_study import require, sha


def inventory(features):
    rows = []
    indexes = {}
    identities = {key: defaultdict(set) for key in ('path', 'speaker', 'source_sha256', 'pcm_sha256')}
    for name in ('human.json', 'ambient.json'):
        indexes[name] = sha(features/name)
        for index, row in enumerate(json.loads((features/name).read_bytes())):
            for key, seen in identities.items():
                if key in row:
                    seen[str(row[key])].add(row['split'])
            if row['split'] not in ('training', 'validation'):
                continue
            path = (features/row['path']).resolve()
            array = np.load(path, mmap_mode='r', allow_pickle=False)
            require(array.ndim == 2 and array.shape[1] == 40 and len(array) >= 335
                    and array.dtype == np.uint16, 'Background feature geometry differs')
            rows.append({'index': index, 'index_file': name, 'path': row['path'], 'split': row['split'],
                'file_bytes': path.stat().st_size, 'file_sha256': sha(path),
                'data_sha256': hashlib.sha256(array.astype('<u2', copy=False).tobytes(order='C')).hexdigest(),
                'shape': list(array.shape), 'dtype': '<u2'})
    for key, seen in identities.items():
        require(all(len(splits) == 1 for splits in seen.values()), 'Cross-split metadata identity: '+key)
    for key in ('file_sha256', 'data_sha256'):
        seen = defaultdict(set)
        for row in rows:
            seen[row[key]].add(row['split'])
        require(all(len(splits) == 1 for splits in seen.values()), 'Cross-split background bytes: '+key)
    return {'source_sha256': sha(Path(__file__)), 'index_sha256': indexes,
        'counts': dict(Counter(row['split'] for row in rows)), 'rows': rows,
        'scope': 'Actual training and validation NPY files and canonical uint16 payloads. All split metadata checked for shared path/speaker/WAV identities. Test arrays unopened. Exact-array disjointness does not exclude partial overlaps, mislabeled source content or certify split design.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--features', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = inventory(args.features)
    with args.output.open('x') as out:
        out.write(json.dumps(result, indent=2)+'\n')
    print(json.dumps({'counts': result['counts'], 'inventory_sha256': sha(args.output)}))


if __name__ == '__main__':
    main()
