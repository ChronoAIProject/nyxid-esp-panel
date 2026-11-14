"""Reconstruct saved batch receipts directly from authenticated raw features."""
import hashlib
import json
from pathlib import Path

import numpy as np

R = Path('research/wakeword_p4/results')
BASE = Path('.local/wakeword/multispeaker-coverage-features-v1')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    source = R / '4090-multispeaker-batch-probe-v1.json'
    report = json.loads(source.read_text())
    records = Path(report['records_path'])
    assert sha(records) == report['records_sha256']
    receipts = [json.loads(line) for line in records.read_text().splitlines()]
    assert len(receipts) == 100
    shared = np.memmap(BASE / 'shared-features.bin', dtype='<u2', mode='r').reshape(-1, 40)
    rows = {r['shared_index']: r for r in json.loads((BASE / 'shared-metadata.json').read_text())}
    ambient = sum([json.loads((BASE / 'narrow' / n).read_text()) for n in ['human.json', 'ambient.json']], [])
    feature_plan = json.loads((R / '4090-multispeaker-coverage-feature-plan-v1.json').read_text())
    arrays = {}
    actual_archives = 0
    pair = {}
    for receipt in receipts:
        arm, step = receipt['arm'], receipt['step']
        assert arm in ['narrow', 'broad'] and 1 <= step <= 50
        assert (step, arm) not in pair
        pair[(step, arm)] = receipt
        windows, labels = [], []
        for d in receipt['descriptors']:
            end = d['end']
            if d['kind'] == 'synthetic':
                row = rows[d['index']]
                assert row['split'] == 'training' and arm in row['arms']
                assert row['label'] == d['label'] == int(row['target_id'] == 'hey_nyx')
                assert d['cell'] == [row['engine'], row['label'], row['text'], row['source_pair_key']['scale'], row['variant']]
                assert 309 <= end <= row['frames']
                if row['label']:
                    assert 3 <= end - row['keyword_frame'] <= 27 and end - 309 <= row['lead_frame']
                else:
                    assert end >= max(309, row['lead_frame'] - 25)
                start = row['offset'] + end - 309
                window = shared[start:start + 309]
            else:
                assert d['kind'] == 'ambient' and d['label'] == 0
                row = ambient[d['index']]
                assert row['split'] == 'training'
                if d['index'] not in arrays:
                    path = BASE / arm / row['path']
                    original = Path('.local/wakeword/features-gpu-ami-extra-v1') / row['path']
                    assert sha(path) == feature_plan['files'][str(original)]
                    arrays[d['index']] = np.load(path, mmap_mode='r', allow_pickle=False)
                values = arrays[d['index']]
                assert 309 <= end <= len(values)
                window = values[end - 309:end]
            assert window.shape == (309, 40)
            windows.append(window)
            labels.append(d['label'])
        x = np.asarray(windows, dtype=np.float32)
        x /= np.float32(25.6)
        np.clip(x, 0, 26, out=x)
        y = np.asarray(labels, dtype=np.float32).reshape(128, 1)
        w = np.asarray([1. if v else 20. for v in labels], dtype=np.float32)
        assert y.sum() == 32 and sum(d['kind'] == 'ambient' for d in receipt['descriptors']) == 48
        for name, array in [('x', x), ('y', y), ('weights', w)]:
            assert hashlib.sha256(array.tobytes()).hexdigest() == receipt[name + '_sha256']
        if 'actual_tensor_path' in receipt:
            assert sha(receipt['actual_tensor_path']) == receipt['actual_tensor_sha256']
            with np.load(receipt['actual_tensor_path'], allow_pickle=False) as saved:
                assert all(np.array_equal(saved[k], v) for k, v in [('x', x), ('y', y), ('weights', w)])
            actual_archives += 1
    for step in range(1, 51):
        for a, b in zip(pair[(step, 'narrow')]['descriptors'], pair[(step, 'broad')]['descriptors'], strict=True):
            assert a['kind'] == b['kind'] and a['label'] == b['label']
            if a['kind'] == 'ambient':
                assert a == b
            else:
                assert a['cell'] == b['cell']
    assert actual_archives == 4
    result = dict(schema='multispeaker-batch-independent-audit-v1', auditor_sha256=sha(__file__),
                  original_report_sha256=sha(source), records_sha256=sha(records),
                  batches=100, raw_windows_reconstructed=12800, actual_npz_archives_verified=4,
                  all_role_label_source_window_and_actual_tensor_hashes_match=True,
                  all_paired_cells_and_ambient_windows_match=True,
                  model_calls=0, optimizer_updates=0,
                  limits='Direct raw-window reconstruction, every emitted batch digest, four actual saved tensor archives. Producer RNG-state equality is separate evidence; no network/gradient/stream/recognition qualification.')
    with (R / '4090-multispeaker-batch-independent-audit-v1.json').open('x') as stream:
        stream.write(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
