"""Full feature/source/role/offset checks and preselected frontend witnesses."""
import hashlib
import json
from pathlib import Path

import numpy as np
from pymicro_features import MicroFrontend
import soundfile as sf

from build_hey_broad_features_v1 import sha

R = Path('research/wakeword_p4/results')
BASE = Path('.local/wakeword/multispeaker-coverage-features-v1')
AMBIENT = Path('.local/wakeword/features-gpu-ami-extra-v1')


def main():
    report_path = R / '4090-multispeaker-coverage-features-v1.json'
    report = json.loads(report_path.read_text())
    plan_path = R / '4090-multispeaker-coverage-feature-plan-v1.json'
    plan = json.loads(plan_path.read_text())
    assert sha(plan_path) == report['plan_sha256']
    assert report['rows'] == 23220 and plan['context_frames'] == 309
    metadata = BASE / 'shared-metadata.json'
    binary = BASE / 'shared-features.bin'
    assert sha(metadata) == report['shared_metadata_sha256']
    assert sha(binary) == report['shared_binary_sha256']
    assert binary.stat().st_size == report['shared_binary_bytes'] == report['frames'] * 80
    raw = np.memmap(binary, mode='r', dtype='<u2').reshape(-1, 40)
    rows = json.loads(metadata.read_text())
    render_path = Path('.local/wakeword/multispeaker-coverage-rendered-v1/records.jsonl')
    assert sha(render_path) == plan['files'][str(render_path)]
    renders = [json.loads(line) for line in render_path.read_text().splitlines()]
    assert len(rows) == len(renders) == 23220
    offset = 0
    witnesses = {}
    for index, (row, render) in enumerate(zip(rows, renders)):
        source = render['source']
        assert row['shared_index'] == render['order'] == index
        assert row['split'] == source['role'] == 'training'
        assert row['source_id'] == source['id']
        assert row['target_id'] == source['target']
        assert row['label'] == source['label'] == int(source['target'] == 'hey_nyx')
        assert row['engine'] == source['engine'] and row['arms'] == source['arms']
        assert row['source_pair_key'] == source['pair_key']
        assert row['conditioning_id'] == source['pair_key']['speaker_name']
        assert row['text'] == source['text'] and row['variant'] == render['variant']
        assert all(row[k] == render[k] for k in ['path', 'sha256', 'pcm_sha256', 'samples', 'generated_container_end_sample'])
        assert row['generated_reference_end_sample'] == render['generated_prefix_end_sample']
        assert row['offset'] == offset and row['frames'] == (row['samples'] + 159) // 160 - 2
        offset += row['frames']
        assert row['frames'] >= 309 and row['lead_frame'] == render['lead_samples'] // 160 - 2
        expected_end = render['generated_prefix_end_sample'] // 160 - 2 if source['label'] else None
        assert row['keyword_frame'] == expected_end
        if expected_end is not None:
            assert expected_end + 27 - 309 <= row['lead_frame'] and expected_end + 27 <= row['frames']
        key = (row['engine'], row['conditioning_id'], row['target_id'], row['variant'])
        witnesses.setdefault(key, [index, index])[1] = index
    assert offset == len(raw) == plan['estimated_frames']
    counts = {}
    for arm in ['narrow', 'broad']:
        root = BASE / arm
        provenance = json.loads((root / 'target-provenance.json').read_text())
        assert provenance == report['arms'][arm]
        assert provenance['schema'] == 'multispeaker-coverage-target-pack-v1'
        assert provenance['arm'] == arm and provenance['target_id'] == 'hey_nyx'
        assert provenance['context_frames'] == 309 and provenance['training_only']
        for name, digest in provenance['files'].items():
            assert sha(root / name) == digest
        assert (root / 'features.bin').stat().st_ino == binary.stat().st_ino
        actual = json.loads((root / 'features.json').read_text())
        assert actual == [r for r in rows if arm in r['arms']]
        assert len(actual) == 12960 and sum(r['label'] for r in actual) == 2592
        for name in ['human.json', 'ambient.json']:
            original = json.loads((AMBIENT / name).read_text())
            expected = [r for r in original if r['split'] == 'training']
            assert json.loads((root / name).read_text()) == expected
            for r in expected:
                dest, source = root / r['path'], AMBIENT / r['path']
                assert sha(dest) == plan['files'][str(source)]
                assert (dest.stat().st_dev, dest.stat().st_ino) == (source.stat().st_dev, source.stat().st_ino)
                values = np.load(dest, mmap_mode='r', allow_pickle=False)
                assert values.ndim == 2 and values.shape[1] == 40 and len(values) >= 309
        counts[arm] = dict(rows=len(actual), positive=2592, negative=10368)
    receipts = []
    selected = sorted({i for ends in witnesses.values() for i in ends})
    for i in selected:
        row = rows[i]
        path = Path(row['path'])
        assert sha(path) == row['sha256']
        pcm, rate = sf.read(path, dtype='int16')
        assert rate == 16000 and pcm.ndim == 1 and len(pcm) == row['samples']
        assert hashlib.sha256(pcm.astype('<i2').tobytes()).hexdigest() == row['pcm_sha256']
        frontend = MicroFrontend()
        expected = []
        padded = np.pad(pcm, (0, (-len(pcm)) % 160))
        for at in range(0, len(padded), 160):
            output = frontend.process_samples(padded[at:at + 160].astype('<i2').tobytes())
            assert output.samples_read == 160
            if output.features:
                expected.append(np.rint(np.asarray(output.features) * 25.6).astype('<u2'))
        expected = np.asarray(expected, dtype='<u2').reshape(-1, 40)
        assert np.array_equal(expected, raw[row['offset']:row['offset'] + row['frames']])
        receipts.append(dict(shared_index=i, frames=len(expected), raw_sha256=hashlib.sha256(expected.tobytes()).hexdigest()))
    result = dict(schema='multispeaker-features-independent-audit-v1',
                  auditor_sha256=sha(Path(__file__)), original_report_sha256=sha(report_path),
                  rows=len(rows), frames=len(raw), arm_counts=counts,
                  whole_binary_hash_and_all_offsets_verified=True,
                  all_source_role_label_geometry_checks_pass=True,
                  original_training_ambient_hardlinks_verified=True,
                  witness_rule='First and last per(engine,speaker,target,variant), chosen without scores',
                  frontend_witnesses=receipts, frontend_witness_count=len(receipts),
                  detector_invocations=0, training_launched=False,
                  limits='Whole binary and all metadata audited; frontend transforms independently traversed only for preselected witnesses using the same pinned C++ library. Not full independent frontend reimplementation or model/holdout qualification.')
    with (R / '4090-multispeaker-features-independent-audit-v1.json').open('x') as stream:
        stream.write(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(rows=len(rows), frontend_witnesses=len(receipts), all_checks_pass=True)), flush=True)


if __name__ == '__main__':
    main()
