"""Run the pinned GPU input producer/auditor on explicit constructed features.

No admitted training data, model, microphone, speaker or remote process is used.
The isolated git receipt authenticates only this labelled synthetic fixture.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np

BASE = 'research/wakeword_p4/'
RESULT = BASE + 'results/4090-engine-adversarial-input-result-v1.json'
AUDIT = BASE + 'results/4090-engine-adversarial-input-independent-audit-v1.json'
PRODUCER = BASE + 'experiments/prepare_engine_adversarial_inputs_v1.py'
CHECKER = BASE + 'experiments/audit_engine_adversarial_inputs_v1.py'


def digest(data):
    return hashlib.sha256(data).hexdigest()


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + '\n')


def run(root, args, label, expect=0):
    result = subprocess.run(args, cwd=root, capture_output=True, text=True, timeout=60)
    (root.parent / (label + '.log')).write_text(result.stdout + result.stderr)
    if expect is not None:
        assert result.returncode == expect, (label, result.returncode, result.stderr)
    return result.returncode


def intake(ref):
    """Public receipt/coordinate metadata only; never read private NPZ arrays."""
    ref = subprocess.check_output(['git', 'rev-parse', ref], text=True).strip()
    hashes = {}

    def read(path):
        data = subprocess.check_output(['git', 'show', ref + ':' + path])
        hashes[path] = digest(data)
        return json.loads(data)

    lock_path = BASE + 'results/4090-engine-adversarial-input-execution-v1.json'
    lock, result, audit = read(lock_path), read(RESULT), read(AUDIT)
    assert result['all_passed'] and result['model_invocations'] == audit['model_invocations'] == 0
    assert audit['all_passed'] and audit['all_rng_coordinates_tensors_exact']
    assert result['execution_sha256'] == audit['execution_sha256'] == hashes[lock_path]
    assert audit['original_result_sha256'] == hashes[RESULT]
    assert audit['source_sha256'] == digest(subprocess.check_output(['git', 'show', ref + ':' + CHECKER]))
    assert lock['batches'] == 50 and lock['rows_per_batch'] == 33
    assert [b['step'] for b in result['batches']] == list(range(1, 51))
    engines, texts = result['engines'], result['texts']
    assert engines == ['kokoro', 'piper', 'qwen'] and len(texts) == len(set(texts)) == 11
    assert 'Hey Nicks.' in texts and 'Hey Willow.' in texts
    cells = {(c['text'], c['engine'], c['variant']): c for c in result['cell_counts']}
    assert len(cells) == len(result['cell_counts']) == 99
    assert set(cells) == {(t, e, v) for t in texts for e in engines for v in range(3)}
    assert all(c['rows'] >= c['sources'] > 0 for c in cells.values())
    for step, batch in enumerate(result['batches'], 1):
        assert len(batch['entries']) == 33 and batch['variant'] == (step - 1) % 3
        assert {(e['text'], e['engine']) for e in batch['entries']} == {(t, e) for t in texts for e in engines}
        assert batch['arrays_path'] == f'.local/wakeword/engine-adversarial-input-v1/{step:03}.npz'
        for entry in batch['entries']:
            assert entry['variant'] == batch['variant'] and entry['row'] >= 0
            assert 225 <= entry['legal_min'] <= entry['end'] <= entry['legal_max']
    public_pins, private_pins = {}, []
    for path, expected in lock['files'].items():
        if BASE in path:
            relative = BASE + path.split(BASE, 1)[1]
            actual = digest(subprocess.check_output(['git', 'show', ref + ':' + relative]))
            assert actual == expected
            public_pins[relative] = actual
        else:
            private_pins.append(path)
    assert audit['actual_windows'] == 1650
    return dict(gpu_commit=ref, input_sha256=hashes, public_pins=public_pins,
        private_pins_not_replayed=private_pins, batches=50, rows=1650,
        actual_public_step_sequence_correct=True, all_declared_cells_balanced=True,
        actual_private_rng_raw_npz_replayed=False,
        limits='Public source/hash/receipt/declared-coordinate metadata check only. '
               'The private tensor/RNG audit remains attributed to GPU execution.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu-ref', required=True)
    parser.add_argument('--actual-ref', help='Optional commit containing real input receipts, checked as metadata only')
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    ref = subprocess.check_output(['git', 'rev-parse', args.gpu_ref], text=True).strip()
    output = args.output_dir.resolve()
    output.mkdir(exist_ok=False)
    root = output / 'constructed-fixture'
    root.mkdir()
    original = {}
    paths = (PRODUCER, CHECKER, BASE + 'experiments/engine_adversarial_hypothesis_v1.md',
             BASE + 'results/4090-engine-adversarial-upstream-evidence-v1.json')
    for name in paths:
        data = subprocess.check_output(['git', 'show', ref + ':' + name])
        original[name] = digest(data)
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)

    # All names, features and provenance below are constructed. The three frame
    # geometries exercise singleton legal ends, a nontrivial interval, and the
    # 225 lower bound clipped by the end of a source. Every variant is present.
    metadata_path = '.local/wakeword/hey-broad-training-features-v1/hey_nyx/features.json'
    binary_path = '.local/wakeword/hey-broad-training-features-v1/hey_nyx/features.bin'
    rows, arrays, offset = [], [], 0
    for text in range(11):
        for engine in ('kokoro', 'piper', 'qwen'):
            for variant in range(3):
                for geometry, (frames, end_sample) in enumerate(((245, 244 * 160), (280, 237 * 160), (230, 212 * 160))):
                    index = len(rows)
                    values = ((np.arange(frames * 40, dtype=np.uint32).reshape(frames, 40)
                               + index * 17) % 800).astype('<u2')
                    arrays.append(values)
                    rows.append(dict(split='training', text='fixture-text-' + str(text), engine=engine,
                        variant=variant, source_pair_key=dict(carrier='bare', scale=1.0),
                        source_id='constructed-' + str(index), offset=offset, frames=frames,
                        generated_container_end_sample=end_sample, fixture_geometry=geometry))
                    offset += frames
    write_json(root / metadata_path, rows)
    (root / binary_path).write_bytes(np.concatenate(arrays).tobytes())
    feature_path = BASE + 'results/4090-hey-broad-features-v1.json'
    write_json(root / feature_path, dict(shared_binary_sha256=digest((root / binary_path).read_bytes()),
        shared_binary_bytes=(root / binary_path).stat().st_size,
        fixture_only=True))
    write_json(root / (BASE + 'results/4090-hey-broad-feature-independent-audit-v1.json'),
        dict(binary_hash_verified=True, original_report_sha256=digest((root / feature_path).read_bytes()),
             fixture_only=True))
    write_json(root / (BASE + 'results/4090-hey-broad-surviving-window-census-execution-v2.json'),
        dict(files={metadata_path: digest((root / metadata_path).read_bytes())}, fixture_only=True))

    run(root, [sys.executable, PRODUCER, 'lock'], 'lock')
    # The real producer verifies its lock against this isolated git ref. This
    # fixture git repo has no remotes and no access to the actual GPU repository.
    run(root, ['git', 'init', '-q'], 'git-init')
    lock = BASE + 'results/4090-engine-adversarial-input-execution-v1.json'
    run(root, ['git', 'add', lock], 'git-add')
    run(root, ['git', '-c', 'user.name=Constructed fixture', '-c',
               'user.email=fixture@local.invalid', 'commit', '-qm', 'Synthetic input lock'], 'git-commit')
    run(root, ['git', 'update-ref', 'refs/remotes/origin/research-hye-nyx-voice-gpu', 'HEAD'], 'git-ref')
    run(root, [sys.executable, PRODUCER, 'run'], 'producer')
    run(root, [sys.executable, CHECKER], 'original-auditor')
    baseline = json.loads((root / RESULT).read_text())
    shutil.copyfile(root / AUDIT, output / 'constructed-original-audit.json')

    # Separate NumPy reconstruction from the actual raw matrix, using explicit
    # row-local coordinate lists rather than the producer's global raw slices.
    all_x = np.concatenate(arrays)
    observations = dict(windows=0, singleton=0, lower_bound_225=0, clipped_value=0)
    for step, batch in enumerate(baseline['batches'], 1):
        assert batch['step'] == step and batch['variant'] == (step - 1) % 3
        with np.load(root / batch['arrays_path'], allow_pickle=False) as archive:
            assert archive['x'].shape == (33, 225, 40)
            pairs = set()
            for j, entry in enumerate(batch['entries']):
                row = rows[entry['row']]
                pairs.add((entry['text'], entry['engine']))
                start, stop = entry['end'] - 224, entry['end']
                raw = arrays[entry['row']][list(range(start, stop))].astype(np.float32)
                expected = np.vstack((np.fmin(raw / np.float32(25.6), np.float32(26)),
                                      np.zeros((1, 40), np.float32)))
                assert np.array_equal(archive['x'][j], expected)
                assert row['source_id'] == entry['source_id'] and row['variant'] == batch['variant']
                assert np.array_equal(raw, all_x[row['offset'] + start:row['offset'] + stop])
                observations['windows'] += 1
                observations['singleton'] += entry['legal_min'] == entry['legal_max']
                observations['lower_bound_225'] += entry['legal_min'] == 225
                observations['clipped_value'] += int(np.any(expected == 26))
            assert len(pairs) == 33
    assert observations['windows'] == 1650 and all(observations.values())

    # Preserve the untouched successful result. Only this constructed copy is
    # mutated. A validator acceptance is not a claim that actual GPU data is bad.
    original_result = (root / RESULT).read_bytes()
    controls = []
    for label, index, replacement in (('duplicate-step-same-variant', 3, 1),
                                      ('out-of-range-step-same-variant', 0, 52)):
        changed = json.loads(original_result)
        changed['batches'][index]['step'] = replacement
        write_json(root / RESULT, changed)
        (root / AUDIT).unlink(missing_ok=True)
        code = run(root, [sys.executable, CHECKER], label, expect=None)
        controls.append(dict(control=label, actual_exit_code=code,
                             chronology_corruption_rejected=code != 0))
    changed = json.loads(original_result)
    changed['batches'][0]['entries'][0]['source_id'] = 'fixture-wrong-source'
    write_json(root / RESULT, changed)
    (root / AUDIT).unlink(missing_ok=True)
    code = run(root, [sys.executable, CHECKER], 'wrong-source-identity', expect=None)
    assert code != 0
    controls.append(dict(control='wrong-source-identity', actual_exit_code=code,
                         source_corruption_rejected=True))
    (root / RESULT).write_bytes(original_result)
    (root / AUDIT).unlink(missing_ok=True)

    report = dict(schema='mac-engine-input-builder-constructed-review-v1', gpu_commit=ref,
        original_source_sha256=original, reviewer_sha256=digest(Path(__file__).read_bytes()),
        numpy=np.__version__, actual_producer_and_auditor_executed=True,
        fixture_rows=len(rows), fixture_windows_checked=observations, controls=controls,
        actual_training_sources_used=False, model_calls=0, updates=0, audio_accessed=False,
        candidate_qualified=False,
        limits=['Constructed input workflow and unchanged producer/auditor coverage only.',
                'Local NumPy runtime is declared; not a replay of private GPU tensors or its runtime.',
                'Chronology controls test validator coverage, not actual GPU output integrity.',
                'No gradient, domain-invariance, recognition, P4 or fresh-data qualification.'])
    if args.actual_ref:
        report['actual_public_receipt_intake'] = intake(args.actual_ref)
    write_json(output / 'report.json', report)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
