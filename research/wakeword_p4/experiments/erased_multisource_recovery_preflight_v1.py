"""Test recovery of erased training-only coverage draws; no model or trainer changes."""
import argparse
import ast
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-ref', required=True)
    parser.add_argument('--complete-pack', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    commit = subprocess.check_output(
        ['git', 'rev-parse', args.source_ref + '^{commit}'], text=True).strip()
    prefix = 'research/wakeword_p4/training/'
    sources = {name: subprocess.check_output(['git', 'show', commit + ':' + prefix + name])
               for name in ['train_stream.py', 'complete_coverage.py',
                            'us_coverage.py', 'factorized_coverage.py']}
    with tempfile.TemporaryDirectory(prefix='nyx-coverage-source-') as temporary:
        source_path = Path(temporary) / 'complete_coverage.py'
        source_path.write_bytes(sources['complete_coverage.py'])
        namespace = {'__file__': str(source_path)}
        exec(compile(sources['complete_coverage.py'], str(source_path), 'exec'), namespace)
        sampler = namespace['CompleteCoverage'](
            args.complete_pack, negative_timeline=True, positive_timeline=True,
            negative_container_mixture=True)

    # Run the actual ordered coverage assignments on unique stage tags. The
    # authentic CompleteCoverage sampler below is tested separately with real
    # arrays; tag inputs are not model inputs or an acoustic experiment.
    tree = ast.parse(sources['train_stream.py'])
    main_fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
    batch = next(n for n in main_fn.body if isinstance(n, ast.FunctionDef) and n.name == 'batch')
    ordered = []
    for n in batch.body:
        if isinstance(n, ast.If) and isinstance(n.test, ast.Compare):
            left = n.test.left
            if isinstance(left, ast.Name) and left.id in (
                    'full_source', 'us_coverage', 'balanced_coverage'):
                ordered.append(n)
    assert len(ordered) == 3

    class Full:
        def first80(self):
            return np.full((80, 1, 1), 3, np.int16)

    class Replacement:
        def __init__(self, source, tag):
            self.tag = tag
            method = next(n for n in ast.walk(ast.parse(source))
                          if isinstance(n, ast.FunctionDef) and n.name == 'replace')
            self.slots = []
            for n in ast.walk(method):
                if isinstance(n, ast.Assign):
                    for target in n.targets:
                        if (isinstance(target, ast.Subscript)
                                and isinstance(target.value, ast.Name)
                                and target.value.id == 'result'
                                and isinstance(target.slice, ast.Slice)):
                            low = ast.literal_eval(target.slice.lower) if target.slice.lower else 0
                            high = ast.literal_eval(target.slice.upper)
                            self.slots.extend(range(low, high))
            assert self.slots

        def replace(self, x):
            x = x.copy()
            x[self.slots] = self.tag
            return x

    env = dict(x=np.zeros((128, 1, 1), np.int16), full_source=Full(),
               complete_coverage=Replacement(sources['complete_coverage.py'], 4),
               us_coverage=Replacement(sources['us_coverage.py'], 5),
               balanced_coverage=Replacement(sources['factorized_coverage.py'], 6))
    exec(compile(ast.Module(body=ordered, type_ignores=[]), prefix + 'train_stream.py', 'exec'), env)
    tags = env['x'][:, 0, 0]
    assert Counter(tags.tolist()) == {3: 40, 5: 20, 6: 20, 0: 48}
    assert not np.any(tags == 4)

    # A prospective intervention recovers exact, already drawn rows at the last
    # coverage stage, before permutation. It replaces only remaining British
    # slots: 8 positives and 12 confusing negatives. No new RNG call is needed.
    destination = np.r_[16:24, 56:68]
    origin = np.r_[0:8, 32:44]
    assert np.all(tags[destination] == 3)
    labels = np.asarray([1] * 32 + [0] * 96, np.float32)
    np.testing.assert_array_equal(labels[destination], labels[origin])
    counts = Counter()
    input_hash = hashlib.sha256()
    recovered_hash = hashlib.sha256()
    for _ in range(50):
        ids = []
        original_draw = sampler.draw_indices

        def capture():
            draw = original_draw()
            ids.extend(draw)
            return draw

        sampler.draw_indices = capture
        try:
            drawn = sampler.replace(np.zeros((80, 150, 40), np.float32))
        finally:
            sampler.draw_indices = original_draw
        assert len(ids) == 40
        features = drawn[origin]
        assert features.shape == (20, 150, 40)
        assert np.isfinite(features).all() and features.min() >= 0 and features.max() <= 26
        np.testing.assert_array_equal(sampler.y[np.asarray(ids)[np.r_[0:8, 16:28]]], labels[destination])
        substrate = np.broadcast_to(tags[:, None, None], (128, 150, 40)).astype(np.float32).copy()
        recovered = substrate.copy()
        recovered[destination] = features
        untouched = np.setdiff1d(np.arange(128), destination)
        np.testing.assert_array_equal(recovered[untouched], substrate[untouched])
        np.testing.assert_array_equal(recovered[destination], features)
        assert np.all(np.any(recovered[destination] != substrate[destination], axis=(1, 2)))
        input_hash.update(features.tobytes())
        recovered_hash.update(recovered[destination].tobytes())
        for slot, i in zip(destination, np.asarray(ids)[np.r_[0:8, 16:28]], strict=True):
            row = sampler.sources[sampler.records[i]['source_index']]['aliases'][0]
            counts[(row['engine'], 'positive' if slot < 32 else row['phrase'])] += 1
    assert input_hash.digest() == recovered_hash.digest()
    engines = sorted({engine for engine, _ in counts})
    assert engines == ['kokoro', 'macos-say', 'openai-tts', 'piper', 'qwen']
    report = dict(
        scope='Original coverage assignment AST plus authentic training-only sampler. '
              'Tag substrate for other sources; not an integrated trainer, model inference, '
              'gradient, quality, or causal-generalization result.',
        source_commit=commit,
        experiment_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        source_sha256={prefix + name: hashlib.sha256(raw).hexdigest() for name, raw in sources.items()},
        complete_pack_sha256=sampler.report['input_sha256'],
        current_surviving_complete_coverage_slots=0,
        current_full_source_slots=40, current_us_slots=20, current_factorized_slots=20,
        current_surviving_positive_slots=dict(full_source=16, us=8, factorized=8,
                                              complete_coverage=0),
        current_surviving_confusing_negative_slots=dict(full_source=24, us=12,
                                                       factorized=12, complete_coverage=0),
        recovery_destination=destination.tolist(), recovery_origin=origin.tolist(),
        batches=50, recovered_positive_windows=400, recovered_negative_windows=600,
        source_counts={engine + '/' + task: count for (engine, task), count in sorted(counts.items())},
        recovered_exact_feature_sha256=input_hash.hexdigest(),
        other_108_slots_unchanged=True, labels_unchanged=True,
        additional_rng_draws=0, current_formal_or_pinned_files_changed=False,
        limitation='This proves a possible sampler insertion has an input effect. '
                   'It does not prove quality improvement or authorize changing an active run. '
                   'Any future experiment needs actual-trainer input validation and a matched '
                   'comparison after reviewing current fixed/per_draw results. No new/final data, '
                   'audio speed changes, playback, recording, hardware or promotion.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
