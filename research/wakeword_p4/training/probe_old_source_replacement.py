"""Exercise the real source pack without a model or real v37 baseline generator."""
import argparse
from collections import Counter
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

from old_source_replacement import ENGINES, NEGATIVES, SLOTS, OldSourceReplacement


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('pack', 'plan', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    args = p.parse_args()
    plan = json.loads(args.plan.read_bytes())
    sampler = OldSourceReplacement(args.pack, seed=plan['seed'])
    for key in ('pack_sha256', 'sources_sha256', 'records_sha256'):
        assert sampler.report[key] == plan[key]
    args.output.mkdir(parents=True, exist_ok=False)
    records = json.loads((args.pack/'records.json').read_bytes())
    inverse, probabilities = {}, {}
    for target, engines in sampler.groups.items():
        target_prior = Fraction(1) if target == 'hey_nyx' else Fraction(1, 3)
        for engine, voices in engines.items():
            total = Fraction()
            for voice, sources in voices.items():
                for source_index, indices in sources.items():
                    chance = target_prior/Fraction(len(ENGINES)*len(voices)*len(sources)*3)
                    for index in indices:
                        assert index not in inverse
                        inverse[index] = (target, engine, voice, source_index, records[index]['variant'])
                        total += chance
            assert total == target_prior/5
            probabilities[target+'/'+engine] = str(total)
    assert len(inverse) == plan['expected_variants'] == 6597
    assert len({v[3] for v in inverse.values()}) == plan['expected_sources'] == 2199
    rng = np.random.default_rng(plan['fixture_seed'])
    x = rng.uniform(0, 26, (128, 150, 40)).astype(np.float32)
    y = np.asarray([1]*32+[0]*96, dtype=np.float32)[:, None]
    weights = np.where(y[:, 0] > 0, 1., 20.).astype(np.float32)
    original = [v.copy() for v in (x, y, weights)]
    kept = sorted(set(range(128))-set(SLOTS))
    reference_rng = np.random.default_rng(plan['seed'])
    draws, counts, variant_counts = [], Counter(), Counter()
    for _ in range(plan['batches']):
        before = np.random.get_state()
        result = sampler.replace(x, y, weights)
        after = np.random.get_state()
        assert before[0] == after[0] and before[2:] == after[2:]
        np.testing.assert_array_equal(before[1], after[1])
        np.testing.assert_array_equal(result[kept], x[kept])
        for actual, saved in zip((x, y, weights), original, strict=True):
            np.testing.assert_array_equal(actual, saved)
        ids = sampler.last_indices
        np.testing.assert_array_equal(result[list(SLOTS)], np.clip(sampler.features[ids].astype(np.float32)/25.6, 0, 26))
        # Independently replay the declared hierarchy directly from provenance,
        # without using the cached sampling tables.
        reference = []
        for slot in SLOTS:
            target = 'hey_nyx' if slot < 16 else NEGATIVES[int(reference_rng.integers(3))]
            engine = ENGINES[int(reference_rng.integers(5))]
            voices = sampler.groups[target][engine]
            voice_names = sorted(voices)
            voice = voice_names[int(reference_rng.integers(len(voice_names)))]
            names = sorted(voices[voice])
            source_index = names[int(reference_rng.integers(len(names)))]
            variant = int(reference_rng.integers(3))
            reference.append(voices[voice][source_index][variant])
        np.testing.assert_array_equal(ids, reference)
        for index in ids:
            target, engine, voice, source_index, variant = inverse[int(index)]
            counts[target+'/'+engine] += 1
            variant_counts[str(variant)] += 1
        draws.append(ids)
    ids = np.asarray(draws, dtype=np.int32)
    assert all(counts[k] > 0 for k in probabilities)
    arrays = args.output/'draws-and-baseline.npz'
    np.savez_compressed(arrays, draws=ids, baseline_x=x, baseline_y=y, baseline_weights=weights,
                        replaced_slots=np.asarray(SLOTS, dtype=np.int32))
    result = {'scope': __doc__, 'source_sha256': sha(Path(__file__)),
        'sampler_source_sha256': sha(Path(__file__).with_name('old_source_replacement.py')),
        'plan_sha256': sha(args.plan), 'data_contract': sampler.report,
        'numpy': np.__version__, 'python': sys.version, 'batches': len(draws),
        'replacement_draws': int(ids.size), 'eligible_canonical_sources': plan['expected_sources'],
        'eligible_variant_rows': len(inverse), 'unique_sampled_variant_rows': len(np.unique(ids)),
        'engine_class_probability_given_label': probabilities,
        'observed_engine_class_counts': dict(counts), 'observed_variant_counts': dict(variant_counts),
        'all_88_unselected_input_slots_exact': True, 'all_original_x_y_weights_unchanged': True,
        'all_normalized_replacements_match_original_candidate_arrays': True,
        'all_draws_match_independent_hierarchy_replay': True, 'numpy_global_rng_unchanged': True,
        'arrays_sha256': sha(arrays), 'model_trained': False, 'model_scored': False,
        'limits': plan['limits']}
    (args.output/'plan.json').write_bytes(args.plan.read_bytes())
    (args.output/'report.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
