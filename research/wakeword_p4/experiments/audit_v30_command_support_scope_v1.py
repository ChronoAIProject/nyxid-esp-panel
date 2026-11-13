"""Distinguish the current positive-coverage gap from frozen v30's failures."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np

from prepare_inherited_positive_recovery_v1 import batch_environment


def sha(blob):
    return hashlib.sha256(blob).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('training', 'metadata', 'saved-study', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    training_blob = args.training.read_bytes()
    assert sha(training_blob) == '41d3a232920143d6bfa39b74fc1068f2675615f2df23ed599155b58a085312f9'
    training = json.loads(training_blob)
    assert training['model_sha256'] == '04aa45646610f0e3a866a17d4098263adf5da912bb6ddfe03c44807a295baa0c'
    assert training['selected_step'] == 70000 and training['source_exponent'] == 0
    assert training['prefix_negatives']['count_per_batch'] == 16
    assert training['features_index_sha256'] == sha(args.metadata.read_bytes()) == '9db652f0769e49be6e924aadbfc3cd3725870b2eb558ea5b25b25481970d1d4b'
    source_commit = '29be79c7e0af59dfc5cddcdc3733a284c4c777ff'
    source = subprocess.check_output(['git', 'show', source_commit + ':research/wakeword_p4/training/train_stream.py'])
    assert sha(source) == training['source_files_sha256']['train_stream.py'] == 'a9c76aee07f2a61557ed77095609671f546bfbbf418d979732da356b98898c20'
    tree = ast.parse(source)
    function = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == 'batch')
    batch, rng, _, drawn = batch_environment(tree, False, False)
    # Retain actual v30 prefix-negative setting. The symbolic fixture is not
    # an actual source/RNG replay; only mutation of its positive rows is tested.
    batch.__globals__['args'].prefix_negatives = 16
    batch.__globals__['prefix_sample'] = lambda count: np.zeros((count, 150, 40), np.float32)
    for _ in range(50):
        x, y, weights = batch()
        inverse = np.argsort(rng.order)
        np.testing.assert_array_equal(x[inverse][:32], drawn['positive'])
        np.testing.assert_array_equal(y[inverse][:32], np.ones((32, 1)))
        np.testing.assert_array_equal(weights[inverse][:32], np.ones(32))
    positives = [r for r in json.loads(args.metadata.read_text()) if r['split'] == 'training' and r['label'] == 1]
    commands = [r for r in positives if r['text'].startswith('Hey Nyx,')]
    assert len(commands) == 34150 and {r['engine'] for r in commands} == {'piper'}
    assert training['sources']['1'] == {'piper': 60000, 'kokoro': 3000, 'qwen-cuda': 300}
    baseline_blob = (args.saved_study / 'report.json').read_bytes()
    assert sha(baseline_blob) == 'de27257b414ffa6eb90ce8aad5649310662e1db18bb93febfcf1f94c097184ee'
    baseline = json.loads(baseline_blob)
    norman = [r for r in baseline['records'] if r['record']['voice'] == 'norman'
              and r['native_phrase_id'] == 'hey_nyx_command']
    assert len(norman) == 9 and not any(r['classification']['reference_window_hit'] for r in norman)
    report = dict(schema='v30-command-support-causal-scope-v1',
        source_sha256=sha(Path(__file__).read_bytes()),
        symbolic_helper_sha256=sha(Path(__file__).with_name('prepare_inherited_positive_recovery_v1.py').read_bytes()),
        v30_training_report_sha256=sha(training_blob), trainer_source_commit=source_commit,
        trainer_source_sha256=sha(source), original_metadata_sha256=training['features_index_sha256'],
        v30_selected_step=70000, v30_positive_main_slots_retained=32,
        actual_v30_prefix_negative_slots=[32, 48],
        actual_batch_ast_symbolic_positive_retention_batches=50,
        source_command_feature_variants=34150, original_sampler_command_support=True,
        expected_command_main_draws_per_batch_at_equal_engines=32 * (1 / 3) * (34150 / 60000),
        baseline_report_sha256=sha(baseline_blob), existing_norman_command_hits='0/9',
        actual_batch_source=ast.get_source_segment(source.decode(), function),
        conclusion='The current later bare-only recipe has lost inherited command support, but original v30 '
                   'already exposed that command distribution in all32 positive main slots. '
                   'The later overwrite cannot explain the pre-existing v30 Norman command failure by itself. '
                   'Restore8 remains a mixed-source/temporal intervention to test, not an established root-cause fix.',
        training=False, model_inference=False, acceptance_claim=False,
        limits=[
            'Source/dataflow support and theoretical expectation, not replay of original v30 ordered draws or proof of per-step command counts.',
            'Symbolic feature fixture reuses the preparation harness; proves positive rows survive batch mutation, not actual TensorFlow optimization.',
            'The v30 prefix-negative setting and later paired/source recipes differ; this is not a matched causal quality comparison.',
            'Existing consumed baseline only, no new scoring, source generation, final, audio/device or formal process operation.'
        ])
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: report[k] for k in ('v30_positive_main_slots_retained',
        'original_sampler_command_support', 'expected_command_main_draws_per_batch_at_equal_engines',
        'existing_norman_command_hits', 'conclusion')}))


if __name__ == '__main__':
    main()
