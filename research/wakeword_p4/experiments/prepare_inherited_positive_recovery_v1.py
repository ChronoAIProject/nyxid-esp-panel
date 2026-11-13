"""Authenticate/apply optional input repair and test its actual batch AST.

Symbolic feature values only. Does not load training arrays or TensorFlow;
full-trainer, real-input optimizer integration remains a separate prerequisite.
"""
import argparse
import ast
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import numpy as np


BASE = '2228dff55d11111c3e68a4a27433ef8b472a91c3d185f124a9018425b5f03904'
PATCHED = 'b97b2debc92ca106ab199af60ac144aa7a78ce4a26e39621bc5b179fb637917e'
PATCH = '65e336cfec60d191748015d447e3c3f780ad4e82ac214160de3547a6297e1f09'
FLAG = 'recover_inherited_positive_draws'


def sha(data):
    return hashlib.sha256(data).hexdigest()


class StripIntervention(ast.NodeTransformer):
    def visit_If(self, node):
        if any(isinstance(n, ast.Attribute) and n.attr == FLAG for n in ast.walk(node.test)):
            return None
        return self.generic_visit(node)

    def visit_Expr(self, node):
        if (isinstance(node.value, ast.Call) and node.value.args
                and isinstance(node.value.args[0], ast.Constant)
                and node.value.args[0].value == '--recover-inherited-positive-draws'):
            return None
        return self.generic_visit(node)


class TrackedRNG:
    def __init__(self):
        self.rng = np.random.default_rng(726)
        self.order = None

    def integers(self, *args, **kwargs):
        return self.rng.integers(*args, **kwargs)

    def permutation(self, *args, **kwargs):
        self.order = self.rng.permutation(*args, **kwargs)
        return self.order


class SymbolicPack:
    def __init__(self, value, slots=None):
        self.value, self.slots = value, slots
        self.rng = np.random.default_rng(1000 + value)

    def first80(self):
        ids = self.rng.uniform(self.value, self.value + .1, 80)
        return np.broadcast_to(ids[:, None, None], (80, 150, 40)).copy().astype(np.float32)

    def replace(self, old):
        result = old.copy()
        values = self.rng.uniform(self.value, self.value + .1, len(self.slots))
        result[self.slots] = values[:, None, None]
        return result


def batch_environment(tree, enabled, complete_enabled):
    batch = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == 'batch')
    args = SimpleNamespace(ambient_sampling='rows', recover_inherited_positive_draws=enabled,
        recover_erased_complete_coverage=complete_enabled, prefix_negatives=0,
        mask_prefix_loss=False, time_mask_width=0, frequency_mask_width=0)
    rng = TrackedRNG()
    packs = [SymbolicPack(5), SymbolicPack(10, np.r_[0:8, 32:44]),
             SymbolicPack(15, np.r_[0:8, 32:44]), SymbolicPack(20, np.r_[8:16, 44:56])]
    drawn = {}

    def sample(label, count, mine=True):
        values = rng.rng.uniform(0, 25 * 25.6, count).astype(np.float32)
        result = np.broadcast_to(values[:, None, None], (count, 150, 40)).copy()
        if label:
            drawn['positive'] = scale(result)
        return result

    def scale(data):
        return np.clip(np.asarray(data, dtype=np.float32) / 25.6, 0, 26)

    env = dict(np=np, args=args, rng=rng, sample=sample, scale=scale,
               ambient={'training': [(None, np.zeros((200, 40), np.uint16))]}, frames=150,
               full_source=packs[0], complete_coverage=packs[1], us_coverage=packs[2],
               balanced_coverage=packs[3], hard_ambient=None, augmentation_rng=None,
               mask_spectrograms=lambda x, *_: x)
    code = compile(ast.fix_missing_locations(ast.Module(body=[deepcopy(batch)], type_ignores=[])),
                   '<authenticated actual batch AST; symbolic packs>', 'exec')
    exec(code, env)
    return env['batch'], rng, packs, drawn


def state(rng, packs):
    return json.dumps([rng.rng.bit_generator.state] + [p.rng.bit_generator.state for p in packs], sort_keys=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu-ref', required=True)
    parser.add_argument('--metadata', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    args.output.mkdir(parents=True)
    metadata = args.metadata.read_bytes()
    assert sha(metadata) == '9db652f0769e49be6e924aadbfc3cd3725870b2eb558ea5b25b25481970d1d4b'
    positives = [r for r in json.loads(metadata) if r['split'] == 'training' and r['label'] == 1]
    engines = Counter(r['engine'] for r in positives)
    assert engines == {'piper': 60000, 'kokoro': 3000, 'qwen-cuda': 300}
    command_texts = {'Hey Nyx, what time is it?', 'Hey Nyx, can you hear me?',
                     'Hey Nyx, turn on the light.', 'Hey Nyx, check the volume.'}
    commands = Counter(r['engine'] for r in positives if r['text'] in command_texts)
    assert commands == {'piper': 34150}
    base = subprocess.check_output(['git', 'show', args.gpu_ref + ':research/wakeword_p4/training/train_stream.py'])
    assert sha(base) == BASE
    patch = Path(__file__).with_name('recover_inherited_positive_draws_v1.patch')
    assert sha(patch.read_bytes()) == PATCH
    source = args.output / 'research/wakeword_p4/training/train_stream.py'
    source.parent.mkdir(parents=True)
    source.write_bytes(base)
    subprocess.run(['git', 'apply', '--check', str(patch.resolve())], cwd=args.output, check=True)
    subprocess.run(['git', 'apply', str(patch.resolve())], cwd=args.output, check=True)
    patched = source.read_bytes()
    assert sha(patched) == PATCHED
    original_tree, patched_tree = ast.parse(base), ast.parse(patched)
    compile(patched, str(source), 'exec')
    stripped = StripIntervention().visit(deepcopy(patched_tree))
    assert ast.dump(stripped) == ast.dump(original_tree)
    cases = []
    destinations = np.arange(24, 32)
    untouched = np.setdiff1d(np.arange(128), destinations)
    for complete_enabled in (False, True):
        contexts = [batch_environment(original_tree, False, complete_enabled),
                    batch_environment(patched_tree, False, complete_enabled),
                    batch_environment(patched_tree, True, complete_enabled)]
        for step in range(1, 51):
            outputs = [context[0]() for context in contexts]
            for a, b in zip(outputs[0], outputs[1], strict=True):
                np.testing.assert_array_equal(a, b)
            np.testing.assert_array_equal(outputs[1][1], outputs[2][1])
            np.testing.assert_array_equal(outputs[1][2], outputs[2][2])
            assert state(*contexts[0][1:3]) == state(*contexts[1][1:3]) == state(*contexts[2][1:3])
            orders = [c[1].order for c in contexts]
            np.testing.assert_array_equal(orders[0], orders[1])
            np.testing.assert_array_equal(orders[0], orders[2])
            before = outputs[1][0][np.argsort(orders[1])]
            after = outputs[2][0][np.argsort(orders[2])]
            np.testing.assert_array_equal(before[untouched], after[untouched])
            np.testing.assert_array_equal(after[destinations], contexts[2][3]['positive'][destinations])
            changed = np.flatnonzero(np.any(before != after, axis=(1, 2)))
            np.testing.assert_array_equal(changed, destinations)
            np.testing.assert_array_equal(outputs[2][1][np.argsort(orders[2])][destinations], np.ones((8, 1)))
            np.testing.assert_array_equal(outputs[2][2][np.argsort(orders[2])][destinations], np.ones(8))
        cases.append(dict(complete_recovery_enabled=complete_enabled, batches_per_arm=50,
                          changed_slots=destinations.tolist(), other120_main_exact=True,
                          original8_positive_draws_restored=True, all_labels_weights_order_rng_exact=True))
    report = dict(schema='inherited-positive-recovery-source-preparation-v1',
        source_sha256=sha(Path(__file__).read_bytes()), patch_sha256=PATCH,
        base_sha256=BASE, patched_sha256=PATCHED, exact_patch_apply_and_compile=True,
        remove_optional_intervention_ast_equals_base=True,
        original_positive_metadata_sha256=sha(metadata), positive_engine_counts=dict(engines),
        command_engine_counts=dict(commands),
        expected_restored_commands_per_batch_at_source_exponent0=
            8 * sum(commands[e] / engines[e] for e in engines) / len(engines),
        symbolic_actual_batch_ast_cases=cases,
        real_training_arrays_loaded=False, optimizer_updates=0, model_inference=False,
        acceptance_claim=False,
        limits='Symbolic feature identities and mocked source packs with actual batch AST. '
               'Does not authenticate actual original positive samples, real sampler arrays, TensorFlow gradients, '
               'command selection frequency or quality. Full-trainer default-off/enabled real two-update capture '
               'with original source IDs/texts/crop endpoints is still required before formal training.')
    (args.output / 'preparation.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
