#!/usr/bin/env python3
"""Prepare a private standalone head probe with REF numerical goldens.

Real histories are training-only PTQ representatives; synthetic byte-boundary
cases test arithmetic, not recognition. No held-out detector scores are read.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
from ai_edge_litert.interpreter import Interpreter, OpResolverType


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--expected-model-sha256', required=True)
    parser.add_argument('--representatives', type=Path, required=True)
    parser.add_argument('--expected-representatives-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if sha(args.model) != args.expected_model_sha256 or sha(args.representatives) != args.expected_representatives_sha256:
        raise ValueError('Model or representative-file identity mismatch')
    reps = np.load(args.representatives, allow_pickle=False)
    if reps.shape != (512, 16, 96, 1) or reps.dtype != np.float32 or not np.isfinite(reps).all():
        raise ValueError('Expected exactly 512 finite training-only float32 histories')
    integers = reps[:256] * 128
    if not np.array_equal(integers, np.rint(integers)) or integers.min() < -128 or integers.max() > 127:
        raise ValueError('Training histories must map exactly to byte-grid embeddings')
    real = integers.astype(np.int8)
    boundaries = np.stack([np.full((16, 96, 1), n, np.int8) for n in (-128, -127, -1, 0, 1, 126, 127)])
    random = np.random.default_rng(726).integers(-128, 128, (32, 16, 96, 1), dtype=np.int8)
    checker = np.tile(np.array([-128, 127], np.int8), 768).reshape(16, 96, 1)
    patterns = np.stack([checker, checker[::-1, ::-1], np.roll(checker, 1, axis=1), np.zeros_like(checker)])
    inputs = np.concatenate([real, boundaries, random, patterns])
    interpreter = Interpreter(model_path=str(args.model), experimental_op_resolver_type=OpResolverType.BUILTIN_REF)
    interpreter.allocate_tensors()
    di, do = interpreter.get_input_details()[0], interpreter.get_output_details()[0]
    if (tuple(di['shape']) != (1, 16, 96, 1) or tuple(do['shape']) != (1, 1)
            or di['dtype'] != np.int8 or do['dtype'] != np.int8
            or di['quantization'] != (1 / 128, 0) or do['quantization'] != (1 / 256, -128)):
        raise ValueError('Unexpected head interface')
    outputs = []
    for x in inputs:
        interpreter.set_tensor(di['index'], x[None])
        interpreter.invoke()
        outputs.append(interpreter.get_tensor(do['index']).reshape(-1))
    expected = np.asarray(outputs, np.int8)
    args.output.mkdir(parents=True, exist_ok=False)
    source = Path(__file__).with_name('head_probe')
    shutil.copytree(source, args.output, dirs_exist_ok=True)
    shutil.copyfile(args.model, args.output / 'main/head.tflite')
    inputs.tofile(args.output / 'main/inputs.bin')
    expected.tofile(args.output / 'main/expected.bin')
    manifest = {
        'scope': 'Standalone head numerical/resources only; no recognition acceptance',
        'model_sha256': sha(args.model), 'model_bytes': args.model.stat().st_size,
        'representatives_sha256': sha(args.representatives), 'cases': len(inputs),
        'real_training_histories': len(real), 'synthetic_numeric_cases': len(inputs) - len(real),
        'input_bytes_per_case': 1536, 'output_bytes_per_case': 1,
        'inputs_sha256': sha(args.output / 'main/inputs.bin'),
        'expected_sha256': sha(args.output / 'main/expected.bin'),
        'resolver': 'LiteRT BUILTIN_REF', 'seed': 726,
        'expected_modes': ['internal64k_model_flash', 'internal64k_model_psram', 'psram64k_model_psram'],
        'gate': 'Each mode completes every case with zero byte mismatches; no tolerance relaxation',
    }
    (args.output / 'vectors-manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
