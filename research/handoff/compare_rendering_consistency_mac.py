"""Compare the GPU's pinned TensorFlow loss to the independent NumPy derivative.

Formula/runtime check only: no original model, initialization, sampler, training
update or coefficient selection. Actual-network preflight is separate.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import types

import numpy as np

from probe_rendering_consistency_gradient_mac import objective_and_gradient


REF = "c3f89a9d08e6c55f19cc675d0601c7dd7d55651d"
SOURCE = "research/wakeword_p4/training/rendering_consistency.py"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Fresh output required")
    root = Path(__file__).resolve().parents[2]
    source = subprocess.check_output(["git", "-C", str(root), "show", REF + ":" + SOURCE])
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    import tensorflow as tf
    tf.config.set_visible_devices([], "GPU")
    tf.config.threading.set_inter_op_parallelism_threads(2)
    tf.config.threading.set_intra_op_parallelism_threads(2)
    module = types.ModuleType("pinned_gpu_rendering_consistency")
    exec(compile(source, REF + ":" + SOURCE, "exec"), module.__dict__)
    normal = np.random.default_rng(726).normal(size=(2, 4, 5, 3))
    full = np.random.default_rng(727).normal(size=(2, 4, 5, 300))
    equal = np.zeros_like(normal)
    equal[..., 0] = 1
    partial_zero = normal.copy()
    partial_zero[0, 0, 0] = 0
    cases = dict(normal3=normal, normal300=full, zero=np.zeros_like(normal),
                 constant_unit=equal, tiny=normal * 1e-10, one_zero_quiet=partial_zero)
    comparisons = []
    for name, values in cases.items():
        for dtype in (np.float64, np.float32):
            x = values.astype(dtype)
            reference_loss, reference_gradient = objective_and_gradient(x)
            tensor = tf.convert_to_tensor(x)
            with tf.GradientTape() as tape:
                tape.watch(tensor)
                observed_loss = module.rendering_consistency(tensor)
            observed_gradient = tape.gradient(observed_loss, tensor)
            if observed_gradient is None:
                raise ValueError("Disconnected consistency gradient")
            actual_loss = float(observed_loss.numpy())
            actual_gradient = observed_gradient.numpy().astype(np.float64)
            if not np.isfinite(actual_gradient).all() or not np.isfinite(actual_loss):
                raise ValueError("Nonfinite consistency result")
            tolerance = 1e-12 if dtype == np.float64 else 2e-6
            np.testing.assert_allclose(actual_loss, reference_loss, atol=tolerance, rtol=tolerance)
            scale = max(1., float(np.max(np.abs(reference_gradient))))
            np.testing.assert_allclose(actual_gradient, reference_gradient,
                                       atol=tolerance * scale, rtol=tolerance)
            comparisons.append(dict(case=name, dtype=np.dtype(dtype).name,
                                    dimensions=x.shape[-1], loss=actual_loss,
                                    gradient_norm=float(np.linalg.norm(actual_gradient)),
                                    loss_absolute_error=abs(actual_loss - reference_loss),
                                    gradient_maximum_absolute_error=float(np.max(np.abs(
                                        actual_gradient - reference_gradient))),
                                    gradient_error_scale=scale, tolerance=tolerance))
    result = dict(scope=__doc__, implementation_commit=REF,
                  implementation_sha256=hashlib.sha256(source).hexdigest(),
                  independent_numpy_source_sha256=hashlib.sha256(
                      Path(__file__).with_name("probe_rendering_consistency_gradient_mac.py").read_bytes()).hexdigest(),
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  tensorflow_version=tf.__version__, numpy_version=np.__version__,
                  comparisons=comparisons, all_passed=True,
                  limits="Pinned loss implementation only. This does not establish actual-network head "
                         "disconnection, BN invariance, initialization gradient scale or training effectiveness.")
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(dict(cases=len(comparisons), all_passed=True,
                         implementation_sha256=result["implementation_sha256"])))


if __name__ == "__main__":
    main()
