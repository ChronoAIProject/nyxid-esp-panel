"""Audit received original initialization tensors and gradients on Mac.

Independent saved-array arithmetic plus replay of every paired window from
authenticated original raw features. No model/autodiff, historical main-tensor
reproduction or same-instance weight-snapshot claim.
"""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath

import numpy as np


REPORT = "0cb06dcb3276125d56b0bc260ef7c6be7e57921fed03b4b799a482e3d675e42f"
PACK = "7b798653f73e2dcd3b01fd5b7092c947a26b9c8ec2525ff295de2e6478042e44"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Fresh output required")
    root, features = args.input.resolve(), args.feature_root.resolve()
    raw_report = (root / "gradient-probe.json").read_bytes()
    assert sha(raw_report) == REPORT
    report = json.loads(raw_report)
    for name, field in (("probe-gradients.npz", "gradients_sha256"),
                        ("probe-batches.npz", "batches_sha256"),
                        ("initialization.json", "initialization_sha256")):
        assert sha((root / name).read_bytes()) == report[field]
    pack_bytes = (features / "factorized-training-features-v1/report.json").read_bytes()
    assert sha(pack_bytes) == PACK
    records = json.loads(pack_bytes)["records"]
    lookup = {(int(r["voice"]), r["length_multiplier"], r["target"], r["arm"]): r
              for r in records}
    assert len(lookup) == 40320
    voices = sorted(np.random.default_rng(726).permutation(904)[:672].tolist())
    rng = np.random.default_rng(227726)
    layout = report["variable_layout"]
    offsets = np.cumsum([0] + [v["size"] for v in layout])
    assert all(v["size"] == int(np.prod(v["shape"])) for v in layout)
    encoder_mask = np.concatenate([np.repeat(v["encoder"], v["size"]) for v in layout])
    assert len(layout) == 15 and encoder_mask.sum() == 21540 and (~encoder_mask).sum() == 301
    with np.load(root / "probe-gradients.npz", allow_pickle=False) as data:
        assert set(data.files) == {"main", "pair", "supervised", "consistency"}
        gradients = {k: data[k].astype(np.float64) for k in data.files}
    assert all(v.shape == (50, offsets[-1]) and np.isfinite(v).all()
               for v in gradients.values())
    # Match the original saved-array arithmetic tolerance, without changing it
    # after viewing a difference. Separate backward passes can round differently.
    np.testing.assert_allclose(gradients["main"] + gradients["pair"],
                               gradients["supervised"], rtol=1e-5, atol=1e-5)
    assert np.count_nonzero(gradients["consistency"][:, ~encoder_mask]) == 0
    ns = np.sqrt(np.sum(gradients["supervised"][:, encoder_mask] ** 2, axis=1))
    nd = np.sqrt(np.sum(gradients["consistency"][:, encoder_mask] ** 2, axis=1))
    assert np.all(ns > 0) and np.all(nd > 0)
    cosines = np.sum(gradients["supervised"][:, encoder_mask] *
                     gradients["consistency"][:, encoder_mask], axis=1) / (ns * nd)
    compared_variables = paired_views = 0
    with np.load(root / "probe-batches.npz", allow_pickle=False) as batches:
        arrays = {k: batches[k] for k in batches.files}
        assert set(arrays) == {p + "_" + s for p in ("main", "pair")
                               for s in ("x", "y", "weight")}
        for prefix, count, positives in (("main", 128, 32), ("pair", 40, 10)):
            x, y, weight = [arrays[prefix + "_" + k] for k in ("x", "y", "weight")]
            assert x.shape == (50, count, 150, 40) and y.shape == (50, count, 1)
            assert weight.shape == (50, count) and all(a.dtype == np.float32 for a in (x, y, weight))
            assert np.isfinite(x).all() and x.min() >= 0 and x.max() <= 26
            assert set(np.unique(y)) == {0., 1.} and np.all(y.sum(axis=(1, 2)) == positives)
            np.testing.assert_array_equal(weight, np.where(y[:, :, 0] == 1, 1., 20.))
        for i, row in enumerate(report["traces"]):
            assert row["step"] == i + 1
            for prefix in ("main", "pair"):
                data = b"".join(arrays[prefix + "_" + k][i].tobytes() for k in ("x", "y", "weight"))
                assert sha(data) == row[prefix + "_tensor_sha256"]
            np.testing.assert_allclose([ns[i], nd[i], cosines[i]],
                                       [row["encoder"][k] for k in (
                                           "supervised_norm", "consistency_norm", "cosine")],
                                       rtol=1e-12, atol=1e-14)
            for j, variable in enumerate(row["per_variable"]):
                assert variable["path"] == layout[j]["path"]
                section = slice(offsets[j], offsets[j + 1])
                a, b = gradients["supervised"][i, section], gradients["consistency"][i, section]
                na, nb = np.sqrt(np.sum(a * a)), np.sqrt(np.sum(b * b))
                np.testing.assert_allclose([na, nb], [variable["supervised_norm"], variable["consistency_norm"]],
                                           rtol=1e-12, atol=1e-14)
                if na and nb:
                    np.testing.assert_allclose(np.sum(a * b) / (na * nb), variable["cosine"],
                                               rtol=1e-12, atol=1e-14)
                else:
                    assert variable["cosine"] is None
                compared_variables += 1
            speed = (.9, 1., 1.1)[int(rng.integers(3))]
            voice_indices = rng.choice(len(voices), 2, replace=False)
            selected = [voices[j] for j in voice_indices]
            cells = [lookup[v, speed, p, a] for v in selected
                     for p in ("hey_nyx", "hey_nick", "hey_next", "hey_mix")
                     for a in ("quiet", "noise25", "combined", "noise15", "rir")]
            legal = [end for end in range(150, 351) if end % 3 == 0
                     and all(r["feature_crop_end"] <= end < r["raw_shape"][0] for r in cells)]
            end = legal[int(rng.integers(len(legal)))]
            trace = report["pair_trace"][i]
            assert trace["block"] == i and trace["voices"] == selected
            assert trace["speed"] == speed and trace["endpoint_inclusive"] == end
            assert trace["pack_indices"] == [r["pack_index"] for r in cells]
            for j, record in enumerate(cells):
                relative = PurePosixPath(record["raw"])
                assert len(relative.parts) == 4 and relative.parts[0] == ".."
                path = (features / Path(*relative.parts[1:])).resolve()
                assert path.is_relative_to(features)
                assert sha(path.read_bytes()) == record["raw_file_sha256"]
                raw = np.load(path, allow_pickle=False)
                assert sha(raw.tobytes()) == record["raw_sha256"]
                assert raw.dtype == np.uint16 and list(raw.shape) == record["raw_shape"]
                window = raw[end - 149:end + 1]
                assert sha(window.tobytes()) == trace["window_sha256"][j]
                np.testing.assert_array_equal(arrays["pair_x"][i, j],
                                               np.clip(window.astype(np.float32) / 25.6, 0, 26))
                assert arrays["pair_y"][i, j, 0] == float(record["target"] == "hey_nyx")
                paired_views += 1
    ratio = float(np.median(ns) / np.median(nd))
    coefficient = min(1., 10. ** np.floor(np.log10(.1 * ratio)))
    np.testing.assert_allclose(ratio, report["R"], rtol=1e-14)
    assert coefficient == report["proposed_fixed_coefficient"]
    np.testing.assert_allclose(coefficient * nd / ns, report["scaled_gradient_ratios"], rtol=1e-12)
    result = dict(scope=__doc__, original_report_sha256=REPORT,
                  original_gradients_sha256=report["gradients_sha256"],
                  original_batches_sha256=report["batches_sha256"],
                  source_pack_report_sha256=PACK, batches=50, exact_paired_window_replays=paired_views,
                  per_variable_gradient_records_verified=compared_variables,
                  all_main_pair_tensor_sha256_match=True, all_labels_and_weights_match=True,
                  all_fixed_seed_ordered_pair_schedules_match=True,
                  consistency_saved_head_gradients_exact_zero=True,
                  maximum_supervised_gradient_sum_error=float(np.max(np.abs(
                      gradients["supervised"] - gradients["main"] - gradients["pair"]))),
                  supervised_median_norm=float(np.median(ns)), consistency_median_norm=float(np.median(nd)),
                  R=ratio, fixed_coefficient=float(coefficient),
                  scaled_gradient_ratio_range=[float(np.min(nd / ns)), float(np.max(nd / ns))],
                  negative_gradient_cosine_batches=int(np.count_nonzero(cosines < 0)),
                  script_sha256=sha(Path(__file__).read_bytes()),
                  limits="Original same-instance H5/inventory was not saved. Model identity is documented "
                         "by original initialization metadata, not independently loaded here. No original-model "
                         "autodiff, historical main tensor reconstruction, training improvement or acceptance claim.")
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
