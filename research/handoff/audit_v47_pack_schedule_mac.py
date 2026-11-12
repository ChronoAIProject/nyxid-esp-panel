"""Authenticate v47 pack records and independently reconstruct its first draws.

Git metadata/source and schedule arithmetic only. No original NPZ/raw array is
read here, and GPU tensor/RNG-preservation audit claims remain GPU-executed.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path, PurePosixPath
import shlex
import subprocess

import numpy as np


REF = "1518be4177556901cde3919665ffd9d96c7c2768"
SMOKE_REF = "37d940dd019a29e4745a5f8fa97ef4dcaed22dca"
ROOT = Path(__file__).resolve().parents[2]
PACK = "1a6cbf491c624c58cf96cfdac882966a7863927f546ac553075248caacfaaf5f"
REPORT = "7b798653f73e2dcd3b01fd5b7092c947a26b9c8ec2525ff295de2e6478042e44"
OLD_REPORT = "93733755fec50b41cf71fcc8e2efa9fdfc14956a61e8b50e98e4bd110d5248b3"
CONTROLS = "fb3f71531d3ad15cbb90f23636c5ac8b113b1ae8103c79df68163c5b55b078ab"
ARMS = ("quiet", "noise25", "combined", "noise15", "rir")
PHRASES = ("hey_nyx", "hey_nick", "hey_next", "hey_mix")


def sha(data):
    return hashlib.sha256(data).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def blob(name, ref=REF):
    return subprocess.check_output([
        "git", "-C", str(ROOT), "show", ref + ":research/wakeword_p4/" + name])


def command(version, name):
    tokens = shlex.split(blob("experiments/" + name).decode().replace("\\\n", " "))
    # Compare the concrete executable command after normalizing only the
    # candidate-specific pack flag/path, output path and log destination.
    start = tokens.index("python3")
    return [token.replace("v" + str(version), "vCANDIDATE")
            .replace("--factorized-coverage-pack", "--balanced-coverage-pack")
            .replace("factorized-training-features-v1", "piper-balanced-training-features-v1")
            for token in tokens[start:]]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists() and not args.trace.exists(), "Fresh outputs required")
    names = (
        "4090-factorized-training-pack-original-v1.json",
        "4090-factorized-training-pack-audit-v1.json",
        "4090-v47-sampler-precheck-v1.json",
        "4090-piper-balanced-training-features-original-v1.json",
        "4090-balanced-rendering-factorial-controls-v1.json",
    )
    inputs = {name: blob("results/" + name) for name in names}
    pack, audit, sampler, old, controls = [json.loads(inputs[name]) for name in names]
    require(sha(inputs[names[0]]) == REPORT == audit["report_sha256"]
            and pack["pack_sha256"] == audit["pack_sha256"] == PACK,
            "Original factorized pack/report identities differ")
    require(sha(inputs[names[3]]) == OLD_REPORT and sha(inputs[names[4]]) == CONTROLS,
            "Original input records differ")
    records = pack["records"]
    require(len(records) == pack["included"] == pack["attempted_variants"] == 40320
            and pack["wake_models_scored"] and not pack["final_data_used"],
            "Pack status/coverage differs")
    groups, cells, arms = defaultdict(list), Counter(), Counter()
    for j, row in enumerate(records):
        i, variant = divmod(j, 5)
        original = (old["records"][3 * i + variant] if variant < 3
                    else controls["records"][2 * i + variant - 3])
        digest = (original["raw_file_sha256"] if variant < 3
                  else original["files"]["raw"]["sha256"])
        source_path = (original["raw"] if variant < 3
                       else original["files"]["raw"]["path"])
        parent = ("piper-balanced-training-features-v1" if variant < 3
                  else "balanced-rendering-factorial-v1")
        require(row["pack_index"] == j and row["source_index"] == i
                and row["variant"] == variant and row["arm"] == ARMS[variant]
                and row["origin"] == dict(report_sha256=OLD_REPORT if variant < 3 else CONTROLS,
                                          record_index=3 * i + variant if variant < 3 else 2 * i + variant - 3)
                and row["raw"] == "../" + parent + "/" + source_path
                and not PurePosixPath(source_path).is_absolute()
                and ".." not in PurePosixPath(source_path).parts
                and row["raw_file_sha256"] == digest, "Raw origin/subset differs")
        require(all(row[k] == original[k] for k in ("source", "voice", "target", "signal_samples", "feature_crop_end"))
                and row["length_multiplier"] == original["length_multiplier" if variant < 3 else "speed"]
                and row["split"] == "training" and row["entire_signal_fits"] and row["mechanical_pass"],
                "Source/container metadata differs")
        if variant < 3:
            require(row["raw_shape"] == original["raw_shape"]
                    and row["raw_sha256"] == original["raw_sha256"], "Original raw data links differ")
        else:
            require(original["files"]["raw"]["size"] == 128 + row["raw_shape"][0] * 80,
                    "Control raw serialization geometry differs")
        length, bands = row["raw_shape"]
        low, high = row["feature_crop_end"], min(350, length)
        require(bands == 40 and 150 <= low <= high
                and low == (32000 + row["signal_samples"] + 159) // 160 - 1,
                "Complete-container end geometry differs")
        # Derive the complete legal interval from inequalities independently.
        legal = [end for end in range(150, length + 1)
                 if (end - 150) * 160 <= 32000
                 and (end + 1) * 160 >= 32000 + row["signal_samples"]]
        require(legal == list(range(low, high + 1)), "Legal end set differs")
        groups[(row["voice"], row["target"])].append(j)
        cells[(row["voice"], row["target"], row["length_multiplier"], row["arm"])] += 1
        arms[row["arm"]] += 1
    voices = sorted(np.random.default_rng(726).permutation(904)[:672].tolist())
    expected_cells = {(str(voice), phrase, speed, arm) for voice in voices
                      for phrase in PHRASES for speed in (.9, 1., 1.1) for arm in ARMS}
    require(set(cells) == expected_cells and set(cells.values()) == {1}
            and len(groups) == 2688 and {len(group) for group in groups.values()} == {15},
            "Training identity/full factorial grouping differs")
    helper = sampler["balanced_coverage"]
    require(helper["pack_sha256"] == PACK and helper["report_sha256"] == REPORT
            and helper["seed"] == 207726 and helper["temporal"]["seed"] == 217726,
            "Sampler input/seed identities differ")
    source_rng, end_rng = np.random.default_rng(207726), np.random.default_rng(217726)
    counts, trace = Counter(), []
    for batch in range(50):
        for slot in range(20):
            # Integer indexing is independent of the helper's choice(array).
            phrase = PHRASES[0] if slot < 8 else PHRASES[1 + int(source_rng.integers(3))]
            voice = str(voices[int(source_rng.integers(len(voices)))])
            eligible = sorted(groups[(voice, phrase)])
            index = eligible[int(source_rng.integers(15))]
            row = records[index]
            low = row["feature_crop_end"] if slot < 14 else 150
            high = min(350, row["raw_shape"][0]) if slot < 14 else row["raw_shape"][0]
            end = int(end_rng.integers(low, high + 1))
            counts[voice + "/" + phrase] += 1
            trace.append(dict(batch=batch, slot=slot + 8 if slot < 8 else slot + 36,
                              pack_index=index, source=row["source"], arm=row["arm"], end=end))
    require(dict(counts) == sampler["counts"] and sampler["batches"] == 50
            and sampler["draws"] == 1000, "Independent source/phrase schedule census differs")
    sources = {name: blob(name) for name in (
        "experiments/assemble_factorized_training_pack_v1.py",
        "experiments/audit_factorized_training_pack_v1.py",
        "experiments/v47_sampler_precheck.py", "training/factorized_coverage.py",
        "training/train_stream.py")}
    require(sha(sources["experiments/assemble_factorized_training_pack_v1.py"]) == pack["script_sha256"]
            and sha(sources["experiments/audit_factorized_training_pack_v1.py"]) == audit["script_sha256"]
            and sha(sources["experiments/v47_sampler_precheck.py"]) == sampler["script_sha256"]
            and sha(sources["training/factorized_coverage.py"]) == helper["source_sha256"]
            and sha(sources["training/train_stream.py"]) == sampler["trainer_sha256"],
            "Original implementation identities differ")
    require(command(47, "v47_factorized_rendering_coverage.sh")
            == command(46, "v46_balanced_training_coverage.sh"), "Other executable recipe flags differ")
    execution_inputs = {name: blob("results/" + name, SMOKE_REF) for name in (
        "4090-v47-execution-source-pins-v1.json", "4090-v47-smoke-training-v1.json",
        "4090-v47-smoke-export-audit-v1.json")}
    pins, smoke, export = [json.loads(data) for data in execution_inputs.values()]
    for path, digest in pins.items():
        require(path.startswith("research/wakeword_p4/")
                and sha(blob(path.removeprefix("research/wakeword_p4/"), SMOKE_REF)) == digest,
                "Pinned executable source differs")
    for name, digest in smoke["source_files_sha256"].items():
        require(Path(name).name == name and sha(blob("training/" + name, SMOKE_REF)) == digest,
                "Original smoke training source differs")
    prior = json.loads(blob("results/4090-v46-training.json", "65ebbcce0c87bb8a51cb5abfcd56a1c9bfe0ab4f"))
    previous_sources = prior["source_files_sha256"]
    require(set(smoke["source_files_sha256"]) - set(previous_sources) == {"factorized_coverage.py"}
            and not set(previous_sources) - set(smoke["source_files_sha256"])
            and [name for name in previous_sources
                 if smoke["source_files_sha256"][name] != previous_sources[name]] == ["train_stream.py"],
            "Inherited training implementations differ")
    allowed_changes = {"steps", "selected_step", "balanced_coverage", "history", "seconds",
                       "model_bytes", "model_sha256", "source_files_sha256"}
    require(set(smoke) == set(prior)
            and all(smoke[name] == prior[name] for name in smoke if name not in allowed_changes),
            "Inherited recipe or initialization differs")
    require(sha(execution_inputs["4090-v47-smoke-training-v1.json"]) == export["training_sha256"]
            and smoke["model_sha256"] == export["model_sha256"]
            == "bbca7175b1f7e928c0e992f63c095c3cf0f2601877662de6d6925259c8c9263c"
            and smoke["steps"] == export["steps"] == 2
            and smoke["selected_step"] == export["selected_step"] == 1
            and export["parameters"] == 22801 and export["model_bytes"] == 54312
            and export["all_saved_weights_finite"] and export["frozen_bn_unchanged"]
            and export["input_shape"] == [1, 3, 40] and export["output_shape"] == [1, 1]
            and export["input_dtype"] == "int8" and export["output_dtype"] == "uint8"
            and export["source_sha256"] == smoke["source_files_sha256"]["audit_stream_export.py"]
            and not export["candidate_promoted"], "Original smoke/export record differs")
    trace_data = (json.dumps(trace, separators=(",", ":")) + "\n").encode()
    result = dict(scope=__doc__, gpu_commit=REF, numpy_version=np.__version__,
                  input_sha256={name: sha(data) for name, data in inputs.items()},
                  implementation_sha256={name: sha(data) for name, data in sources.items()},
                  pack_sha256=PACK, connected_records=40320, original_subset_connected=24192,
                  control_subset_connected=16128, groups=2688, variants_per_group=15,
                  arms=dict(arms), all_legal_endpoint_intervals_exact=True,
                  schedule_draws=1000, source_phrase_census_matches_gpu=True,
                  locally_reconstructed_trace_sha256=sha(trace_data),
                  other_executable_recipe_flags_identical=True, all_checks_pass=True,
                  execution_source_ref=SMOKE_REF,
                  execution_input_sha256={name: sha(data) for name, data in execution_inputs.items()},
                  original_training_sources_authenticated=len(smoke["source_files_sha256"]),
                  pinned_execution_sources_authenticated=len(pins),
                  inherited_initialization_and_other_recipe_records_identical=True,
                  original_smoke_export_record_contract_matches=True,
                  accepted=False, script_sha256=sha(Path(__file__).read_bytes()),
                  limits="Original git records/source and arithmetic only. GPU raw/NPZ "
                         "files and tensor batches were not received/replayed on Mac. "
                         "The local ordered source/end trace is independently reconstructed; "
                         "GPU report retains only the source/phrase census, so equality of "
                         "every original GPU ordered draw/end cannot be independently "
                         "verified from that report. GPU untouched-slot/RNG and finite-weight/export claims remain "
                         "GPU-executed; Mac authenticates their source/record identities. "
                         "No inference, optimization, model acceptance, "
                         "heldout/final or device/audio access.")
    for path, data in ((args.trace, trace_data), (args.output, (json.dumps(result, indent=2) + "\n").encode())):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
