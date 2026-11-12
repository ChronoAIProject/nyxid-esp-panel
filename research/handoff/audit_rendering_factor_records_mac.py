"""Audit the frozen rendering-factor source grid and artifact ledger.

This reads original git records only. It does not read, rehash or replay the
GPU's waveform/feature arrays and cannot certify their contents or recognition.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import subprocess


REF = "04f8a31061130413b971496e569b253a972f1877"
ROOT = Path(__file__).resolve().parents[2]
PACK_SHA = "93733755fec50b41cf71fcc8e2efa9fdfc14956a61e8b50e98e4bd110d5248b3"
MANIFEST_SHA = "e3a69b86294bf91db29777428e785252411710834d22ba92a037d595c963cdc0"
GENERATOR_SHA = "c36a726bdaea5859f40ce0728689cfa7383678be8aece89c459e2f3d4a779175"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def blob(name):
    return subprocess.check_output([
        "git", "-C", str(ROOT), "show", REF + ":research/wakeword_p4/" + name])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), "Fresh output required")
    names = (
        "4090-piper-balanced-training-features-original-v1.json",
        "4090-piper-balanced-training-full-manifest-v1.jsonl",
        "4090-balanced-rendering-factorial-reference-v1.json",
        "4090-balanced-rendering-factorial-controls-v1.json",
    )
    inputs = {name: blob("results/" + name) for name in names}
    pack = json.loads(inputs[names[0]])
    manifest = [json.loads(line) for line in inputs[names[1]].splitlines()]
    reference, controls = [json.loads(inputs[n]) for n in names[2:]]
    require(sha(inputs[names[0]]) == PACK_SHA, "Original feature report differs")
    require(sha(inputs[names[1]]) == MANIFEST_SHA == pack["source_manifest_sha256"],
            "Original source manifest differs")
    require(sha(blob("experiments/balanced_rendering_factorial_v1.py")) == GENERATOR_SHA,
            "Rendering implementation differs")
    for phase, report in (("reference", reference), ("controls", controls)):
        require(report["phase"] == phase and report["verified"] == 16128
                and report["all_exact"] and report["script_sha256"] == GENERATOR_SHA
                and report["source_report_sha256"] == PACK_SHA
                and not report["wake_models_scored"] and not report["final_data_used"],
                "Original phase identity/status differs")
    old, rows = pack["records"], controls["records"]
    require(len(old) == 24192 and len(manifest) == 8064 and len(rows) == 16128,
            "Incomplete source/variant/control records")
    cells, arm_counts, paths = Counter(), Counter(), set()
    symbolic_checks = Counter()
    gains = []
    for i, source in enumerate(manifest):
        quiet, noisy25, combined = old[3 * i:3 * i + 3]
        for variant, original in enumerate((quiet, noisy25, combined)):
            require(original["pack_index"] == 3 * i + variant
                    and original["variant"] == variant
                    and original["source"] == source["files"]["wav"]["path"]
                    and original["source_sha256"] == source["files"]["wav"]["sha256"]
                    and original["source_pcm_sha256"] == source["pcm_sha256"]
                    and original["voice"] == str(source["speaker_id"])
                    and original["target"] == source["target"]
                    and original["length_multiplier"] == source["length_scale"]
                    and original["split"] == source["split"] == "training"
                    and original["source_samples"] == source["samples"],
                    "Original source correspondence differs")
        require(quiet["rir_tail_samples"] == 0 and combined["rir_tail_samples"] == 960
                and combined["signal_samples"] == quiet["signal_samples"] + 960,
                "Original RIR tail differs")
        for offset, arm in enumerate(("noise15", "rir")):
            row = rows[2 * i + offset]
            expected = quiet if arm == "noise15" else combined
            require(row["source_index"] == i and row["arm"] == arm
                    and row["source"] == quiet["source"]
                    and row["source_sha256"] == quiet["source_sha256"]
                    and row["voice"] == quiet["voice"] and row["target"] == quiet["target"]
                    and row["speed"] == quiet["length_multiplier"]
                    and row["noise_seed"] == combined["seed"], "Control identity differs")
            require(row["signal_samples"] == expected["signal_samples"]
                    and row["lead_samples"] == 32000 and row["tail_samples"] == 16000
                    and row["level_gain"] == expected["level_gain"]
                    and math.isfinite(row["level_gain"]) and row["level_gain"] > 0
                    and row["gain_after_mix"] == 1.0, "Control level/geometry differs")
            gains.append(row["gain_after_mix"])
            samples = row["signal_samples"] + 48000
            end = (32000 + row["signal_samples"] + 159) // 160 - 1
            require(end == row["feature_crop_end"] == expected["feature_crop_end"]
                    and (end - 150) * 160 <= 32000
                    and (end + 1) * 160 >= 32000 + row["signal_samples"],
                    "Full process-container coverage differs")
            require(set(row["files"]) == {
                "speech", "signal", "padded_signal", "noise", "wave", "raw", "wav"},
                "Missing/extra artifact role")
            for role, entry in row["files"].items():
                path = PurePosixPath(entry["path"])
                require(not path.is_absolute() and ".." not in path.parts
                        and str(path) not in paths, "Unsafe or duplicate artifact path")
                paths.add(str(path))
                expected_path = f"artifacts/{i}-{arm}"
                expected_path += ".wav" if role == "wav" else f"-{role}.npy"
                require(str(path) == expected_path and len(entry["sha256"]) == 64
                        and all(c in "0123456789abcdef" for c in entry["sha256"]),
                        "Artifact path/hash format differs")
                if role == "wav":
                    size = 44 + 2 * samples
                elif role == "raw":
                    size = 128 + 80 * expected["raw_shape"][0]
                else:
                    size = 128 + 8 * (row["signal_samples"]
                                      if role in ("speech", "signal") else samples)
                require(entry["size"] == size, "Recorded serialization geometry differs")
            files = row["files"]
            if arm == "noise15":
                # Original quiet has no limiting, so the unnoised control's
                # padded float64 array should reference the same array bytes.
                require(quiet["gain_after_mix"] == 1.0
                        and files["padded_signal"]["sha256"] == quiet["float_sha256"],
                        "Dry padded-signal hash correspondence differs")
                symbolic_checks["dry_padded_to_original_quiet_hash"] += 1
            else:
                require(files["padded_signal"]["sha256"] == files["wave"]["sha256"],
                        "No-noise RIR padded/wave hashes differ")
                symbolic_checks["rir_padded_to_final_wave_hash"] += 1
            cells[(row["voice"], row["target"], row["speed"], arm)] += 1
            arm_counts[arm] += 1
    voices = {cell[0] for cell in cells}
    expected_cells = {(voice, phrase, speed, arm) for voice in voices
                      for phrase in ("hey_nyx", "hey_nick", "hey_next", "hey_mix")
                      for speed in (.9, 1., 1.1) for arm in ("noise15", "rir")}
    require(len(voices) == 672 and set(cells) == expected_cells
            and set(cells.values()) == {1}, "Incomplete or repeated full factorial grid")
    result = dict(
        scope=__doc__, gpu_commit=REF,
        input_sha256={name: sha(data) for name, data in inputs.items()},
        generator_sha256=GENERATOR_SHA, source_records=8064,
        original_variants_connected=24192, control_records_connected=len(rows),
        conditioning_ids=len(voices), arm_counts=dict(arm_counts),
        unique_artifact_paths=len(paths), all_grid_cells_once=True,
        all_recorded_full_containers=True, all_recorded_headroom_gains_one=True,
        symbolic_hash_correspondences=dict(symbolic_checks),
        all_checks_pass=True, accepted=False,
        script_sha256=sha(Path(__file__).read_bytes()),
        limits="Original git report/source bytes, full-grid metadata, symbolic hash "
               "correspondences and serialization/length arithmetic only. Artifact "
               "files were not received or independently rehashed/replayed on Mac. "
               "The generator all_exact field is not independent quality proof; GPU "
               "array/PCM/frontend audit remains separate. No model inference, "
               "threshold selection, training, heldout/final or device/audio access. "
               "Original speeds retained. No acoustic/phonetic, acceptance or parity claim.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
