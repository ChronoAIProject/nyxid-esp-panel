"""Train V2 on variable alignments; select by continuous validation behavior."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import numpy as np
import tensorflow as tf

from ambient_sampling import probabilities, coverage, stratified_probabilities
from negative_sampling import candidate_ends
from spec_augment import mask_spectrograms
from features import FRAMES
from evaluate import events, event_indices, smooth, THRESHOLDS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=12000)
    parser.add_argument("--eval-interval", type=int, default=3000)
    parser.add_argument("--architecture", choices=("compact", "official64", "teacher60"), default="compact")
    parser.add_argument("--context-frames", type=int, default=FRAMES)
    parser.add_argument("--initial-weights", type=Path)
    parser.add_argument("--freeze-batchnorm", action="store_true")
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--negative-candidates", type=int, default=1)
    parser.add_argument("--time-mask-width", type=int, default=0)
    parser.add_argument("--frequency-mask-width", type=int, default=0)
    parser.add_argument("--ambient-sampling", choices=("rows", "duration", "stratified"), default="rows")
    parser.add_argument("--source-exponent", type=float, default=.5,
                        help="Engine sampling: .5=sqrt clip counts, 0=equal engines")
    parser.add_argument("--event-validation", action="store_true",
                        help="Choose checkpoints by latched keyword events, not window peaks")
    args = parser.parse_args()
    frames = args.context_frames
    if frames < 150 or frames % 3:
        parser.error("context frames must be >=150 and divisible by 3")
    if args.initial_weights and (args.architecture != "teacher60" or frames != 150 or not args.freeze_batchnorm):
        parser.error("reconstructed weights require teacher60, context150 and frozen batchnorm")
    if args.negative_candidates < 1:
        parser.error("negative candidates must be positive")
    if not 0 <= args.time_mask_width <= frames or not 0 <= args.frequency_mask_width <= 40:
        parser.error("mask widths must fit the training spectrogram")
    if args.device == "cpu":
        tf.config.set_visible_devices([], "GPU")
    gpus = tf.config.get_visible_devices("GPU")
    if args.device == "cuda" and not gpus:
        raise RuntimeError("CUDA training requested but TensorFlow found no GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print(json.dumps({"tensorflow": tf.__version__, "gpus": [g.name for g in gpus]}), flush=True)
    source_hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in Path(__file__).parent.glob("*.py")}
    root = args.output
    if root.exists() and any(root.iterdir()):
        raise ValueError("Use a fresh output directory; preserve partial/failed runs")
    root.mkdir(parents=True, exist_ok=True)
    assert not (root/"training.json").exists(), "Use a fresh output directory"
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    tf.keras.utils.set_random_seed(726)
    rng = np.random.default_rng(726)
    mining_rng = np.random.default_rng(4726)
    augmentation_rng = np.random.default_rng(10726)
    sys.path.insert(0, str(args.upstream.resolve()))
    from microwakeword import mixednet, utils
    from microwakeword.layers import modes

    raw = np.memmap(args.features/"features.bin", mode="r", dtype="<u2").reshape(-1, 40)
    records = json.loads((args.features/"features.json").read_text())
    feature_metadata_hashes = {
        name: hashlib.sha256((args.features/name).read_bytes()).hexdigest()
        for name in ("features.json", "human.json", "ambient.json", "augmentation.json",
                     "negative-additions.json", "base-provenance.json")
        if (args.features/name).is_file()}
    groups = {}
    for split in ("training", "validation", "testing"):
        groups[split] = {}
        for label in (0, 1):
            groups[split][label] = {}
            for row in records:
                if row["split"] == split and row["label"] == label:
                    groups[split][label].setdefault(row["engine"], []).append(row)
    ambient_rows = json.loads((args.features/"human.json").read_text())
    ambient_rows += json.loads((args.features/"ambient.json").read_text())
    ambient = {s: [(r, np.load(args.features/r["path"], mmap_mode="r")) for r in ambient_rows if r["split"] == s]
               for s in groups}

    ambient_weights = probabilities([r for r, _ in ambient["training"]])
    ambient_coverage = coverage([r for r, _ in ambient["training"]])
    if args.ambient_sampling == "stratified":
        ambient_weights = stratified_probabilities([r for r, _ in ambient["training"]])
    effective_weights = (np.full(len(ambient_weights), 1/len(ambient_weights))
                         if args.ambient_sampling == "rows" else ambient_weights)
    for group in ambient_coverage.values():
        group["effective_probability"] = 0.
    for (row, _), weight in zip(ambient["training"], effective_weights):
        key = ('human' if 'speaker' in row or row.get('kind') == 'human' else 'ambient')+':'+str(row.get('category', 'unspecified'))
        ambient_coverage[key]["effective_probability"] += float(weight)

    def scale(data):
        return np.clip(np.asarray(data, dtype=np.float32)/25.6, 0, 26)

    def sequence(row):
        return raw[row["offset"]:row["offset"]+row["frames"]]

    def sample(label, count, mine=True):
        sources = groups["training"][label]
        engines = list(sources)
        probabilities = np.asarray([len(sources[e]) for e in engines], dtype=float)**args.source_exponent
        probabilities /= probabilities.sum()
        output = []
        candidate_count = args.negative_candidates if not label and mine else 1
        for _ in range(count):
            rows = sources[str(rng.choice(engines, p=probabilities))]
            row = rows[int(rng.integers(len(rows)))]
            data = sequence(row)
            if label:
                end = int(row["keyword_frame"]+rng.integers(3, 28))
            else:
                # Cover prefixes, full confusing phrases, suffixes and quiet
                # boundaries, rather than only one phrase-end crop.
                lo = max(frames, row["lead_frame"]-25)
                ends = candidate_ends(rng, mining_rng, lo, len(data)+1, candidate_count)
                end = int(ends[0])
            end = min(len(data), max(frames, end))
            if candidate_count > 1:
                output.append(np.asarray([data[int(e)-frames:int(e)] for e in ends]))
            else:
                output.append(data[end-frames:end])
        output = np.asarray(output)
        if candidate_count > 1:
            # Inference mode: no gradients, dropout or batch-statistic updates.
            scores = np.asarray(infer(scale(output.reshape(-1, frames, 40)))).reshape(count, candidate_count)
            output = output[np.arange(count), scores.argmax(axis=1)]
        return output

    def batch(mine=True, augment=True):
        noise = []
        for _ in range(48):
            index = (int(rng.choice(len(ambient["training"]), p=ambient_weights))
                     if args.ambient_sampling != "rows" else int(rng.integers(len(ambient["training"]))))
            _, data = ambient["training"][index]
            at = int(rng.integers(0, len(data)-frames+1))
            noise.append(data[at:at+frames])
        x = scale(np.concatenate((sample(1, 32), sample(0, 48, mine=mine), np.asarray(noise))))
        y = np.array([1]*32+[0]*96, dtype=np.float32)[:, None]
        # A stronger negative penalty follows the upstream always-listening
        # recipe; its operating threshold is calibrated, not assumed to be .97.
        weights = np.array([1.]*32+[20.]*96, dtype=np.float32)
        order = rng.permutation(len(x))
        x = x[order]
        if augment:
            x = mask_spectrograms(x, augmentation_rng, args.time_mask_width,
                                  args.frequency_mask_width)
        return x, y[order], weights[order]

    architecture = argparse.ArgumentParser()
    mixednet.model_parameters(architecture)
    architecture_args = ["--residual_connection", "0,0,0,0", "--stride", "3"]
    if args.architecture == "official64":
        architecture_args += ["--pointwise_filters", "64,64,64,64",
                              "--repeat_in_block", "1,1,1,1",
                              "--mixconv_kernel_sizes", "[5], [7,11], [9,15], [23]",
                              "--first_conv_filters", "32", "--first_conv_kernel_size", "5"]
    if args.architecture == "teacher60":
        architecture_args += ["--pointwise_filters", "60,60,60,60",
                              "--repeat_in_block", "1,1,1,1",
                              "--mixconv_kernel_sizes", "[5], [9], [13], [21]",
                              "--first_conv_filters", "30", "--first_conv_kernel_size", "5"]
    flags = architecture.parse_args(architecture_args)
    model = mixednet.model(flags, (frames, 40), None)
    heads = [getattr(layer, "cell", layer) for layer in model.layers
             if isinstance(getattr(layer, "cell", layer), tf.keras.layers.Dense)]
    if len(heads) != 1:
        raise ValueError("Expected exactly one classifier head")
    head_weights = [w.copy() for w in heads[0].get_weights()]
    head_sha = hashlib.sha256(b"".join(w.tobytes() for w in head_weights)).hexdigest()
    initial_sha = None
    if args.initial_weights:
        initial_sha = hashlib.sha256(args.initial_weights.read_bytes()).hexdigest()
        if initial_sha != "8e7de8447bb95660844d687c5e44cd7d5c93f3106d08681e2ceda7cd6a28643b":
            raise ValueError("Unexpected reconstructed initialization weights")
        model.load_weights(args.initial_weights)
        heads[0].set_weights(head_weights)
    frozen_bn = []
    if args.freeze_batchnorm:
        for layer in model.layers:
            if isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.trainable = False
                frozen_bn.append(layer.name)
        if not frozen_bn:
            raise ValueError("No batch normalization layers found")
    initialization = dict(context_frames=frames, initial_weights_sha256=initial_sha,
                          classifier_reset=bool(args.initial_weights), classifier_initial_sha256=head_sha,
                          frozen_batchnorm=frozen_bn, seed=726)
    (root/"initialization.json").write_text(json.dumps(initialization, indent=2)+"\n")
    print(json.dumps(initialization), flush=True)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3, global_clipnorm=5), loss="binary_crossentropy")
    # Valid stride-3, kernel-5 convolution consumes only 179 of 180
    # input frames. Phase 0 windows therefore end one 10 ms frame before
    # the streaming output they used to be compared with. Shift by the
    # unused suffix to align exact causal history (verified on real speech).
    window_phase = ((frames-flags.first_conv_kernel_size) % flags.stride
                    if flags.first_conv_filters > 0 else 0)

    @tf.function(input_signature=[tf.TensorSpec([None, frames, 40], tf.float32)])
    def infer(x):
        return model(x, training=False)

    def score(data):
        windows = np.lib.stride_tricks.sliding_window_view(data, frames, axis=0)[window_phase::3].transpose(0, 2, 1)
        outputs = []
        for at in range(0, len(windows), 512):
            outputs.append(np.asarray(infer(scale(windows[at:at+512]))).reshape(-1))
        return smooth(np.concatenate(outputs)) if outputs else np.array([])

    def validate():
        positives, negatives = [], []
        for row in records:
            if row["split"] != "validation":
                continue
            values = score(sequence(row))
            if row["label"]:
                times = (frames+2+3*(np.arange(len(values))+4))*.01
                end = (row["keyword_frame"]+2)*.01
                valid = (times >= end-.10) & (times <= end+.80)
                positives.append((row["engine"], float(values[valid].max()) if valid.any() else 0.,
                                  values, times, end))
            else:
                negatives.append((row["engine"], float(values.max()) if len(values) else 0.))
        human = [(r.get("source_seconds", r["seconds"])/3600, score(data)) for r, data in ambient["validation"]]
        hours = sum(h for h, _ in human)
        grid = []
        for threshold in THRESHOLDS:
            count = sum(events(s, threshold) for _, s in human)
            def hit(row):
                if not args.event_validation:
                    return row[1] >= threshold
                return any(row[4]-.1 <= row[3][i] <= row[4]+.8 for i in event_indices(row[2], threshold))
            recalls = {e: float(np.mean([hit(r) for r in positives if r[0] == e]))
                       for e in {r[0] for r in positives}}
            false_rates = {e: float(np.mean([v >= threshold for source, v in negatives if source == e]))
                           for e in {source for source, _ in negatives}}
            grid.append({"threshold": threshold, "recall": float(np.mean([hit(r) for r in positives])),
                         "macro_recall": float(np.mean(list(recalls.values()))), "recall_by_engine": recalls,
                         "hard_negative_false_rate": float(np.mean([v >= threshold for _, v in negatives])),
                         "max_source_false_rate": max(false_rates.values()),
                         "ambient_hours": hours, "ambient_events": count, "ambient_faph": count/hours})
        eligible = [r for r in grid if r["ambient_faph"] <= .5 and r["max_source_false_rate"] <= .01]
        if eligible:
            selected = max(eligible, key=lambda r: (r["macro_recall"], -r["ambient_faph"], -r["max_source_false_rate"]))
            rank = (1, selected["macro_recall"], -selected["ambient_faph"], -selected["max_source_false_rate"])
        else:
            selected = min(grid, key=lambda r: (r["ambient_faph"], r["max_source_false_rate"], -r["macro_recall"]))
            rank = (0, -selected["ambient_faph"], -selected["max_source_false_rate"], selected["macro_recall"])
        return selected, rank, grid

    history = []
    best = None
    selected_step = None
    started = time.monotonic()
    with (root/"progress.jsonl").open("w") as log:
        for step in range(1, args.steps+1):
            if step == int(args.steps*.5): model.optimizer.learning_rate.assign(3e-4)
            if step == int(args.steps*.8): model.optimizer.learning_rate.assign(1e-4)
            x, y, weights = batch()
            loss = float(model.train_on_batch(x, y, sample_weight=weights))
            if step % 500 == 0:
                print(json.dumps({"step": step, "loss": loss, "seconds": round(time.monotonic()-started, 1)}), flush=True)
            if step % args.eval_interval == 0 or step == args.steps:
                selected, rank, grid = validate()
                item = {"step": step, "selected": selected, "grid": grid,
                        "seconds": round(time.monotonic()-started, 1)}
                history.append(item)
                log.write(json.dumps(item)+"\n"); log.flush()
                print(json.dumps({"step": step, **selected, "seconds": item["seconds"]}), flush=True)
                if best is None or rank > best:
                    best = rank
                    selected_step = step
                    model.save_weights(root/"best.weights.h5")
    model.load_weights(root/"best.weights.h5")
    streaming = utils.to_streaming_inference(model, {"stride": 3, "spectrogram_length": frames}, modes.Modes.STREAM_INTERNAL_STATE_INFERENCE)
    streaming(np.zeros((1, 3, 40), dtype=np.float32))
    streaming.export(str(root/"saved_model"))

    def representative():
        yield [np.zeros((1, 3, 40), dtype=np.float32)]
        yield [np.full((1, 3, 40), 26, dtype=np.float32)]
        for _ in range(100):
            x, _, _ = batch(mine=False, augment=False)
            for at in range(0, frames, 3):
                yield [x[:1, at:at+3]]
    converter = tf.lite.TFLiteConverter.from_saved_model(str(root/"saved_model"))
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter._experimental_variable_quantization = True
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.uint8
    converter.representative_dataset = representative
    data = converter.convert()
    (root/"hey_nyx.tflite").write_bytes(data)
    report = {**initialization, "recipe": "stream-v2", "parameters": model.count_params(), "steps": args.steps,
              "tensorflow": tf.__version__, "gpus": [g.name for g in gpus],
              "gpu_details": [tf.config.experimental.get_device_details(g) for g in gpus],
              "selected_step": selected_step,
              "validation_window_phase": window_phase,
              "source_exponent": args.source_exponent,
              "ambient_sampling": args.ambient_sampling, "ambient_sampling_coverage": ambient_coverage,
              "validation_detection": "latched events after full receptive-field warmup" if args.event_validation else "window peak",
              "features_index_sha256": hashlib.sha256((args.features/"features.json").read_bytes()).hexdigest(),
              "feature_metadata_sha256": feature_metadata_hashes,
              "architecture": args.architecture, "architecture_args": architecture_args,
              "batch_size": 128, "negative_weight": 20, "history": history,
              "negative_candidates": args.negative_candidates, "mining_seed": 4726,
              "spec_augment": {"time_mask_width": args.time_mask_width,
                               "frequency_mask_width": args.frequency_mask_width,
                               "seed": 10726, "fill": 0,
                               "policy": "One mask per enabled axis per training sample; width uniform inclusive 0..maximum; independent RNG; disabled for validation and INT8 calibration"},
              "mining_policy": "Training-only highest-scoring window per sampled confusing-negative row; baseline RNG preserved; disabled for INT8 calibration",
              "seconds": round(time.monotonic()-started, 1), "model_bytes": len(data),
              "model_sha256": hashlib.sha256(data).hexdigest(),
              "source_files_sha256": source_hashes,
              "sources": {label: {e: len(rs) for e, rs in groups['training'][label].items()} for label in (0, 1)},
              "selection": "Float continuous validation <=0.5 ambient events/h and <=1% confusing clips per source; then maximize mean source recall. Quantized validation must be recalibrated.",
              "scope": "Research candidate. Same v1 evaluation sets are development regressions; real-user acceptance remains unmeasured."}
    (root/"training.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps({k: v for k, v in report.items() if k != "history"}), flush=True)


if __name__ == "__main__":
    main()
