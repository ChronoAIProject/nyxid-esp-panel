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

from features import FRAMES
from evaluate import events, smooth, THRESHOLDS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=12000)
    parser.add_argument("--eval-interval", type=int, default=3000)
    parser.add_argument("--architecture", choices=("compact", "official64", "wide128", "transfer60"), default="compact")
    parser.add_argument("--pretrained-model", type=Path,
                        help="Reviewed ESPHome v2 encoder for transfer60; final classifier is reset")
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--initial-weights", type=Path,
                        help="Warm-start matching float weights in a fresh run; optimizer state is reset")
    parser.add_argument("--export-only", action="store_true",
                        help="Export matching --initial-weights without optimization; requires --steps 0")
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--input-frames", type=int, default=FRAMES)
    parser.add_argument("--negative-weight", type=float, default=20.)
    parser.add_argument("--source-sampling", choices=("sqrt", "uniform"), default="sqrt",
                        help="Choose TTS engines by square-root corpus size or equally; sample clips within each engine")
    parser.add_argument("--hard-negative-fraction", type=float, default=0.,
                        help="Experimental fraction of each negative source batch selected by current-model score; training split only")
    parser.add_argument("--hard-negative-pool", type=int, default=4,
                        help="Candidate pool multiplier when hard-negative mining is enabled")
    parser.add_argument("--hard-negative-start-step", type=int, default=1000)
    parser.add_argument("--specaugment", action="store_true")
    parser.add_argument("--pooling", choices=("none", "max", "average"), default="none")
    parser.add_argument("--align-stream", action="store_true",
                        help="Align float validation windows to the streaming convolution phase")
    args = parser.parse_args()
    if args.export_only:
        if not args.initial_weights or args.steps != 0:
            parser.error("--export-only requires --initial-weights and --steps 0")
    elif args.steps < 1:
        parser.error("Training requires positive --steps; use --export-only for a saved checkpoint")
    frames = args.input_frames
    minimum_frames = 149 if args.architecture == "transfer60" else 180
    if frames < minimum_frames or args.negative_weight <= 0 or args.learning_rate <= 0:
        parser.error("Input context is too short or weight/learning rate is not positive")
    if args.pretrained_model and args.architecture != "transfer60":
        parser.error("--pretrained-model requires --architecture transfer60")
    if args.pretrained_model and args.initial_weights:
        parser.error("Choose a pretrained encoder or learned initial weights, not both")
    if not 0 <= args.hard_negative_fraction <= 1 or args.hard_negative_pool < 1 or args.hard_negative_start_step < 0:
        parser.error("Invalid hard-negative mining settings")
    initial_weights_hash = hashlib.sha256(args.initial_weights.read_bytes()).hexdigest() if args.initial_weights else None
    source_hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in Path(__file__).parent.glob("*.py")}
    root = args.output
    root.mkdir(parents=True, exist_ok=True)
    assert not (root/"training.json").exists(), "Use a fresh output directory"
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    tf.keras.utils.set_random_seed(726)
    rng = np.random.default_rng(726)
    sys.path.insert(0, str(args.upstream.resolve()))
    from microwakeword import mixednet, utils
    from microwakeword.layers import modes

    raw = np.memmap(args.features/"features.bin", mode="r", dtype="<u2").reshape(-1, 40)
    records = json.loads((args.features/"features.json").read_text())
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

    def scale(data):
        return np.clip(np.asarray(data, dtype=np.float32)/25.6, 0, 26)

    def sequence(row):
        return raw[row["offset"]:row["offset"]+row["frames"]]

    def sample(label, count):
        sources = groups["training"][label]
        engines = list(sources)
        probabilities = (np.sqrt([len(sources[e]) for e in engines])
                         if args.source_sampling == "sqrt" else np.ones(len(engines)))
        probabilities /= probabilities.sum()
        output = []
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
                end = int(rng.integers(lo, len(data)+1))
            end = min(len(data), max(frames, end))
            output.append(data[end-frames:end])
        return np.asarray(output)

    current_step = 0

    def negative_sample(data, count):
        if len(data) == count:
            return data
        values = np.asarray(infer(scale(data))).reshape(-1)
        hard_count = int(round(count*args.hard_negative_fraction))
        # Keep source types balanced and retain ordinary random windows.
        # Both pools contain training data only. No PCM speed or labels change.
        hard = np.argsort(values, kind="stable")[-hard_count:] if hard_count else np.array([], dtype=int)
        remaining = np.setdiff1d(np.arange(len(data)), hard)
        random = rng.choice(remaining, count-hard_count, replace=False)
        return data[np.concatenate((hard, random))]

    def batch(mine=True):
        multiplier = (args.hard_negative_pool
                      if mine and args.hard_negative_fraction and current_step >= args.hard_negative_start_step else 1)
        noise = []
        for _ in range(48*multiplier):
            _, data = ambient["training"][int(rng.integers(len(ambient["training"])))]
            at = int(rng.integers(0, len(data)-frames+1))
            noise.append(data[at:at+frames])
        x = scale(np.concatenate((sample(1, 32), negative_sample(sample(0, 48*multiplier), 48),
                                  negative_sample(np.asarray(noise), 48))))
        y = np.array([1]*32+[0]*96, dtype=np.float32)[:, None]
        # A stronger negative penalty follows the upstream always-listening
        # recipe; its operating threshold is calibrated, not assumed to be .97.
        weights = np.array([1.]*32+[args.negative_weight]*96, dtype=np.float32)
        if args.specaugment:
            # Feature-domain masking changes neither PCM speed nor transport.
            for item in x:
                if rng.random() < .5:
                    width = int(rng.integers(1, 5))
                    at = int(rng.integers(0, 41-width))
                    item[:, at:at+width] = 0
                if rng.random() < .3:
                    width = int(rng.integers(1, 4))
                    at = int(rng.integers(0, frames-width+1))
                    item[at:at+width] = 0
        order = rng.permutation(len(x))
        return x[order], y[order], weights[order]

    architecture = argparse.ArgumentParser()
    mixednet.model_parameters(architecture)
    architecture_args = ["--residual_connection", "0,0,0,0", "--stride", "3"]
    if args.architecture in ("official64", "wide128"):
        filters = "64,64,64,64" if args.architecture == "official64" else "128,128,128,128"
        architecture_args += ["--pointwise_filters", filters,
                              "--repeat_in_block", "1,1,1,1",
                              "--mixconv_kernel_sizes", "[5], [7,11], [9,15], [23]",
                              "--first_conv_filters", "32", "--first_conv_kernel_size", "5"]
    if args.pooling != "none":
        architecture_args += ["--pooled", "1", "--max_pool", "1" if args.pooling == "max" else "0"]
    if args.architecture == "transfer60":
        if args.pooling != "none":
            parser.error("transfer60 uses the reviewed source topology without pooling")
        architecture_args += ["--pointwise_filters", "60,60,60,60", "--first_conv_filters", "30",
                              "--first_conv_kernel_size", "5"]
    flags = architecture.parse_args(architecture_args)
    transfer_metadata = None
    if args.architecture == "transfer60":
        from transfer import build_model
        model, transfer_metadata = build_model(frames, args.pretrained_model)
    else:
        model = mixednet.model(flags, (frames, 40), None)
    if args.initial_weights:
        model.load_weights(args.initial_weights)
    model.compile(optimizer=tf.keras.optimizers.Adam(args.learning_rate, global_clipnorm=5), loss="binary_crossentropy")

    @tf.function(input_signature=[tf.TensorSpec([None, frames, 40], tf.float32)])
    def infer(x):
        return model(x, training=False)

    def score(data):
        phase = validation_phase
        windows = np.lib.stride_tricks.sliding_window_view(data, frames, axis=0)[phase::3].transpose(0, 2, 1)
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
                end_frames = validation_phase + effective_frames if args.align_stream else frames
                times = (end_frames+2+3*(np.arange(len(values))+4))*.01
                end = (row["keyword_frame"]+2)*.01
                valid = (times >= end-.10) & (times <= end+.80)
                positives.append((row["engine"], float(values[valid].max()) if valid.any() else 0.))
            else:
                negatives.append((row["engine"], float(values.max()) if len(values) else 0.))
        human = [(len(data)*.01/3600, score(data)) for _, data in ambient["validation"]]
        hours = sum(h for h, _ in human)
        grid = []
        for threshold in THRESHOLDS:
            count = sum(events(s, threshold) for _, s in human)
            recalls = {e: float(np.mean([v >= threshold for source, v in positives if source == e]))
                       for e in {source for source, _ in positives}}
            false_rates = {e: float(np.mean([v >= threshold for source, v in negatives if source == e]))
                           for e in {source for source, _ in negatives}}
            grid.append({"threshold": threshold, "recall": float(np.mean([v >= threshold for _, v in positives])),
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

    effective_frames = (frames-flags.first_conv_kernel_size)//flags.stride*flags.stride+flags.first_conv_kernel_size
    validation_phase = (-effective_frames)%flags.stride if args.align_stream else 0
    history = []
    best = None
    started = time.monotonic()
    with (root/"progress.jsonl").open("w") as log:
        for step in range(1, args.steps+1):
            current_step = step
            if step == int(args.steps*.5): model.optimizer.learning_rate.assign(args.learning_rate*.3)
            if step == int(args.steps*.8): model.optimizer.learning_rate.assign(args.learning_rate*.1)
            x, y, weights = batch()
            loss = float(model.train_on_batch(x, y, sample_weight=weights))
            if step % 500 == 0:
                print(json.dumps({"step": step, "loss": loss, "seconds": round(time.monotonic()-started, 1)}), flush=True)
            if step % args.eval_interval == 0 or step == args.steps:
                selected, rank, grid = validate()
                item = {"step": step, "selected": selected, "grid": grid,
                        "checkpoint": f"step-{step}.weights.h5",
                        "seconds": round(time.monotonic()-started, 1)}
                # Keep the measured checkpoints. A later audit of selection
                # criteria must not require recreating discarded weights.
                model.save_weights(root/item["checkpoint"])
                history.append(item)
                log.write(json.dumps(item)+"\n"); log.flush()
                print(json.dumps({"step": step, **selected, "seconds": item["seconds"]}), flush=True)
                if best is None or rank > best:
                    best = rank
                    model.save_weights(root/"best.weights.h5")
    if args.export_only:
        model.save_weights(root/"best.weights.h5")
    else:
        model.load_weights(root/"best.weights.h5")
    streaming = utils.to_streaming_inference(model, {"stride": 3, "spectrogram_length": frames}, modes.Modes.STREAM_INTERNAL_STATE_INFERENCE)
    streaming(np.zeros((1, 3, 40), dtype=np.float32))
    streaming.export(str(root/"saved_model"))

    def representative():
        yield [np.zeros((1, 3, 40), dtype=np.float32)]
        yield [np.full((1, 3, 40), 26, dtype=np.float32)]
        for _ in range(100):
            # Quantization calibration keeps the ordinary training distribution,
            # independent of the optional optimization-time mining policy.
            x, _, _ = batch(mine=False)
            for at in range(0, frames-2, 3):
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
    report = {"recipe": "stream-v2", "parameters": model.count_params(), "steps": args.steps,
              "architecture": args.architecture, "architecture_args": architecture_args,
              "input_frames": frames, "negative_weight": args.negative_weight,
              "source_sampling": args.source_sampling,
              "hard_negative_mining": {"fraction": args.hard_negative_fraction,
                                       "pool_multiplier": args.hard_negative_pool,
                                       "start_step": args.hard_negative_start_step,
                                       "scope": "Training split only; separately balanced TTS and ambient pools"},
              "specaugment": args.specaugment, "align_stream": args.align_stream,
              "pooling": args.pooling,
              "initial_learning_rate": args.learning_rate, "transfer": transfer_metadata,
              "initial_weights_sha256": initial_weights_hash,
              "checkpoint_export_only": args.export_only,
              "batch_size": 128, "history": history,
              "seconds": round(time.monotonic()-started, 1), "model_bytes": len(data),
              "model_sha256": hashlib.sha256(data).hexdigest(),
              "source_files_sha256": source_hashes,
              "sources": {label: {e: len(rs) for e, rs in groups['training'][label].items()} for label in (0, 1)},
              "selection": ("Explicit saved checkpoint; no checkpoint selection or optimization in this export. Fresh seed726 training-only representative sampling; recalibrate the new INT8 hash on validation."
                            if args.export_only else
                            "Float continuous validation <=0.5 ambient events/h and <=1% confusing clips per source; then maximize mean source recall. Quantized validation must be recalibrated."),
              "scope": "Research candidate. Same v1 evaluation sets are development regressions; real-user acceptance remains unmeasured."}
    (root/"training.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps({k: v for k, v in report.items() if k != "history"}), flush=True)


if __name__ == "__main__":
    main()
