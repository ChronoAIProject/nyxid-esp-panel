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

from batchnorm_policy import apply_policy, snapshot
from hard_ambient import HardAmbient
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
    parser.add_argument("--architecture", choices=("compact", "official64", "teacher60", "teacher120"), default="compact")
    parser.add_argument("--context-frames", type=int, default=FRAMES)
    parser.add_argument("--initial-weights", type=Path)
    parser.add_argument("--freeze-batchnorm", action="store_true")
    parser.add_argument("--batchnorm-policy", choices=("train", "frozen", "affine"))
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--prefix-negatives", type=int, default=0,
                        help="Replace this many of 48 confusing negatives with training-positive prefixes")
    parser.add_argument("--mask-prefix-loss", action="store_true",
                        help="Keep sampled prefixes but give their unverified labels zero training weight")
    parser.add_argument("--phone-binary-pack", type=Path)
    parser.add_argument("--full-source-pack", type=Path)
    parser.add_argument("--full-source-temporal-pack", type=Path)
    parser.add_argument("--five-score-pack", type=Path)
    parser.add_argument("--complete-coverage-pack", type=Path)
    parser.add_argument("--negative-full-timeline", action="store_true")
    parser.add_argument("--positive-full-timeline", action="store_true")
    parser.add_argument("--negative-container-mixture", action="store_true")
    parser.add_argument("--phone-source-balanced", action="store_true")
    parser.add_argument("--phrase-auxiliary-manifest", type=Path)
    parser.add_argument("--phrase-auxiliary-weight", type=float, default=.1)
    parser.add_argument("--hard-ambient-index", type=Path)
    parser.add_argument("--hard-ambient-count", type=int, default=0,
                        help="Replace this many of48 ambient slots with audited training-only real-speech hard windows")
    parser.add_argument("--negative-candidates", type=int, default=1)
    parser.add_argument("--label-smoothing", type=float, default=0.,
                        help="Training BCE only: y*(1-epsilon)+epsilon/2; evaluation labels stay unchanged")
    parser.add_argument("--time-mask-width", type=int, default=0)
    parser.add_argument("--frequency-mask-width", type=int, default=0)
    parser.add_argument("--ambient-sampling", choices=("rows", "duration", "stratified"), default="rows")
    parser.add_argument("--source-exponent", type=float, default=.5,
                        help="Engine sampling: .5=sqrt clip counts, 0=equal engines")
    parser.add_argument("--event-validation", action="store_true",
                        help="Choose checkpoints by latched keyword events, not window peaks")
    args = parser.parse_args()
    if not 0 <= args.label_smoothing <= .2:
        parser.error("label smoothing must be finite and within0..0.2")
    bn_policy = args.batchnorm_policy or ("frozen" if args.freeze_batchnorm else "train")
    if args.freeze_batchnorm and bn_policy != "frozen":
        parser.error("freeze-batchnorm conflicts with the selected policy")
    frames = args.context_frames
    if frames < 150 or frames % 3:
        parser.error("context frames must be >=150 and divisible by 3")
    if args.initial_weights and (args.architecture not in ("teacher60", "teacher120") or frames != 150 or bn_policy not in ("frozen", "affine")):
        parser.error("reconstructed weights require teacher60/teacher120, context150 and fixed batchnorm statistics")
    if not 0 <= args.prefix_negatives <= 48:
        parser.error("prefix negatives must be 0..48")
    if args.mask_prefix_loss and not args.prefix_negatives:
        parser.error("mask-prefix-loss requires positive prefix-negatives")
    if args.phrase_auxiliary_manifest and (args.architecture != 'teacher60' or frames != 150 or bn_policy != 'frozen'
            or args.mask_prefix_loss or args.label_smoothing != 0
            or not 0 < args.phrase_auxiliary_weight <= 1):
        parser.error("phrase auxiliary requires teacher60/context150, frozenBN, original BCE and no prefix masking; weight within (0,1]")
    if args.negative_container_mixture and not (args.negative_full_timeline and args.positive_full_timeline):
        parser.error("negative-container-mixture requires v42 positive and negative full timeline")
    if args.positive_full_timeline and not args.negative_full_timeline:
        parser.error("positive-full-timeline requires negative-full-timeline")
    if args.negative_full_timeline and not args.complete_coverage_pack:
        parser.error("negative-full-timeline requires complete-coverage-pack")
    if args.complete_coverage_pack and (not args.full_source_pack or args.full_source_temporal_pack or args.five_score_pack or args.time_mask_width or args.frequency_mask_width):
        parser.error('complete coverage requires original full-source BCE without sequence/jitter/masking')
    if args.full_source_temporal_pack and not args.full_source_pack:
        parser.error('temporal expansion requires full-source base pack')
    if args.five_score_pack and (not args.full_source_pack or args.full_source_temporal_pack
            or args.time_mask_width or args.frequency_mask_width):
        parser.error('five-score supplement requires original full-source base without temporal jitter or masking')
    if args.full_source_pack and (args.architecture!='teacher60' or frames!=150 or bn_policy!='frozen'
            or args.phone_binary_pack or args.phrase_auxiliary_manifest or args.prefix_negatives
            or args.mask_prefix_loss or args.label_smoothing or args.negative_candidates!=1
            or args.hard_ambient_index):
        parser.error('full source requires teacher60/context150 frozenBN, no extra pack/head/prefix/mining/hard ambient')
    if args.phone_source_balanced and not args.phone_binary_pack:
        parser.error("source balancing requires phone binary pack")
    if args.phone_binary_pack and (args.architecture != 'teacher60' or frames != 150 or bn_policy != 'frozen' or args.phrase_auxiliary_manifest or args.label_smoothing):
        parser.error('phone pack requires teacher60/context150 frozenBN, original BCE, no auxiliary head')
    if not 0 <= args.hard_ambient_count <= 48 or bool(args.hard_ambient_index) != (args.hard_ambient_count > 0):
        parser.error("hard ambient index and count1..48 must be supplied together")
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
    prefix_rng = np.random.default_rng(20726)
    hard_ambient_rng = np.random.default_rng(30726)
    sys.path.insert(0, str(args.upstream.resolve()))
    from microwakeword import mixednet, utils
    from microwakeword.layers import modes

    raw = np.memmap(args.features/"features.bin", mode="r", dtype="<u2").reshape(-1, 40)
    records = json.loads((args.features/"features.json").read_text())
    feature_metadata_hashes = {
        name: hashlib.sha256((args.features/name).read_bytes()).hexdigest()
        for name in ("features.json", "human.json", "ambient.json", "augmentation.json",
                     "negative-additions.json", "base-provenance.json", "dialogue-augmentation.json")
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
    hard_ambient = (HardAmbient(args.hard_ambient_index, args.features, frames)
                    if args.hard_ambient_index else None)

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
    for key, group in ambient_coverage.items():
        hard_fraction = args.hard_ambient_count/48
        group["effective_probability_after_hard_replacement"] = (
            group["effective_probability"]*(1-hard_fraction)
            + (hard_ambient.report["category_probabilities"].get(key, 0.)*hard_fraction
               if hard_ambient is not None else 0.))

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

    def prefix_sample(count):
        sources = groups["training"][1]
        engines = list(sources)
        weights = np.asarray([len(sources[e]) for e in engines], dtype=float)**args.source_exponent
        weights /= weights.sum()
        output = []
        for _ in range(count):
            rows = sources[str(prefix_rng.choice(engines, p=weights))]
            row = rows[int(prefix_rng.integers(len(rows)))]
            # Supervise incomplete training positives, not evaluation utterances.
            # Existing endpoint labels are proxies, not phonetic ground truth.
            end = int(row["keyword_frame"])-int(prefix_rng.integers(15, 36))
            if end < frames or row["split"] != "training" or row["label"] != 1:
                raise ValueError("Invalid prefix crop; never clamp across the negative boundary")
            output.append(sequence(row)[end-frames:end])
        return np.asarray(output)

    full_source = None
    if args.full_source_pack:
        from phone_binary import FullSourceBatch
        full_source = FullSourceBatch(args.full_source_pack, args.full_source_temporal_pack)
        (root/'full_source.json').write_text(json.dumps(full_source.report, indent=2)+'\n')

    complete_coverage = None
    if args.complete_coverage_pack:
        from complete_coverage import CompleteCoverage
        complete_coverage = CompleteCoverage(args.complete_coverage_pack, negative_timeline=args.negative_full_timeline, positive_timeline=args.positive_full_timeline, negative_container_mixture=args.negative_container_mixture)
        (root/'complete_coverage.json').write_text(json.dumps(complete_coverage.report,indent=2)+'\n')

    def batch(mine=True, augment=True):
        noise = []
        for _ in range(48):
            index = (int(rng.choice(len(ambient["training"]), p=ambient_weights))
                     if args.ambient_sampling != "rows" else int(rng.integers(len(ambient["training"]))))
            _, data = ambient["training"][index]
            at = int(rng.integers(0, len(data)-frames+1))
            noise.append(data[at:at+frames])
        x = scale(np.concatenate((sample(1, 32), sample(0, 48, mine=mine), np.asarray(noise))))
        if full_source is not None:
            x[:80] = full_source.first80()
            if complete_coverage is not None:
                x[:80] = complete_coverage.replace(x[:80])
        if args.prefix_negatives:
            # Keep baseline RNG and all original draws unchanged. Only replace
            # the selected confusing-negative slots before the same permutation.
            x[32:32+args.prefix_negatives] = scale(prefix_sample(args.prefix_negatives))
        if hard_ambient is not None and mine:
            # Keep all original draws; independently replace only the ambient slots.
            x[80:80+args.hard_ambient_count] = scale(hard_ambient.sample(args.hard_ambient_count, hard_ambient_rng))
        y = np.array([1]*32+[0]*96, dtype=np.float32)[:, None]
        # A stronger negative penalty follows the upstream always-listening
        # recipe; its operating threshold is calibrated, not assumed to be .97.
        weights = np.array([1.]*32+[20.]*96, dtype=np.float32)
        if args.mask_prefix_loss:
            # Do not replace inputs, advance RNGs, or renormalize other weights.
            # Frozen BN prevents these ignored inputs changing batch statistics.
            weights[32:32+args.prefix_negatives] = 0.
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
    if args.architecture == "teacher120":
        architecture_args += ["--pointwise_filters", "120,120,120,120",
                              "--repeat_in_block", "1,1,1,1",
                              "--mixconv_kernel_sizes", "[5], [9], [13], [21]",
                              "--first_conv_filters", "60", "--first_conv_kernel_size", "5"]
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
        expected_initial_sha = {
            "teacher60": "8e7de8447bb95660844d687c5e44cd7d5c93f3106d08681e2ceda7cd6a28643b",
            "teacher120": "f00bd48d5a7dde94eaac64cd108fd9db68792b0bdcc6af87e1d05a07367f4337",
        }[args.architecture]
        if initial_sha != expected_initial_sha:
            raise ValueError("Unexpected reconstructed initialization weights")
        model.load_weights(args.initial_weights)
        heads[0].set_weights(head_weights)
    bn_layers = apply_policy(model, bn_policy)
    bn_initial = snapshot(model)
    frozen_bn = bn_layers if bn_policy == "frozen" else []
    initialization = dict(context_frames=frames, initial_weights_sha256=initial_sha,
                          classifier_reset=bool(args.initial_weights), classifier_initial_sha256=head_sha,
                          frozen_batchnorm=frozen_bn, batchnorm_policy=bn_policy,
                          batchnorm_initial=bn_initial, seed=726)
    (root/"initialization.json").write_text(json.dumps(initialization, indent=2)+"\n")
    print(json.dumps(initialization), flush=True)
    loss = (tf.keras.losses.BinaryCrossentropy(label_smoothing=args.label_smoothing)
            if args.label_smoothing else "binary_crossentropy")
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3, global_clipnorm=5), loss=loss)
    phone_binary = None
    if args.phone_binary_pack:
        from phone_binary import PhoneBinary
        phone_binary = PhoneBinary(args.phone_binary_pack, source_balanced=args.phone_source_balanced)
        (root/'phone_binary.json').write_text(json.dumps(phone_binary.report, indent=2)+'\n')
    auxiliary = None
    five_score = None
    if args.five_score_pack:
        from five_score_training import FiveScoreTraining
        five_score = FiveScoreTraining(model, args.five_score_pack)
        (root/'five_score.json').write_text(json.dumps(five_score.report,indent=2)+'\n')
    if args.phrase_auxiliary_manifest:
        from phrase_auxiliary import PhraseAuxiliary
        if model.losses:
            raise ValueError('Auxiliary protocol does not support regularization losses')
        auxiliary = PhraseAuxiliary(model, args.features, args.phrase_auxiliary_manifest,
                                    args.phrase_auxiliary_weight)
        (root/'phrase_auxiliary.json').write_text(json.dumps(auxiliary.report, indent=2)+'\n')
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
            if phone_binary is not None:
                x, y, weights = phone_binary.append(x, y, weights)
            if five_score is not None:
                sx,sy,sw=five_score.batch()
                loss=float(five_score.step(x,y,weights,sx,sy,sw))
            elif auxiliary is None:
                loss = float(model.train_on_batch(x, y, sample_weight=weights))
            else:
                auxiliary_x, auxiliary_y = auxiliary.batch()
                loss = float(auxiliary.step(x, y, weights, auxiliary_x, auxiliary_y))
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
                    if auxiliary is not None:
                        auxiliary.network.save_weights(root/'best_auxiliary.weights.h5')
    model.load_weights(root/"best.weights.h5")
    bn_selected = snapshot(model)
    if bn_policy in ("frozen", "affine") and bn_selected["moving_statistics_sha256"] != bn_initial["moving_statistics_sha256"]:
        raise ValueError("Fixed batch normalization statistics changed during training")
    if bn_policy == "affine" and bn_selected["affine_sha256"] == bn_initial["affine_sha256"]:
        raise ValueError("Affine batch normalization policy did not update any affine weights")
    if bn_policy == "frozen" and bn_selected["affine_sha256"] != bn_initial["affine_sha256"]:
        raise ValueError("Frozen batch normalization affine weights changed")
    export_model = model
    if five_score is not None:
        export_model = mixednet.model(flags, (frames,40), None)
        export_model.set_weights(model.get_weights())
    streaming = utils.to_streaming_inference(export_model, {"stride": 3, "spectrogram_length": frames}, modes.Modes.STREAM_INTERNAL_STATE_INFERENCE)
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
    report = {**initialization, "batchnorm_selected": bn_selected, "recipe": "stream-v2", "parameters": model.count_params(), "steps": args.steps,
              "tensorflow": tf.__version__, "gpus": [g.name for g in gpus],
              "gpu_details": [tf.config.experimental.get_device_details(g) for g in gpus],
              "selected_step": selected_step,
              "complete_coverage": complete_coverage.report if complete_coverage is not None else None,
              "five_score": five_score.report if five_score is not None else None,
              "validation_window_phase": window_phase,
              "source_exponent": args.source_exponent,
              "ambient_sampling": args.ambient_sampling, "ambient_sampling_coverage": ambient_coverage,
              "validation_detection": "latched events after full receptive-field warmup" if args.event_validation else "window peak",
              "features_index_sha256": hashlib.sha256((args.features/"features.json").read_bytes()).hexdigest(),
              "feature_metadata_sha256": feature_metadata_hashes,
              "architecture": args.architecture, "architecture_args": architecture_args,
              "phone_binary": phone_binary.report if phone_binary is not None else None,
              "full_source": full_source.report if full_source is not None else None,
              "effective_training_batch_size": 144 if phone_binary is not None else 128,
              "batch_size": 128, "negative_weight": 20, "history": history,
              "label_smoothing": args.label_smoothing,
              "label_smoothing_policy": "Training BCE targets y*(1-epsilon)+epsilon/2; sample weights unchanged; evaluation labels/windows unchanged",
              "negative_candidates": args.negative_candidates, "mining_seed": 4726,
              "mask_prefix_loss": args.mask_prefix_loss,
              "phrase_auxiliary": auxiliary.report if auxiliary is not None else None,
              "mask_prefix_loss_policy": "Zero weight on all sampled prefix slots only; inputs, label placeholders, other weights and RNG draws unchanged; no claim of verified phonetic completeness",
              "hard_ambient": ({**hard_ambient.report, "count_per_batch": args.hard_ambient_count,
                                "seed": 30726, "remaining_broad_ambient_slots": 48-args.hard_ambient_count,
                                "policy": "Audited training-only mined windows; replace ambient slots after original draws; label0 weight20; disabled for INT8 calibration; no new independent exposure."}
                               if hard_ambient is not None else None),
              "prefix_negatives": {"count_per_batch": args.prefix_negatives, "seed": 20726,
                                   "gap_frames_inclusive": [15, 35],
                                   "policy": "Training-positive crops ending 150..350ms before unchanged keyword_frame; replace confusing-negative slots; baseline RNG unchanged; label0 weight20. Endpoint is an energy proxy, not phone timing."},
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
