"""Train the upstream MCU-sized MixedNet on real prepared Hey Nyx features."""
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
from ai_edge_litert.interpreter import Interpreter, OpResolverType


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=4000)
    args = parser.parse_args()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    assert not (root / "training.json").exists(), "Use a fresh training directory"
    tf.config.threading.set_intra_op_parallelism_threads(6)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    tf.keras.utils.set_random_seed(726)
    sys.path.insert(0, str(args.upstream.resolve()))
    from microwakeword import mixednet, utils
    from microwakeword.layers import modes
    from features import FRAMES, background, features

    rng = np.random.default_rng(726)
    human_records = json.loads((args.features / "human.json").read_text())
    human = {split: [np.load(args.features / r["path"], mmap_mode="r") for r in human_records if r["split"] == split]
             for split in ("training", "validation", "testing")}
    synthetic = {split: {label: np.load(args.features / f"{split}-{label}.npy", mmap_mode="r") for label in (0, 1)}
                 for split in ("training", "validation", "testing")}
    noise = []
    for _ in range(256):
        pcm = background(rng, 29120) * rng.uniform(30, 4000)
        noise.append(features(pcm))
    noise = np.array(noise)

    def sample_human(split, count, random):
        arrays = human[split]
        lengths = np.array([len(a) for a in arrays])
        selected = random.choice(len(arrays), size=count, p=lengths/lengths.sum())
        output = []
        for i in selected:
            at = random.integers(200, len(arrays[i])-FRAMES)
            output.append(arrays[i][at:at+FRAMES])
        return np.asarray(output)

    def scale(raw):
        return np.clip(raw.astype(np.float32)/25.6, 0, 26)

    def batch():
        pos = synthetic["training"][1]
        neg = synthetic["training"][0]
        raw = np.concatenate((pos[rng.integers(len(pos),size=28)],
                              neg[rng.integers(len(neg),size=18)],
                              sample_human("training", 16, rng), noise[rng.integers(len(noise),size=2)]))
        x = scale(raw)
        # Light SpecAugment supplements waveform-domain gain/noise/room variants.
        for row in range(len(x)):
            if rng.random() < .35:
                start = rng.integers(0, 37); x[row, :, start:start+rng.integers(1,4)] = 0
            if rng.random() < .25:
                start = rng.integers(0, 175); x[row, start:start+rng.integers(1,6), :] = 0
        y = np.array([1]*28+[0]*36,dtype=np.float32)[:,None]
        weight = np.array([1.]*46+[1.5]*16+[.5]*2,dtype=np.float32)
        order = rng.permutation(64)
        return x[order], y[order], weight[order]

    val_x = scale(np.concatenate((synthetic["validation"][1], synthetic["validation"][0],
                                  sample_human("validation", 2000, np.random.default_rng(512)))))
    val_y = np.array([1]*len(synthetic["validation"][1]) + [0]*(len(val_x)-len(synthetic["validation"][1])),dtype=np.float32)
    architecture = argparse.ArgumentParser()
    mixednet.model_parameters(architecture)
    flags = architecture.parse_args(["--residual_connection", "0,0,0,0", "--stride", "3"])
    model = mixednet.model(flags, (FRAMES, 40), None)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3,global_clipnorm=5),loss="binary_crossentropy")
    best = float("inf")
    history = []
    started = time.monotonic()
    with (root / "progress.jsonl").open("w") as log:
        for step in range(1, args.steps+1):
            if step == int(args.steps*.5):model.optimizer.learning_rate.assign(3e-4)
            if step == int(args.steps*.8):model.optimizer.learning_rate.assign(1e-4)
            x,y,w = batch()
            loss = float(model.train_on_batch(x,y,sample_weight=w))
            if step % 100 == 0:
                item = {"step":step,"training_loss":loss,"elapsed_seconds":round(time.monotonic()-started,1)}
                print(json.dumps(item),flush=True);log.write(json.dumps(item)+"\n");log.flush()
            if step % 500 == 0 or step == args.steps:
                scores = model.predict(val_x,batch_size=128,verbose=0).reshape(-1)
                clipped = np.clip(scores,1e-7,1-1e-7)
                bce = float(np.mean(-(val_y*np.log(clipped)+(1-val_y)*np.log(1-clipped))))
                metrics = {"step":step,"validation_loss":bce,
                           "positive_recall_at_097":float(np.mean(scores[val_y==1]>=.97)),
                           "negative_window_rate_at_097":float(np.mean(scores[val_y==0]>=.97))}
                history.append(metrics);print(json.dumps(metrics),flush=True)
                if bce < best:
                    best=bce;model.save_weights(root/"best.weights.h5")
    model.load_weights(root/"best.weights.h5")
    streaming = utils.to_streaming_inference(model, {"stride":3,"spectrogram_length":FRAMES},modes.Modes.STREAM_INTERNAL_STATE_INFERENCE)
    streaming(np.zeros((1,3,40),dtype=np.float32))
    streaming.export(str(root/"saved_model"))

    def representative():
        # Fix the full intended input range to 0..26. Training and inference
        # both clip to this range; quantization parameters remain in metadata.
        yield [np.zeros((1,3,40),dtype=np.float32)]
        yield [np.full((1,3,40),26,dtype=np.float32)]
        for _ in range(300):
            x,_,_=batch()
            for at in range(0,FRAMES,3):
                yield [x[:1,at:at+3]]
    converter=tf.lite.TFLiteConverter.from_saved_model(str(root/"saved_model"))
    converter.optimizations=[tf.lite.Optimize.DEFAULT]
    converter._experimental_variable_quantization=True
    converter.target_spec.supported_ops=[tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type=tf.int8
    converter.inference_output_type=tf.uint8
    converter.representative_dataset=representative
    data=converter.convert()
    path=root/"hey_nyx.tflite";path.write_bytes(data)
    interpreter=Interpreter(model_content=data,experimental_op_resolver_type=OpResolverType.BUILTIN_REF)
    interpreter.allocate_tensors()
    inp,out=interpreter.get_input_details()[0],interpreter.get_output_details()[0]
    assert inp['shape'].tolist()==[1,3,40] and inp['dtype']==np.int8 and out['dtype']==np.uint8
    report={"phrase":"Hey Nyx","architecture":"microWakeWord MixedNet","upstream_revision":"4665173cd35f1cff9a61e06fc427f124766c488e",
            "parameters":model.count_params(),"steps":args.steps,"batch_size":64,"training_seconds":round(time.monotonic()-started,1),
            "history":history,"best_validation_loss":best,"model_bytes":len(data),"model_sha256":hashlib.sha256(data).hexdigest(),
            "input_shape":inp['shape'].tolist(),"input_quantization":list(inp['quantization']),"output_quantization":list(out['quantization']),
            "frontend":{"sample_rate":16000,"window_ms":30,"hop_ms":10,"channels":40,"raw_divisor":25.6,"clip":[0,26]},
            "scope":"First trained candidate. Threshold and continuous held-out evaluation still required."}
    (root/"training.json").write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)


if __name__=="__main__":main()
