"""Convert the trained 960 ms word classifier to a fixed INT8 TFLite graph."""
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TF_NUM_INTRAOP_THREADS"] = "2"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"
import numpy as np
import tensorflow as tf

from fit_c161_history_gain_v1 import REF_KERNEL, RESULTS, read, save, sha
from train_c161_word_window_v1 import STEM, WIDTH


def main():
    fit = read(RESULTS/f"{STEM}-training-v1.json")
    weights = RESULTS/f"{STEM}-weights-v1.npz"
    calibration = RESULTS/f"{STEM}-calibration-v1.npz"
    assert sha(weights) == fit["weights_sha256"] and sha(calibration) == fit["calibration_sha256"]
    with np.load(weights) as raw:
        w = {k: v.copy() for k, v in raw.items()}

    @tf.function(input_signature=[tf.TensorSpec((1, WIDTH, 1, 20), tf.float32)])
    def classifier(x):
        for layer, stride in ((0,1), (2,2)):
            kernel = w[f"{layer}.weight"].transpose(2,1,0)[:,None,:,:]
            x = tf.nn.relu(tf.nn.conv2d(x, kernel, (1,stride,1,1), "VALID") + w[f"{layer}.bias"])
        # Reorder the dense weights to consume time-major flattening directly.
        dense = w["5.weight"].reshape(24,24,13).transpose(0,2,1).reshape(24,312)
        x = tf.nn.relu(tf.linalg.matmul(tf.reshape(x,(1,312)), dense.T) + w["5.bias"])
        return tf.math.sigmoid(tf.linalg.matmul(x,w["7.weight"].T) + w["7.bias"])

    with np.load(calibration) as rows:
        x = rows["inputs"].transpose(0,2,1)[:,:,None,:].astype(np.float32)/255
        expected = rows["probabilities"]
    float_error = max(abs(float(classifier(v[None]).numpy().item()) - p) for v,p in zip(x,expected,strict=True))
    assert float_error < 0.0001, float_error
    converter = tf.lite.TFLiteConverter.from_concrete_functions([classifier.get_concrete_function()])
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = lambda: ([v[None]] for v in x)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.uint8
    binary = converter.convert()
    it = REF_KERNEL(model_content=binary)
    it.allocate_tensors()
    inp, = it.get_input_details()
    out, = it.get_output_details()
    assert out["quantization"] == (1/256,0)
    scale,zero = inp["quantization"]
    actual = []
    for v in x:
        it.set_tensor(inp["index"], np.clip(np.rint(v[None]/scale)+zero,-128,127).astype(np.int8))
        it.invoke()
        actual.append(int(it.get_tensor(out["index"]).item()))
    name = RESULTS/f"{STEM}-v1.tflite"
    with name.open("xb") as f:
        f.write(binary)
    save(RESULTS/f"{STEM}-export-v1.json",dict(sha256=sha(name), bytes=len(binary), weights_sha256=sha(weights),
         float_max_error=float(float_error), calibration_rows=len(x), int8_max_probability_error=float(abs(np.array(actual)/256-expected).max()),
         threshold_crossings=int(((np.array(actual)>=235) != (expected>=235/256)).sum()), input_scale=scale, input_zero=zero,
         operators=[r["op_name"] for r in it._get_ops_details()]))


if __name__=="__main__":
    main()
