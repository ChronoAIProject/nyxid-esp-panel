"""Transfer the class-distilled head to a fully integer TFLite window classifier."""
from pathlib import Path

import numpy as np
import torch

import train_c161_onset_evidence as fit
from fit_c161_history_gain_v1 import RESULTS, read, save, sha

STEM = fit.STEM + "-posterior-int8"
MODEL = RESULTS / (STEM + ".tflite")
EXPORT = RESULTS / (STEM + "-export.json")


def transfer(net):
    import tensorflow as tf
    L = tf.keras.layers
    inputs = L.Input(batch_shape=(1,144,40,1), name="native_features")
    x = inputs
    def block(x, conv, bn, weight, stride, pad):
        var = bn.running_var.numpy()
        gain = bn.weight.detach().numpy() / np.sqrt(var + bn.eps)
        bias = (conv.bias.detach().numpy()-bn.running_mean.numpy())*gain + bn.bias.detach().numpy()
        x = L.ZeroPadding2D(padding=pad)(x)
        layer = L.Conv2D(len(gain), weight.shape[:2], strides=stride, activation="relu")
        x = layer(x)
        layer.set_weights([weight*gain,bias])
        return x
    for i in (0,3,6):
        conv, bn = net.encoder[i:i+2]
        w = conv.weight.detach().numpy().transpose(2,3,1,0)
        x = block(x,conv,bn,w,conv.stride,(conv.padding[0],conv.padding[1]))
    x = L.Reshape((72,1,200))(x)
    for i in (0,3):
        conv, bn = net.temporal[i:i+2]
        w = conv.weight.detach().numpy().transpose(2,1,0)
        if i == 0:
            # NHWC flattens frequency before channels, PyTorch flattens channels before frequency.
            w = w.reshape(5,40,5,64).transpose(0,2,1,3).reshape(5,200,64)
        x = block(x,conv,bn,w[:,None,:,:],(1,1),(2,0))
    x = L.AveragePooling2D((12,1))(x)
    dense = L.Dense(5)
    x = dense(L.Flatten()(x))
    head = net.names[-1]
    dense.set_weights([head.weight.detach().numpy().reshape(5,64,6).transpose(2,1,0).reshape(384,5),
                       head.bias.detach().numpy()])
    return tf.keras.Model(inputs,L.Softmax()(x))


def main():
    import tensorflow as tf
    from ai_edge_litert.interpreter import Interpreter, OpResolverType
    tf.config.set_visible_devices([], "GPU")
    torch.set_num_threads(1)
    cfg = read(fit.FIT)
    path = fit.weight_file("posterior")
    cache = fit.base.WORK / "features.npz"
    assert sha(cache) == cfg["files"][str(cache)] and sha(path) == cfg["models"]["posterior"]["weights_sha256"]
    net = fit.base.Small(cfg["alphabet"]).eval()
    with np.load(path) as f:
        net.load_state_dict({k:torch.from_numpy(f[k]) for k in f})
    model = transfer(net)
    with np.load(cache) as f:
        corpus, labels = f["x"], f["labels"]
    rng = np.random.default_rng(79183)
    ids = np.concatenate([rng.choice(np.flatnonzero(labels==i),40,replace=False) for i in range(5)])
    x = corpus[ids].astype("float32")[:,:,:,None]/255
    with torch.inference_mode():
        expected = net(torch.tensor(x).permute(0,3,1,2))[0].softmax(dim=-1).numpy()
    fp = np.concatenate([model(v[None],training=False).numpy() for v in x])
    np.testing.assert_allclose(fp, expected, atol=2e-5, rtol=2e-5)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = lambda: ([v[None]] for v in x)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    raw = converter.convert()
    vm = Interpreter(model_content=raw,num_threads=1,experimental_op_resolver_type=OpResolverType.BUILTIN_REF)
    vm.allocate_tensors()
    inp, = vm.get_input_details()
    out, = vm.get_output_details()
    assert inp["dtype"] == out["dtype"] == np.int8 and tuple(inp["shape"]) == (1,144,40,1)
    assert tuple(out["shape"]) == (1,5) and out["quantization"] == (1/256,-128)
    assert all(t["dtype"] in (np.int8,np.int32) for t in vm.get_tensor_details())
    with MODEL.open("xb") as f:
        f.write(raw)
    save(EXPORT,dict(model_sha256=sha(MODEL), bytes=len(raw), training_fit_sha256=sha(fit.FIT),
                     weight_sha256=sha(path), script_sha256=sha(Path(__file__)), training_cache_sha256=sha(cache),
                     representative_indices=ids.tolist(), tensorflow=tf.__version__,
                     float_transfer_max_abs_probability_error=float(np.abs(fp-expected).max()),
                     input_quantization=inp["quantization"], output_quantization=out["quantization"],
                     nyx_min_int8_code=0, ops=[r["op_name"] for r in vm._get_ops_details()]))
    print(MODEL, len(raw), flush=True)


if __name__ == "__main__":
    main()
