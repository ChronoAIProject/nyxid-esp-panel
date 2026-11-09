"""Import an explicitly supplied ESPHome v2 30/60-channel encoder.

The source model's quantized convolution weights already include folded batch
normalization. Reconstruct these as biased float layers, without a second BN.
Source binaries remain outside git. The final classifier is reset for Hey Nyx.
"""
import hashlib

import numpy as np
import tensorflow as tf
from ai_edge_litert.interpreter import Interpreter, OpResolverType


def build_model(frames, source=None, reuse_head=False):
    from microwakeword.layers import stream

    audio = tf.keras.layers.Input(shape=(frames, 40), name="input_audio")
    net = tf.keras.ops.expand_dims(audio, axis=2)
    first = tf.keras.layers.Conv2D(30, (5, 1), strides=(3, 1), use_bias=True, name="first_conv")
    net = stream.Stream(cell=first, use_one_step=False, pad_time_dim=None, pad_freq_dim="valid")(net)
    net = tf.keras.layers.ReLU()(net)
    convolutions, depthwise = [first], []
    for block, kernel in enumerate((5, 9, 13, 21)):
        net = stream.Stream(cell=tf.keras.layers.Identity(), ring_buffer_size_in_time_dim=kernel-1,
                            use_one_step=False)(net)
        dw = tf.keras.layers.DepthwiseConv2D((kernel, 1), use_bias=True, name=f"depthwise_{block}")
        pw = tf.keras.layers.Conv2D(60, (1, 1), use_bias=True, name=f"pointwise_{block}")
        net = tf.keras.layers.ReLU()(pw(dw(net)))
        depthwise.append(dw)
        convolutions.append(pw)
    if net.shape[1] > 1:
        net = stream.Stream(cell=tf.keras.layers.Identity(), ring_buffer_size_in_time_dim=net.shape[1]-1,
                            use_one_step=False)(net)
    head = tf.keras.layers.Dense(1, activation="sigmoid", name="classifier")
    model = tf.keras.Model(audio, head(tf.keras.layers.Flatten()(net)))
    if source is None:
        return model, None

    interpreter = Interpreter(model_path=str(source), experimental_op_resolver_type=OpResolverType.BUILTIN_REF)
    interpreter.allocate_tensors()
    if tuple(interpreter.get_input_details()[0]["shape"]) != (1, 3, 40):
        raise ValueError("Expected a 10 ms / three-frame ESPHome v2 input")
    details = {r["index"]: r for r in interpreter.get_tensor_details()}

    def dequantize(index):
        detail = details[int(index)]
        array = interpreter.get_tensor(int(index)).astype(np.float32)
        quant = detail["quantization_parameters"]
        if not len(quant["scales"]):
            raise ValueError("Expected explicit quantization parameters")
        shape = [1]*array.ndim
        shape[quant["quantized_dimension"]] = len(quant["scales"])
        return (array-quant["zero_points"].reshape(shape))*quant["scales"].reshape(shape)

    ops = interpreter._get_ops_details()
    conv_ops = [op for op in ops if op["op_name"] == "CONV_2D"]
    dw_ops = [op for op in ops if op["op_name"] == "DEPTHWISE_CONV_2D"]
    dense_ops = [op for op in ops if op["op_name"] == "FULLY_CONNECTED"]
    if (len(conv_ops), len(dw_ops), len(dense_ops)) != (5, 4, 1):
        raise ValueError("Expected the reviewed 30/60-channel v2 topology")
    for layer, op in zip(convolutions, conv_ops):
        kernel, bias = [dequantize(index) for index in op["inputs"][1:3]]
        weights = [kernel.transpose(1, 2, 3, 0), bias]
        if [tuple(w.shape) for w in weights] != [tuple(w.shape) for w in layer.weights]:
            raise ValueError("Source convolution shape differs from reviewed topology")
        layer.set_weights(weights)
    for layer, op in zip(depthwise, dw_ops):
        kernel, bias = [dequantize(index) for index in op["inputs"][1:3]]
        weights = [kernel.transpose(1, 2, 3, 0), bias]
        if [tuple(w.shape) for w in weights] != [tuple(w.shape) for w in layer.weights]:
            raise ValueError("Source depthwise shape differs from reviewed topology")
        layer.set_weights(weights)
    if reuse_head:
        kernel, bias = [dequantize(index) for index in dense_ops[0]["inputs"][1:3]]
        if tuple(kernel.T.shape) != tuple(head.kernel.shape):
            raise ValueError("Source classifier audit requires its original 149-frame context")
        head.set_weights([kernel.T, bias])
    return model, {"source_model_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                   "source_topology": "ESPHome v2: first 30, pointwise 60, depthwise 5/9/13/21, stride 3",
                   "import": "Dequantized convolutions and folded biases; no duplicate batch normalization",
                   "classifier": "Original head reused for audit" if reuse_head else "Reset for Hey Nyx"}
