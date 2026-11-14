"""FP32 folded reference for final learned-grid whole-wave qualification.

Consumes the actual serialized FP32 W/b, not the old raw-BN H5. No quantizers,
activation-step treatment or input rounding belongs in this float reference.
Importing this module does not import TensorFlow or invoke a model.
"""
import numpy as np


def parameters(interpreter):
    """Read final FP32 constants and actual fused activations from LiteRT."""
    # Public op details identify constants; activation metadata is supplied by
    # inspect_graph below, parsed directly from the same serialized artifact.
    operations = [o for o in interpreter._get_ops_details()
                  if o['op_name'] in ('CONV_2D', 'DEPTHWISE_CONV_2D', 'FULLY_CONNECTED')]
    assert len(operations) == 10
    pairs = [(interpreter.get_tensor(int(o['inputs'][1])).copy(),
              interpreter.get_tensor(int(o['inputs'][2])).copy()) for o in operations]
    assert all(a.dtype == np.float32 and np.isfinite(a).all() for pair in pairs for a in pair)
    return pairs


def inspect_graph(data):
    from ai_edge_litert import schema_py_generated as s
    model = s.ModelT.InitFromObj(s.Model.GetRootAsModel(data, 0))
    graph = model.subgraphs[0]
    kinds = {s.BuiltinOperator.CONV_2D: 'conv',
             s.BuiltinOperator.DEPTHWISE_CONV_2D: 'depthwise',
             s.BuiltinOperator.FULLY_CONNECTED: 'dense'}
    configs = []
    for op in graph.operators:
        code = model.operatorCodes[op.opcodeIndex].builtinCode
        if code not in kinds:
            continue
        options = op.builtinOptions
        activation = options.fusedActivationFunction
        assert activation in (s.ActivationFunctionType.NONE, s.ActivationFunctionType.RELU)
        configs.append(dict(kind=kinds[code], relu=activation == s.ActivationFunctionType.RELU))
    assert [c['kind'] for c in configs] == ['conv', 'depthwise', 'conv', 'depthwise', 'conv', 'depthwise', 'conv', 'depthwise', 'conv', 'dense']
    return configs


def forward(values, pairs, configs, tf, streaming):
    """Whole-stream causal outputs, or final valid offline-window probability.

    Streaming prepends each layer's own zero reset history independently.
    Offline applies valid temporal convolutions to the entire225-frame window.
    Thus phase224+zero and nominal225 can be compared without using one
    streaming result as its own offline reference.
    """
    x = tf.convert_to_tensor(values, tf.float32)
    tf.debugging.assert_equal(tf.shape(x)[-1], 40)
    assert len(pairs) == len(configs) == 10
    for j, (pair, config) in enumerate(zip(pairs[:-1], configs[:-1], strict=True)):
        weight, bias = (tf.convert_to_tensor(v, tf.float32) for v in pair)
        kernel = int(weight.shape[1])
        stride = 3 if j == 0 else 1
        if streaming:
            x = tf.pad(x, [[0, 0], [kernel - stride, 0], [0, 0]])
        expanded = x[:, :, None, :]
        if config['kind'] == 'conv':
            filtered = tf.nn.conv2d(expanded, tf.transpose(weight, [1, 2, 3, 0]),
                                    strides=[1, stride, 1, 1], padding='VALID')
        else:
            assert config['kind'] == 'depthwise' and stride == 1
            filtered = tf.nn.depthwise_conv2d(expanded, tf.transpose(weight, [1, 2, 3, 0]),
                                              strides=[1, 1, 1, 1], padding='VALID')
        x = filtered[:, :, 0, :] + bias
        if config['relu']:
            x = tf.nn.relu(x)
    tf.debugging.assert_equal(tf.shape(x)[-1], 60)
    if streaming:
        frames = tf.signal.frame(tf.pad(x, [[0, 0], [29, 0], [0, 0]]), 30, 1, axis=1)
        embedding = tf.reshape(frames, [tf.shape(x)[0], tf.shape(x)[1], 1800])
    else:
        tf.debugging.assert_equal(tf.shape(x)[1], 30)
        embedding = tf.reshape(x, [tf.shape(x)[0], 1, 1800])
    weight, bias = (tf.convert_to_tensor(v, tf.float32) for v in pairs[-1])
    logits = tf.linalg.matmul(embedding, weight, transpose_b=True) + bias
    if configs[-1]['relu']:
        logits = tf.nn.relu(logits)
    return tf.math.sigmoid(logits)
