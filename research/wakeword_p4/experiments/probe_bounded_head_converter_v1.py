"""Synthetic converter topology probe only; no trained model or dataset access."""
import hashlib
import json
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow.lite.python import schema_py_generated as schema

tf.config.set_visible_devices([], 'GPU')
tf.config.threading.set_intra_op_parallelism_threads(2)
ROOT = Path('research/wakeword_p4')

def main():
    out = ROOT/'results/4090-bounded-head-converter-probe-v1.json'
    assert not out.exists()
    x = tf.keras.Input(shape=(1,), batch_size=1)
    # z is the original scalar affine preactivation in this algebraic toy.
    u = tf.keras.layers.Dense(1, activation=tf.nn.relu6)(x)
    z = tf.keras.layers.Rescaling(8/3, offset=-8)(u)
    y = tf.keras.layers.Activation('sigmoid')(z)
    model = tf.keras.Model(x, y)
    model.layers[1].set_weights([np.array([[3/8]], np.float32), np.array([3], np.float32)])
    samples = np.concatenate([np.linspace(-3000, 400, 1001), np.linspace(-8, 8, 257)]).astype(np.float32)
    actual = model(samples[:, None], training=False).numpy().ravel()
    expected = np.exp(-np.logaddexp(0, -np.clip(samples, -8, 8)))
    error = float(np.max(np.abs(actual-expected)))
    assert error <= 1e-6
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = lambda: ([np.array([[v]], np.float32)] for v in samples)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.uint8
    data = converter.convert()
    graph = schema.Model.GetRootAsModel(data, 0)
    sub = graph.Subgraphs(0)
    names = {v:k for k,v in vars(schema.BuiltinOperator).items() if isinstance(v,int)}
    ops = []
    for i in range(sub.OperatorsLength()):
        op = sub.Operators(i)
        name = names[graph.OperatorCodes(op.OpcodeIndex()).BuiltinCode()]
        row = dict(op=name, inputs=op.InputsAsNumpy().tolist(), outputs=op.OutputsAsNumpy().tolist())
        if name == 'FULLY_CONNECTED':
            options = schema.FullyConnectedOptions()
            table = op.BuiltinOptions(); options.Init(table.Bytes, table.Pos)
            row['fused_activation'] = int(options.FusedActivationFunction())
        ops.append(row)
    runtime = tf.lite.Interpreter(model_content=data, num_threads=2)
    runtime.allocate_tensors()
    tensors = [{k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in dict(index=t['index'],name=t['name'],shape=t['shape'],scale=t['quantization'][0],zero=t['quantization'][1]).items()} for t in runtime.get_tensor_details()]
    fused = any(o.get('fused_activation') == schema.ActivationFunctionType.RELU6 for o in ops)
    result = dict(schema='bounded-head-converter-probe-v1', tensorflow=tf.__version__, script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), synthetic_flatbuffer_sha256=hashlib.sha256(data).hexdigest(), float_algebra_max_error=error, clipping_probability_bound=float(1/(1+np.exp(8))), fc_relu6_fused=fused, ops=ops,tensors=tensors, limits='Toy topology only, not a trained candidate or numerical admission. No trained weights/audio/features/development accessed. Input quantization is deliberately unrealistic for a full encoder. Fused bounded FC does not prove upstream encoder/state accuracy; full original 270 gates remain required before any development.')
    out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))

if __name__ == '__main__': main()
