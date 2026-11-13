"""Raw UINT8 streaming output for future explicitly versioned event policy.

Input quantization and invocation cadence match the frozen reference Detector.
No output float conversion or score rounding occurs here.
"""
import numpy as np
from ai_edge_litert.interpreter import Interpreter, OpResolverType


class Uint8StreamDetector:
    def __init__(self, model):
        self.model = str(model)
        self.reset()

    def reset(self):
        self.interpreter = Interpreter(model_path=self.model,
            experimental_op_resolver_type=OpResolverType.BUILTIN_REF, num_threads=1)
        self.interpreter.allocate_tensors()
        self.input = self.interpreter.get_input_details()[0]
        self.output = self.interpreter.get_output_details()[0]
        if (self.input['dtype'] != np.int8 or self.output['dtype'] != np.uint8
                or tuple(self.input['shape']) != (1, 3, 40)
                or tuple(self.output['shape']) != (1, 1)
                or tuple(self.output['quantization']) != (1/256, 0)):
            raise ValueError('Requires authenticated INT8 [1,3,40] / UINT8 [1,1], scale1/256 zero0')
        self.scale, self.zero = self.input['quantization']
        if not np.isfinite(self.scale) or self.scale <= 0:
            raise ValueError('Invalid input quantization scale')

    def predict(self, raw):
        raw = np.asarray(raw)
        if raw.ndim != 2 or raw.shape[1] != 40 or not np.isfinite(raw).all():
            raise ValueError('Expected finite [frames,40] features')
        values = np.clip(raw.astype(np.float32)/25.6, 0, 26)
        values = np.clip(np.rint(values/self.scale)+self.zero, -128, 127).astype(np.int8)
        result = []
        for at in range(0, len(values)-2, 3):
            self.interpreter.set_tensor(self.input['index'], values[None, at:at+3])
            self.interpreter.invoke()
            result.append(self.interpreter.get_tensor(self.output['index'])[0, 0])
        return np.asarray(result, dtype=np.uint8)
