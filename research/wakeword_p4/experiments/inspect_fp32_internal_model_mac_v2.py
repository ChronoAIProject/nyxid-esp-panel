"""Inspect trained internal-state FP32 with the explicit input ExpandDims; no inference."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from ai_edge_litert import schema_py_generated as s


OPS = {"CALL_ONCE", "VAR_HANDLE", "READ_VARIABLE", "RESHAPE", "CONCATENATION",
       "STRIDED_SLICE", "ASSIGN_VARIABLE", "CONV_2D", "DEPTHWISE_CONV_2D",
       "FULLY_CONNECTED", "LOGISTIC", "EXPAND_DIMS"}
STATE_SHAPES = sorted([[1, 2, 1, 40], [1, 4, 1, 30], [1, 8, 1, 60],
                       [1, 12, 1, 60], [1, 20, 1, 60], [1, 29, 1, 60]])


def require(condition, message):
    if not condition:
        raise ValueError(message)


def inspect(data):
    require(data[4:8] == b"TFL3", "TFLite identifier")
    model = s.Model.GetRootAsModel(data, 0)
    require(model.Version() == 3 and model.SubgraphsLength() == 2, "Main plus initializer graphs")
    names = {v: k for k, v in vars(s.BuiltinOperator).items() if isinstance(v, int)}
    codes = []
    for i in range(model.OperatorCodesLength()):
        code = model.OperatorCodes(i)
        require(not code.CustomCode(), "No custom operators")
        codes.append(dict(op=names.get(code.BuiltinCode(), "UNKNOWN"), version=code.Version()))
    require({c["op"] for c in codes} == OPS, "Exact registered FP32 operator set")

    def kind(op):
        return codes[op.OpcodeIndex()]["op"]

    def shape(tensor):
        return [int(tensor.Shape(i)) for i in range(tensor.ShapeLength())]

    main = model.Subgraphs(0)
    require(main.InputsLength() == main.OutputsLength() == 1, "Single audio input and probability output")
    audio, probability = main.Tensors(main.Inputs(0)), main.Tensors(main.Outputs(0))
    require(audio.Type() == probability.Type() == s.TensorType.FLOAT32 and
            shape(audio) == [1, 3, 40] and shape(probability) == [1, 1], "Exact FP32 input/output ABI")
    # Prospectively support the trained export's shape-only input expansion.
    expanded, dense, logistic = [], [], []
    for i in range(main.OperatorsLength()):
        op = main.Operators(i)
        if kind(op) == "EXPAND_DIMS":
            require(op.InputsLength() == 2 and op.OutputsLength() == 1 and
                    op.Inputs(0) == main.Inputs(0), "ExpandDims only expands original audio")
            axis = main.Tensors(op.Inputs(1))
            target = main.Tensors(op.Outputs(0))
            raw = bytes(model.Buffers(axis.Buffer()).DataAsNumpy())
            require(axis.Type() == s.TensorType.INT32 and shape(axis) == [] and
                    raw == b"\x02\x00\x00\x00" and target.Type() == s.TensorType.FLOAT32 and
                    shape(target) == [1, 3, 1, 40], "Fixed scalar axis2 and shape-only audio expansion")
            expanded.append(i)
        elif kind(op) == "FULLY_CONNECTED":
            dense.append(op)
        elif kind(op) == "LOGISTIC":
            logistic.append(op)
    require(len(expanded) == len(dense) == len(logistic) == 1,
            "One input expansion and original Dense/sigmoid head")
    require(dense[0].OutputsLength() == logistic[0].InputsLength() == logistic[0].OutputsLength() == 1 and
            dense[0].Outputs(0) == logistic[0].Inputs(0) and logistic[0].Outputs(0) == main.Outputs(0),
            "Raw Dense directly feeds sole sigmoid probability output")
    tensor_inventory = []
    for gi in range(model.SubgraphsLength()):
        graph = model.Subgraphs(gi)
        for ti in range(graph.TensorsLength()):
            tensor = graph.Tensors(ti)
            require(tensor.Type() in (s.TensorType.FLOAT32, s.TensorType.INT32, s.TensorType.RESOURCE),
                    "No quantized, float16 or custom tensor types")
            q = tensor.Quantization()
            require(q is None or q.ScaleLength() == q.ZeroPointLength() == 0, "No quantization grids")
            signature = tensor.ShapeSignatureAsNumpy()
            require(isinstance(signature, int) or np.all(signature >= 0), "No dynamic tensor dimensions")
            buffer = model.Buffers(tensor.Buffer())
            if tensor.Type() == s.TensorType.FLOAT32 and buffer.DataLength():
                raw = bytes(buffer.DataAsNumpy())
                require(len(raw) == int(np.prod(shape(tensor))) * 4 and
                        np.isfinite(np.frombuffer(raw, dtype="<f4")).all(), "Finite complete FP32 constants")
            tensor_inventory.append(dict(graph=gi, tensor=ti, type=tensor.Type(), shape=shape(tensor)))

    def handles(graph):
        result = {}
        for i in range(graph.OperatorsLength()):
            op = graph.Operators(i)
            if kind(op) == "VAR_HANDLE":
                require(op.InputsLength() == 0 and op.OutputsLength() == 1, "Handle arity")
                options = s.VarHandleOptions()
                raw = op.BuiltinOptions()
                options.Init(raw.Bytes, raw.Pos)
                key = (options.Container(), options.SharedName())
                require(key not in result.values() and key[1], "Unique named variable")
                result[int(op.Outputs(0))] = key
        return result

    main_handles = handles(main)
    reads, writes, init_ids = {}, [], []
    for i in range(main.OperatorsLength()):
        op = main.Operators(i)
        name = kind(op)
        if name == "CALL_ONCE":
            options = s.CallOnceOptions()
            raw = op.BuiltinOptions()
            options.Init(raw.Bytes, raw.Pos)
            init_ids.append(options.InitSubgraphIndex())
        elif name == "READ_VARIABLE":
            key = main_handles[int(op.Inputs(0))]
            require(key not in reads and op.InputsLength() == op.OutputsLength() == 1, "One read per state")
            reads[key] = main.Tensors(op.Outputs(0))
        elif name == "ASSIGN_VARIABLE":
            key = main_handles[int(op.Inputs(0))]
            require(op.InputsLength() == 2 and op.OutputsLength() == 0, "State update arity")
            tensor = main.Tensors(op.Inputs(1))
            require(key in reads and tensor.Type() == s.TensorType.FLOAT32 and
                    shape(tensor) == shape(reads[key]), "State update shape/type")
            writes.append(key)
        elif name == "FULLY_CONNECTED":
            options = s.FullyConnectedOptions()
            raw = op.BuiltinOptions()
            options.Init(raw.Bytes, raw.Pos)
            require(options.FusedActivationFunction() == s.ActivationFunctionType.NONE, "Original unbounded Dense head")
    require(init_ids == [1] and len(main_handles) == len(reads) == len(writes) == 6 and
            len(set(writes)) == 6 and set(writes) == set(reads), "Six states read/carried/initialized once")
    init = model.Subgraphs(1)
    init_handles = handles(init)
    require(set(init_handles.values()) == set(reads), "Initializer/main variable identities")
    states, used = [], set()
    for i in range(init.OperatorsLength()):
        op = init.Operators(i)
        require(kind(op) in ("VAR_HANDLE", "ASSIGN_VARIABLE"), "Constant initializer only")
        if kind(op) != "ASSIGN_VARIABLE":
            continue
        require(op.InputsLength() == 2 and op.OutputsLength() == 0, "Initializer assignment arity")
        key = init_handles[int(op.Inputs(0))]
        require(key in reads and key not in used, "One initializer per original state")
        used.add(key)
        tensor = init.Tensors(op.Inputs(1))
        require(tensor.Type() == reads[key].Type() == s.TensorType.FLOAT32 and
                shape(tensor) == shape(reads[key]), "Initializer/read shape and FP32 type")
        buffer = model.Buffers(tensor.Buffer())
        raw = bytes(buffer.DataAsNumpy())
        require(len(raw) == int(np.prod(shape(tensor))) * 4 and not any(raw), "Exact positive-zero state bytes")
        offset = buffer._tab.Vector(buffer._tab.Offset(4))
        states.append(dict(shared_name=key[1].decode(), shape=shape(tensor), bytes=len(raw),
                           buffer=int(tensor.Buffer()), offset=offset, all_real_zero=True))
    require(used == set(reads) and sorted(row["shape"] for row in states) == STATE_SHAPES,
            "All six original context225 state shapes")
    return dict(schema="fp32-internal-model-contract-v2", model_sha256=hashlib.sha256(data).hexdigest(),
                model_bytes=len(data), operators=codes, inputs=1, outputs=1, audio_shape=shape(audio),
                probability_shape=shape(probability), states=states, state_bytes=sum(row["bytes"] for row in states),
                tensor_inventory_sha256=hashlib.sha256(json.dumps(tensor_inventory, sort_keys=True).encode()).hexdigest(),
                model_invocations=0,
                limits="FlatBuffer contract/finite constants/initial-state inspection only. Does not prove source-H5 parameter equivalence, probability numerics, recognition, allocation, timing or P4 compatibility.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    result = inspect(args.model.read_bytes())
    with args.report.open("x") as output:
        output.write(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
