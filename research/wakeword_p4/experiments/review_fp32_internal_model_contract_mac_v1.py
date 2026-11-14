"""Exercise ABI/initializer/head rejection on copied random-model bytes only."""
import argparse
import hashlib
import json
from pathlib import Path
import struct

import flatbuffers
from ai_edge_litert import schema_py_generated as s

from inspect_fp32_internal_model_mac_v1 import inspect, STATE_SHAPES


def pack(model):
    builder = flatbuffers.Builder(0)
    builder.Finish(model.Pack(builder), file_identifier=b"TFL3")
    return bytes(builder.Output())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    original = args.model.read_bytes()
    before = hashlib.sha256(original).hexdigest()
    accepted = inspect(original)
    rejected = []

    def reject(label, data):
        try:
            inspect(data)
        except ValueError as error:
            rejected.append(dict(case=label, error=str(error)))
        else:
            raise AssertionError("Expected rejection: " + label)

    # A well-formed FlatBuffer with six additional unused float32 inputs
    # exercises the actual ABI count guard, without an external-state adapter.
    model = s.ModelT.InitFromObj(s.Model.GetRootAsModel(original, 0))
    main_graph = model.subgraphs[0]
    new_inputs = list(main_graph.inputs)
    for i, shape in enumerate(STATE_SHAPES):
        tensor = s.TensorT()
        tensor.name = f"constructed_extra_state_{i}".encode()
        tensor.type = s.TensorType.FLOAT32
        tensor.shape = shape
        tensor.buffer = 0
        new_inputs.append(len(main_graph.tensors))
        main_graph.tensors.append(tensor)
    main_graph.inputs = new_inputs
    reject("seven-input ABI with unchanged internal-state body", pack(model))

    changed = bytearray(original)
    struct.pack_into("<f", changed, accepted["states"][0]["offset"], 1.0)
    reject("nonzero first state initializer", changed)

    model = s.ModelT.InitFromObj(s.Model.GetRootAsModel(original, 0))
    model.subgraphs[0].tensors[model.subgraphs[0].outputs[0]].type = s.TensorType.FLOAT16
    reject("float16 probability output", pack(model))

    model = s.ModelT.InitFromObj(s.Model.GetRootAsModel(original, 0))
    count = 0
    for op in model.subgraphs[0].operators:
        if model.operatorCodes[op.opcodeIndex].builtinCode == s.BuiltinOperator.FULLY_CONNECTED:
            op.builtinOptions.fusedActivationFunction = s.ActivationFunctionType.RELU6
            count += 1
    assert count == 1
    reject("bounded activation on original Dense head", pack(model))
    assert hashlib.sha256(args.model.read_bytes()).hexdigest() == before
    result = dict(schema="mac-fp32-internal-model-contract-constructed-review-v1",
                  original_model_sha256=before, accepted_original=True,
                  original_states=len(accepted["states"]), original_state_bytes=accepted["state_bytes"],
                  rejected_cases=rejected, all_checks_passed=len(rejected) == 4,
                  inspector_sha256=hashlib.sha256(Path(__file__).with_name("inspect_fp32_internal_model_mac_v1.py").read_bytes()).hexdigest(),
                  review_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  original_bytes_unchanged=True, model_invocations=0, media_reads=0,
                  limits="Static random-model/copy-mutation checks only. Seven-input fixture is not an implemented external-state adapter. No trained model, numerical inference, allocation, timing or recognition tested.")
    with args.report.open("x") as output:
        output.write(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
