"""Check copied trained FlatBuffer rejection without running inference."""
import argparse
import hashlib
import json
from pathlib import Path

import flatbuffers
from ai_edge_litert import schema_py_generated as s

from inspect_fp32_internal_model_mac_v2 import inspect


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--report", type=Path, required=True)
    a = p.parse_args()
    original = a.model.read_bytes()
    accepted = inspect(original)
    cases = []

    def reject(label, edit):
        model = s.ModelT.InitFromObj(s.Model.GetRootAsModel(original, 0))
        edit(model)
        builder = flatbuffers.Builder(0)
        builder.Finish(model.Pack(builder), file_identifier=b"TFL3")
        try:
            inspect(bytes(builder.Output()))
        except ValueError as e:
            cases.append(dict(case=label, rejected=True, message=str(e)))
        else:
            raise AssertionError("Accepted invalid fixture: " + label)

    def concrete_batch(model):
        g = model.subgraphs[0]
        t = g.tensors[g.inputs[0]]
        t.shape = [-1, *t.shape[1:]]

    def symbolic_time(model):
        g = model.subgraphs[0]
        t = g.tensors[g.inputs[0]]
        t.shapeSignature = [t.shapeSignature[0], -1, *t.shapeSignature[2:]]

    def symbolic_output(model):
        g = model.subgraphs[0]
        g.tensors[g.outputs[0]].shapeSignature = [-1, 1]

    def wrong_axis(model):
        g = model.subgraphs[0]
        op = next(o for o in g.operators if model.operatorCodes[o.opcodeIndex].builtinCode == s.BuiltinOperator.EXPAND_DIMS)
        axis = g.tensors[op.inputs[1]]
        model.buffers[axis.buffer].data = [1, 0, 0, 0]

    def bounded_head(model):
        op = next(o for o in model.subgraphs[0].operators if model.operatorCodes[o.opcodeIndex].builtinCode == s.BuiltinOperator.FULLY_CONNECTED)
        op.builtinOptions.fusedActivationFunction = s.ActivationFunctionType.RELU6

    for label, edit in [("dynamic concrete batch", concrete_batch), ("symbolic time", symbolic_time),
                        ("symbolic probability output", symbolic_output), ("ExpandDims wrong axis", wrong_axis),
                        ("bounded Dense head", bounded_head)]:
        reject(label, edit)
    assert a.model.read_bytes() == original
    result = dict(schema="mac-fp32-internal-trained-contract-review-v1",
                  original_sha256=hashlib.sha256(original).hexdigest(), accepted_original=True,
                  permitted_metadata=accepted["symbolic_batch_metadata"], rejected_cases=cases,
                  all_passed=len(cases) == 5, original_bytes_unchanged=True, model_invocations=0,
                  inspector_sha256=hashlib.sha256(Path(__file__).with_name("inspect_fp32_internal_model_mac_v2.py").read_bytes()).hexdigest(),
                  review_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    with a.report.open("x") as f:
        f.write(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
