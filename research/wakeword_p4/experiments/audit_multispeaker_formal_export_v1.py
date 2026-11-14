"""Inspect saved FlatBuffers and independently replay fixed smoke inputs; no exports."""
import hashlib
import json
from pathlib import Path
import numpy as np
from ai_edge_litert.interpreter import Interpreter
from tensorflow.lite.python import schema_py_generated as S

ROOT = Path('.local/wakeword/multispeaker-formal-export-v1')
RESULTS = Path('research/wakeword_p4/results')
def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

static = json.loads((ROOT/'result.json').read_text())
assert static['formal_training_completed'] is True
runtime = json.loads((ROOT/'runtime.json').read_text())
plan_path = RESULTS/'4090-multispeaker-formal-export-plan-v1.json'
plan = json.loads(plan_path.read_text())
assert static['execution_sha256'] == runtime['execution_sha256'] == sha(plan_path)
assert runtime['static_result_sha256'] == sha(ROOT/'result.json')
names = {v:k for k,v in vars(S.BuiltinOperator).items() if isinstance(v,int)}
rows = []
for model in static['models']:
    arm = model['condition']
    material = next(m for m in plan['material'] if m['condition']==arm)
    assert sha(material['path']) == material['sha256']
    expected_features = np.load(material['path'], allow_pickle=False)[0,0]
    for kind, artifact in model['artifacts'].items():
        path = Path(artifact['path']); data = path.read_bytes()
        assert len(data)==artifact['bytes'] and sha(path)==artifact['sha256']
        graph = S.Model.GetRootAsModel(data,0)
        main = graph.Subgraphs(0)
        assert main.InputsLength()==main.OutputsLength()==1
        inp,out = main.Tensors(main.Inputs(0)),main.Tensors(main.Outputs(0))
        assert list(inp.ShapeAsNumpy())==[1,3,40] and list(out.ShapeAsNumpy())==[1,1]
        ops=[names[graph.OperatorCodes(main.Operators(i).OpcodeIndex()).BuiltinCode()] for i in range(main.OperatorsLength())]
        assert 'CUSTOM' not in ops and ops==artifact['ops']
        initializers=[]
        for si in range(1,graph.SubgraphsLength()):
            sg=graph.Subgraphs(si)
            for ti in range(sg.TensorsLength()):
                t=sg.Tensors(ti); buf=graph.Buffers(t.Buffer())
                if not buf.DataLength():continue
                if kind=='int8':
                    assert t.Type()==S.TensorType.INT8
                    zero=int(t.Quantization().ZeroPoint(0)); values=buf.DataAsNumpy().view(np.int8)
                else:
                    assert t.Type()==S.TensorType.FLOAT32
                    zero=0; values=buf.DataAsNumpy().view('<f4')
                assert np.all(values==zero)
                initializers.append(dict(shape=t.ShapeAsNumpy().tolist(),zero=zero,all_real_zero=True))
        assert len(initializers)==6 and initializers==artifact['initializers']
        if kind=='int8':
            assert inp.Type()==S.TensorType.INT8 and out.Type()==S.TensorType.UINT8
            assert all(main.Tensors(i).Type()!=S.TensorType.FLOAT32 for i in range(main.TensorsLength()))
            assert 'DEQUANTIZE' not in ops
        else:assert inp.Type()==out.Type()==S.TensorType.FLOAT32
        row=next(r for r in runtime['artifacts'] if r['arm']==arm and r['kind']==kind)
        assert sha(row['trace_path'])==row['trace_sha256']
        with np.load(row['trace_path'],allow_pickle=False) as trace:
            assert np.array_equal(trace['features'],expected_features)
            a,b=trace['first'],trace['second']
            assert a.shape==b.shape==(103,) and np.array_equal(a,b) and np.isfinite(a).all()
            scale,zero=row['input_quantization']
            expected=(np.clip(np.rint(expected_features/scale)+zero,-128,127).astype(np.int8)
                      if kind=='int8' else expected_features)
            assert np.array_equal(expected,trace['supplied'])
            if kind=='int8':assert a.dtype==np.uint8 and row['output_quantization']==[1/256,0]
            else:assert a.dtype==np.float32 and np.all((a>=0)&(a<=1))
            # One additional fresh interpreter, preserving uninterrupted stream state.
            runner=Interpreter(model_path=str(path)); runner.allocate_tensors()
            ai,bo=runner.get_input_details()[0],runner.get_output_details()[0]
            assert list(ai['quantization'])==row['input_quantization']
            assert list(bo['quantization'])==row['output_quantization']
            actual=[]
            for start in range(0,309,3):
                runner.set_tensor(ai['index'],expected[None,start:start+3]);runner.invoke()
                actual.append(runner.get_tensor(bo['index'])[0,0])
            assert np.array_equal(np.asarray(actual),a)
        rows.append(dict(arm=arm,kind=kind,model_sha256=sha(path),trace_sha256=row['trace_sha256'],
                         initializer_count=6,independent_replay_calls=103,all_checks_passed=True))
report=dict(schema='multispeaker-formal-export-independent-audit-v1',
            static_report_sha256=sha(ROOT/'result.json'),runtime_report_sha256=sha(ROOT/'runtime.json'),
            auditor_sha256=sha(__file__),artifacts=rows,all_checks_passed=len(rows)==4,
            limits='Saved material/quantization/repeat/FlatBuffer and 412 fresh replay calls only. Not FP32-INT8 accuracy, waveform qualification, recognition, formal training or P4 resource admission.')
with (RESULTS/'4090-multispeaker-formal-export-independent-audit-v1.json').open('x') as f:
    f.write(json.dumps(report,indent=2)+'\n')
print(json.dumps(report))
