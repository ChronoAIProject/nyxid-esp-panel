"""Export audited raw continuation final10000 only; one FP32 conversion/arm."""
import argparse,hashlib,importlib.metadata,json,subprocess,traceback
from pathlib import Path
import numpy as np
R=Path('research/wakeword_p4/results');PLAN=R/'4090-hey-common-carrier-raw-fp32-export-execution-v1.json';RESULT=R/'4090-hey-common-carrier-raw-fp32-export-result-v1.json';OUT=Path('.local/wakeword/hey-common-carrier-raw-fp32-export-v1')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def plan(check_only=False):
    required=[R/'4090-hey-common-carrier-raw-formal-result-v1.json',R/'4090-hey-common-carrier-raw-formal-independent-audit-v1.json',R/'4090-hey-common-carrier-raw-formal-execution-v1.json']
    for p in required:
        if not p.exists():print(json.dumps(dict(status='not_ready',missing=str(p))));return 2
    result,audit,execution=[json.loads(p.read_text()) for p in required]
    assert result['all_passed'] and audit['all_passed'] and audit['original_result_sha256']==sha(required[0]) and result['execution_sha256']==sha(required[2])
    weights=[]
    for case in result['cases']:
        assert case['repeat_exact'] is None and len(case['repeats'])==1
        final=case['repeats'][0];assert final['steps']==10000 and final['frozen_bn_exact'] and sha(final['weights_path'])==final['weights_sha256']
        weights.append(dict(target=case['target'],condition=case['condition'],path=final['weights_path'],sha256=final['weights_sha256']))
    assert {(w['target'],w['condition']) for w in weights}=={(t,c) for t in ('hey_willow','hey_nyx') for c in ('control','command_context')}
    if check_only:print(json.dumps(dict(status='ready',models=4)));return 0
    files=dict(execution['files']);paths=[*required,Path(__file__),Path(__file__).with_name('probe_hey_context_transfer_shapes_v1.py')]+[Path(w['path']) for w in weights]
    for p in paths:files[str(p)]=sha(p)
    for p,h in files.items():assert sha(p)==h
    with PLAN.open('x') as f:f.write(json.dumps(dict(files=files,weights=weights,versions={n:importlib.metadata.version(n) for n in ('numpy','tensorflow','keras','ai-edge-litert')},policy='All4 independently audited rawfinal10000 only, one builtin FP32 conversion each. Original raw sigmoid/frozenBN unchanged, no calibration/INT8/PTQ/retry/head rewrite/selection. Six streaming histories serialized realzero; original H5 authenticated. Static ABI is not newweight waveform phase/numeric qualification or recognition admission.'),indent=2)+'\n')
    return 0

def run():
    assert subprocess.check_output(['git','show','origin/research-hye-nyx-voice-gpu:'+str(PLAN)])==PLAN.read_bytes();p=json.loads(PLAN.read_text())
    for f,h in p['files'].items():assert sha(f)==h
    for n,v in p['versions'].items():assert importlib.metadata.version(n)==v
    OUT.mkdir(exist_ok=False)
    try:
        import probe_hey_context_transfer_shapes_v1 as helper
        from microwakeword import utils
        from microwakeword.layers import modes
        from ai_edge_litert import schema_py_generated as s
        tf=helper.tf;models=[]
        for w in p['weights']:
            folder=OUT/(w['target']+'-'+w['condition']);folder.mkdir();model=helper.create(225);model.load_weights(w['path'])
            for layer in model.layers:
                cell=getattr(layer,'cell',layer)
                if isinstance(cell,tf.keras.layers.BatchNormalization):layer.trainable=cell.trainable=False
            original=[v.numpy().copy() for v in model.variables]
            stream=utils.to_streaming_inference(model,{'stride':3,'spectrogram_length':225},modes.Modes.STREAM_INTERNAL_STATE_INFERENCE);stream(np.zeros((1,3,40),np.float32),training=False)
            states=[v for v in stream.variables if 'states' in v.name];assert len(states)==6
            for state in states:state.assign(tf.zeros_like(state))
            stream.export(str(folder/'saved_model'));converter=tf.lite.TFLiteConverter.from_saved_model(str(folder/'saved_model'));converter.target_spec.supported_ops=[tf.lite.OpsSet.TFLITE_BUILTINS]
            data=converter.convert();path=folder/'fp32.tflite';path.write_bytes(data);graph=s.Model.GetRootAsModel(data,0);sub=graph.Subgraphs(0)
            assert sub.InputsLength()==sub.OutputsLength()==1
            assert sub.Tensors(sub.Inputs(0)).Type()==sub.Tensors(sub.Outputs(0)).Type()==s.TensorType.FLOAT32
            assert sub.Tensors(sub.Inputs(0)).ShapeAsNumpy().tolist()==[1,3,40] and sub.Tensors(sub.Outputs(0)).ShapeAsNumpy().tolist()==[1,1]
            assert all(graph.OperatorCodes(sub.Operators(i).OpcodeIndex()).BuiltinCode()!=s.BuiltinOperator.CUSTOM for i in range(sub.OperatorsLength()))
            initializers=0
            for index in range(1,graph.SubgraphsLength()):
                sg=graph.Subgraphs(index)
                for ti in range(sg.TensorsLength()):
                    tensor=sg.Tensors(ti);buffer=graph.Buffers(tensor.Buffer())
                    if buffer.DataLength():assert tensor.Type()==s.TensorType.FLOAT32 and np.all(buffer.DataAsNumpy().view('<f4')==0);initializers+=1
            assert initializers==6 and all(a.tobytes()==v.numpy().tobytes() for a,v in zip(original,model.variables,strict=True))
            receipt=dict(**w,artifacts=dict(fp32=dict(path=str(path),bytes=len(data),sha256=sha(path))),static_abi_passed=True,parameters_unchanged=True,zero_initializers=6,converter_invocations=1)
            (folder/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');models.append(receipt)
        with RESULT.open('x') as f:f.write(json.dumps(dict(execution_sha256=sha(PLAN),models=models,all_static_passed=True,limits=p['policy']),indent=2)+'\n')
    except Exception:
        (OUT/'failure.log').write_text(traceback.format_exc());raise

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['plan','run']);ap.add_argument('--check-only',action='store_true');a=ap.parse_args();raise SystemExit(plan(a.check_only) if a.action=='plan' else run())
