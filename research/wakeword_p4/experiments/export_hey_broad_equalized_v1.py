"""Exact equalized-channel PTQ with original representative replay."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import numpy as np
from plan_hey_broad_trained_quantization_v1 import sha
from run_hey_broad_trained_quantization_v1 import ROOT,published

PLAN=ROOT/'results/4090-hey-broad-equalized-export-plan-v1.json'
LOCK=ROOT/'results/4090-hey-broad-equalized-export-execution-v1.json'
RESULT=ROOT/'results/4090-hey-broad-equalized-export-result-v1.json'
MATERIAL=ROOT/'results/4090-hey-broad-ptq-material-result-v1.json'
FLOAT=ROOT/'results/4090-hey-broad-equalized-float-result-v1.json'
OUT=Path('.local/wakeword/hey-broad-equalized-export-v1')

def verify(files):
    for p,h in files.items():assert sha(p)==h,p

def plan():
    original=ROOT/'results/4090-hey-broad-trained-quantization-execution-v1.json'
    parent=json.loads(original.read_text());files=dict(parent['files'])
    additions=[original,Path(__file__),Path(__file__).with_name('probe_hey_broad_bounded_float_v1.py'),FLOAT,ROOT/'results/4090-hey-broad-ptq-material-plan-v1.json',Path(__file__).with_name('replay_hey_broad_ptq_material_v1.py')]
    additions += [ROOT/'results'/n for n in ['4090-hey-broad-equalized-weights-receipt-v1.json','4090-hey-broad-equalized-calibration-result-v1.json','4090-hey-broad-equalized-calibration-plan-v1.json','4090-hey-broad-equalized-float-plan-v1.json','4090-hey-broad-channel-ranges-result-v1.json']]
    additions += [Path(__file__).with_name(n) for n in ['equalize_hey_broad_channels_v1.py','probe_hey_broad_equalized_float_v1.py','probe_hey_broad_equalized_calibration_v1.py']]
    for p in additions:files[str(p)]=sha(p)
    record=dict(schema='hey-broad-equalized-export-plan-v1',files=files,models=parent['models'],versions=parent['versions'],
        policy='One export per exact equalized arm after published270float and full7502/15site/sixstate scale controls. Load immutable transformed original H5 from receipt then shared boundedhead. Original PTQ zero/full26+100x75 authenticated arrays, no source resets, same variablequantization/SavedModel/INT8 IO. No optimizer/factor/BN movingstats changes. Record fullgraph/tensor metadata; unchanged270numeric gates remain required.',
        limits='Controlled PTQ structural intervention, not recognition/numerical admission. Original exports preserved; unchanged270 whole-wave gates still required before any fresh development. No newdev/acceptance/threshold access. Narrow static topology gate does not prove upstream states or calibration dynamics.')
    with PLAN.open('x') as f:f.write(json.dumps(record,indent=2)+'\n')

def lock():
    published(PLAN);p=json.loads(PLAN.read_text());verify(p['files'])
    published(MATERIAL);published(FLOAT)
    control=json.loads(FLOAT.read_text());assert len(control['cases'])==270 and control['all_passed']
    material=json.loads(MATERIAL.read_text());assert len(material['arms'])==2
    files=dict(p['files']);files[str(PLAN)]=sha(PLAN);files[str(MATERIAL)]=sha(MATERIAL)
    for row in material['arms']:
        assert row['verified_batches']==120100 and row['shape']==[100,1,225,40]
        assert sha(row['path'])==row['sha256'];files[row['path']]=row['sha256']
    calibration=ROOT/'results/4090-hey-broad-equalized-calibration-result-v1.json'
    published(calibration); check=json.loads(calibration.read_text());assert check['all_passed'] and len(check['cases'])==2
    receipt_path=ROOT/'results/4090-hey-broad-equalized-weights-receipt-v1.json'
    published(receipt_path); receipt=json.loads(receipt_path.read_text())
    for arm in receipt['arms']:
        assert sha(arm['path'])==arm['sha256'];files[arm['path']]=arm['sha256']
    record=dict(schema='hey-broad-equalized-export-execution-v1',files=files,versions=p['versions'],models=p['models'],material=material['arms'],equalized_weights=receipt['arms'])
    with LOCK.open('x') as f:f.write(json.dumps(record,indent=2)+'\n')

def run():
    published(LOCK);locked=json.loads(LOCK.read_text());verify(locked['files'])
    for n,v in locked['versions'].items():assert importlib.metadata.version(n)==v
    OUT.mkdir(exist_ok=False)
    import probe_hey_context_transfer_shapes_v1 as helper
    from probe_hey_broad_bounded_float_v1 import bounded
    tf=helper.tf
    from microwakeword import utils
    from microwakeword.layers import modes
    from tensorflow.lite.python import schema_py_generated as schema
    from ai_edge_litert.interpreter import Interpreter
    names={v:k for k,v in vars(schema.BuiltinOperator).items() if isinstance(v,int)}
    rows=[]
    for target,folder in locked['models'].items():
        dest=OUT/target;dest.mkdir()
        original=helper.create(225)
        weight=next(r for r in locked['equalized_weights'] if r['target']==target)
        original.load_weights(weight['path'])
        for layer in original.layers:
            cell=getattr(layer,'cell',layer)
            if isinstance(cell,tf.keras.layers.BatchNormalization):layer.trainable=cell.trainable=False
        initial=[v.numpy().copy() for v in original.variables]
        model=bounded(original,tf)
        assert all(np.array_equal(a,v.numpy()) for a,v in zip(initial,original.variables,strict=True))
        stream=utils.to_streaming_inference(model,{'stride':3,'spectrogram_length':225},modes.Modes.STREAM_INTERNAL_STATE_INFERENCE)
        stream(np.zeros((1,3,40),np.float32),training=False)
        assert len([v for v in stream.variables if 'states' in v.name])==6
        stream.export(str(dest/'saved_model'))
        source=next(r for r in locked['material'] if r['target']==target)
        material=np.load(source['path'],mmap_mode='r',allow_pickle=False)
        def representative():
            yield [np.zeros((1,3,40),np.float32)]
            yield [np.full((1,3,40),26,np.float32)]
            for value in material:
                for at in range(0,225,3):yield [np.array(value[:,at:at+3],copy=True)]
        converter=tf.lite.TFLiteConverter.from_saved_model(str(dest/'saved_model'))
        converter.optimizations=[tf.lite.Optimize.DEFAULT]
        converter._experimental_variable_quantization=True
        converter.target_spec.supported_ops=[tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type=tf.int8;converter.inference_output_type=tf.uint8
        converter.representative_dataset=representative
        data=converter.convert();path=dest/'hey_nyx.tflite';path.write_bytes(data)
        graph=schema.Model.GetRootAsModel(data,0);sub=graph.Subgraphs(0);ops=[]
        for i in range(sub.OperatorsLength()):
            op=sub.Operators(i);name=names[graph.OperatorCodes(op.OpcodeIndex()).BuiltinCode()]
            row=dict(op=name,inputs=op.InputsAsNumpy().tolist(),outputs=op.OutputsAsNumpy().tolist())
            if name=='FULLY_CONNECTED':
                options=schema.FullyConnectedOptions();table=op.BuiltinOptions();options.Init(table.Bytes,table.Pos)
                row['fused_activation']=int(options.FusedActivationFunction())
            ops.append(row)
        interpreter=Interpreter(model_content=data,num_threads=1);interpreter.allocate_tensors()
        details=interpreter.get_tensor_details();byindex={d['index']:d for d in details}
        logits=[byindex[o['inputs'][0]] for o in ops if o['op']=='LOGISTIC'];assert len(logits)==1
        head_fused=any(o.get('fused_activation')==schema.ActivationFunctionType.RELU6 for o in ops)
        tensors=[dict(index=d['index'],name=d['name'],shape=d['shape'].tolist(),dtype=str(d['dtype']),scale=d['quantization'][0],zero=d['quantization'][1]) for d in details]
        row=dict(target=target,path=str(path),bytes=len(data),sha256=sha(path),ops=ops,tensors=tensors,fc_relu6_fused=head_fused,logistic_input_scale=logits[0]['quantization'][0],static_gate_passed=bool(head_fused and 0<logits[0]['quantization'][0]<=16/255+1e-6))
        rows.append(row)
        with (OUT/'arms.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        print(json.dumps(dict(target=target,bytes=len(data),static_gate_passed=row['static_gate_passed'])),flush=True)
    with RESULT.open('x') as f:f.write(json.dumps(dict(schema='hey-broad-equalized-export-result-v1',execution_sha256=sha(LOCK),arms=rows,limits=json.loads(PLAN.read_text())['limits']),indent=2)+'\n')

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=['plan','lock','run']);args=parser.parse_args()
    {'plan':plan,'lock':lock,'run':run}[args.action]()
