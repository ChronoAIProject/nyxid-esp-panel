"""Prospective matched fourfinal exports with explicit real-zero startup."""
import argparse,importlib.metadata,json
from pathlib import Path
import numpy as np
from plan_hey_broad_trained_quantization_v1 import sha
from run_hey_broad_trained_quantization_v1 import ROOT,published
PLAN=ROOT/'results/4090-hey-broad-paired-qat-export-plan-v1.json'
LOCK=ROOT/'results/4090-hey-broad-paired-qat-export-execution-v1.json'
RESULT=ROOT/'results/4090-hey-broad-paired-qat-export-result-v1.json'
OUT=Path('.local/wakeword/hey-broad-paired-qat-export-v1')
def verify(files):
 for p,h in files.items():assert sha(p)==h,p
def plan():
 parent=ROOT/'results/4090-hey-broad-paired-qat-formal-execution-v1.json';base=json.loads(parent.read_text());files=dict(base['files'])
 for p in [parent,Path(__file__),Path(__file__).with_name('audit_hey_broad_paired_qat_formal_v1.py'),Path(__file__).with_name('probe_hey_broad_bounded_float_v1.py'),Path(__file__).with_name('probe_hey_context_transfer_shapes_v1.py'),ROOT/'results/4090-hey-broad-ptq-material-result-v1.json']:files[str(p)]=sha(p)
 record=dict(files=files,versions=base['versions'],policy='Exactly four fixedfinal10000 H5 after independent full training audit. Identical bounded export/sixzero states/INT8variablequantization options and original representative inputs+order, no source resets/retry/selection. Verify original exported initializer bytes represent realzero. New controlled startup contract, original primed artifacts retained.',limits='Training/export/static identities only, original270numeric gates still required; no freshdev/acceptance/promotion.')
 with PLAN.open('x') as f:f.write(json.dumps(record,indent=2)+'\n')
def lock():
 published(PLAN);p=json.loads(PLAN.read_text());verify(p['files']);files=dict(p['files']);files[str(PLAN)]=sha(PLAN)
 result=ROOT/'results/4090-hey-broad-paired-qat-formal-result-v1.json';audit=ROOT/'results/4090-hey-broad-paired-qat-formal-independent-audit-v1.json'
 for f in [result,audit]:published(f);files[str(f)]=sha(f)
 r=json.loads(result.read_text());a=json.loads(audit.read_text());assert r['all_passed'] and len(r['cases'])==len(a['cases'])==4 and a['result_sha256']==sha(result)
 weights=[]
 for c in r['cases']:
  final=c['final_repeat'];assert c['steps']==final['steps']==10000 and sha(final['weights_path'])==final['weights_sha256'];files[final['weights_path']]=final['weights_sha256'];weights.append(dict(target=c['target'],condition=c['condition'],path=final['weights_path'],sha256=final['weights_sha256']))
 material=json.loads((ROOT/'results/4090-hey-broad-ptq-material-result-v1.json').read_text())['arms']
 for m in material:assert sha(m['path'])==m['sha256'];files[m['path']]=m['sha256']
 with LOCK.open('x') as f:f.write(json.dumps(dict(files=files,versions=p['versions'],weights=weights,material=material),indent=2)+'\n')
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
    for weight in locked['weights']:
        target=weight['target'];condition=weight['condition']
        dest=OUT/f'{target}-{condition}';dest.mkdir()
        original=helper.create(225)
        original.load_weights(weight['path'])
        for layer in original.layers:
            cell=getattr(layer,'cell',layer)
            if isinstance(cell,tf.keras.layers.BatchNormalization):layer.trainable=cell.trainable=False
        initial=[v.numpy().copy() for v in original.variables]
        model=bounded(original,tf)
        assert all(np.array_equal(a,v.numpy()) for a,v in zip(initial,original.variables,strict=True))
        stream=utils.to_streaming_inference(model,{'stride':3,'spectrogram_length':225},modes.Modes.STREAM_INTERNAL_STATE_INFERENCE)
        stream(np.zeros((1,3,40),np.float32),training=False)
        states=[v for v in stream.variables if 'states' in v.name]
        assert len(states)==6
        for state in states:state.assign(tf.zeros_like(state))
        assert all(np.count_nonzero(state.numpy())==0 for state in states)
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
        init=graph.Subgraphs(1);initializers=[]
        for index in range(init.TensorsLength()):
            tensor=init.Tensors(index);buffer=graph.Buffers(tensor.Buffer())
            if buffer.DataLength():
                quant=tensor.Quantization();assert quant is not None and quant.ScaleLength()==1 and tensor.Type()==schema.TensorType.INT8
                code=buffer.DataAsNumpy().view(np.int8);zero=int(quant.ZeroPoint(0));assert np.all(code==zero),'Nonzero exported initializer'
                initializers.append(dict(shape=tensor.ShapeAsNumpy().tolist(),bytes=len(code),zero=zero,scale=float(quant.Scale(0)),all_real_zero=True))
        assert len(initializers)==6
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
        row=dict(target=target,condition=condition,original_weight_sha256=weight['sha256'],initializers=initializers,path=str(path),bytes=len(data),sha256=sha(path),ops=ops,tensors=tensors,fc_relu6_fused=head_fused,logistic_input_scale=logits[0]['quantization'][0],static_gate_passed=bool(head_fused and 0<logits[0]['quantization'][0]<=16/255+1e-6))
        rows.append(row)
        with (OUT/'arms.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        print(json.dumps(dict(target=target,bytes=len(data),static_gate_passed=row['static_gate_passed'])),flush=True)
    with RESULT.open('x') as f:f.write(json.dumps(dict(schema='hey-broad-paired-qat-export-result-v1',execution_sha256=sha(LOCK),arms=rows,limits=json.loads(PLAN.read_text())['limits']),indent=2)+'\n')

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['plan','lock','run']);a=p.parse_args();{'plan':plan,'lock':lock,'run':run}[a.action]()
