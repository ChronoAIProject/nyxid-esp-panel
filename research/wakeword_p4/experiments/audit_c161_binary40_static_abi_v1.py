"""Read frozen FlatBuffers; no inference, conversion, training or arena estimate."""
import os
os.environ.update(OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2',TF_NUM_INTRAOP_THREADS='2',TF_NUM_INTEROP_THREADS='1',TF_CPP_MIN_LOG_LEVEL='3',TF_ENABLE_ONEDNN_OPTS='0')
import hashlib,json,math
from collections import Counter
from pathlib import Path
from tensorflow.lite.python import schema_py_generated as s
R=Path('research/wakeword_p4/results')
OP={v:k for k,v in vars(s.BuiltinOperator).items() if isinstance(v,int)}
TYPE={v:k for k,v in vars(s.TensorType).items() if isinstance(v,int)}
WIDTH={'FLOAT32':4,'FLOAT16':2,'INT32':4,'UINT8':1,'INT64':8,'BOOL':1,'INT16':2,'INT8':1,'FLOAT64':8,'UINT64':8,'UINT32':4,'UINT16':2}
def sha(b):return hashlib.sha256(b).hexdigest()
def inspect(path,n,h):
 raw=Path(path).read_bytes();assert len(raw)==n and sha(raw)==h
 m=s.Model.GetRootAsModel(raw,0)
 def tensor(g,index):
  t=g.Tensors(index);shape=[t.Shape(j) for j in range(t.ShapeLength())];kind=TYPE[t.Type()];q=t.Quantization();buf=m.Buffers(t.Buffer())
  return dict(index=index,name=t.Name().decode(),shape=shape,dtype=kind,logical_bytes=math.prod(shape)*WIDTH[kind] if kind in WIDTH and all(x>=0 for x in shape) else None,buffer_index=t.Buffer(),serialized_buffer_bytes=buf.DataLength(),is_variable=t.IsVariable(),scales=[q.Scale(j) for j in range(q.ScaleLength())] if q else [],zero_points=[q.ZeroPoint(j) for j in range(q.ZeroPointLength())] if q else [])
 graphs=[];versions={};states=[]
 for gi in range(m.SubgraphsLength()):
  g=m.Subgraphs(gi);ops=Counter()
  for i in range(g.OperatorsLength()):
   op=g.Operators(i);code=m.OperatorCodes(op.OpcodeIndex());name=OP[code.BuiltinCode()];ops[name]+=1;versions.setdefault(name,set()).add(code.Version())
   if name=='READ_VARIABLE':
    assert op.OutputsLength()==1
    states.append(dict(subgraph=gi,operator_index=i,handle_tensor=op.Inputs(0),value=tensor(g,op.Outputs(0))))
  graphs.append(dict(index=gi,name=g.Name().decode(),inputs=[tensor(g,g.Inputs(i)) for i in range(g.InputsLength())],outputs=[tensor(g,g.Outputs(i)) for i in range(g.OutputsLength())],operators=ops,tensors=[tensor(g,i) for i in range(g.TensorsLength())]))
 signatures=[]
 for i in range(m.SignatureDefsLength()):
  sd=m.SignatureDefs(i);g=m.Subgraphs(sd.SubgraphIndex())
  signatures.append(dict(key=sd.SignatureKey().decode(),subgraph=sd.SubgraphIndex(),inputs={sd.Inputs(j).Name().decode():tensor(g,sd.Inputs(j).TensorIndex()) for j in range(sd.InputsLength())},outputs={sd.Outputs(j).Name().decode():tensor(g,sd.Outputs(j).TensorIndex()) for j in range(sd.OutputsLength())}))
 return dict(path=path,bytes=n,sha256=h,operator_versions={k:sorted(v) for k,v in versions.items()},subgraphs=graphs,read_variable_states=states,signatures=signatures)
def main():
 a=inspect('.local/wakeword/causal-source-balanced-int8-v1/int8.tflite',60952,'c16154040378b8acccdb57c19917c46cea0a9ea6a39f105004b964ddd093ab81')
 b=inspect('.local/wakeword/binary40-trained-int8-v1/int8.tflite',64800,'454058039dd41e729ce82ac7f33a84edec4af9554615a632e3b33d47de38009c')
 reportpath=R/'4090-binary40-trained-export-original-v1.json';original=json.loads(reportpath.read_text());sig=b['signatures'][0];pairs=[]
 for i in range(7):
  x=sig['inputs'][f'cache{i}'];y=sig['outputs'][f'cache{i}_out'];e=original['inputs'][f'cache{i}'];f=original['outputs'][f'cache{i}_out']
  for got,expected in [(x,e),(y,f)]:assert got['shape']==expected['shape'] and got['scales']==[expected['scale']] and got['zero_points']==[expected['zero']] and got['dtype']=='INT8'
  same=x['scales']==y['scales'] and x['zero_points']==y['zero_points'];assert same and x['shape']==y['shape']
  pairs.append(dict(name=f'cache{i}',input_tensor=x['index'],output_tensor=y['index'],shape=x['shape'],input_bytes=x['logical_bytes'],output_bytes=y['logical_bytes'],scale=x['scales'][0],real_zero_code=x['zero_points'][0],exact_byte_feedback=same))
 assert len(a['read_variable_states'])==6 and len({x['handle_tensor'] for x in a['read_variable_states']})==6
 c161_bytes=sum(x['value']['logical_bytes'] for x in a['read_variable_states']);b40_bytes=sum(x['input_bytes'] for x in pairs);assert (c161_bytes,b40_bytes)==(6020,6544)
 out=dict(schema='c161-binary40-static-abi-original-v1',script_sha256=sha(Path(__file__).read_bytes()),original_b40_export_report_sha256=sha(reportpath.read_bytes()),models=dict(c161=a,binary40=b),binary40_cache_pairs=pairs,summary=dict(c161_internal_state_value_bytes=c161_bytes,binary40_cache_input_bytes=b40_bytes,binary40_cache_output_bytes=sum(x['output_bytes'] for x in pairs),distinct_model_state_value_bytes=c161_bytes+b40_bytes,binary40_feedback_bytes_per_native_call=b40_bytes,model_file_bytes=125752,binary40_cache_zero_code=-128),limitations='Static logical tensor payloads only. Not allocator liveness/aliasing/alignment, scratch, actual arena/heap, model mapping, frontend, task stacks, audio queues, speed or P4 operator support. Do not sum every listed tensor as an arena estimate. B40 input/output tensors coexist logically but actual memory aliases depend on runtime. C161 READ_VARIABLE outputs describe state values, not a measurement of resource allocation. Existing handoff/model bytes unchanged; no inference or independent exposure.')
 with (R/'4090-c161-binary40-static-abi-original-v1.json').open('x') as f:json.dump(out,f,indent=2);f.write('\n')
 print(json.dumps(dict(summary=out['summary'],operator_versions={k:v['operator_versions'] for k,v in out['models'].items()}),indent=2))
if __name__=='__main__':main()
