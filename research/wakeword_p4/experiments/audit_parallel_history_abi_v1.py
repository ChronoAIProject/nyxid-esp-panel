"""Authenticate exact third-model ABI; independent host metadata, no invokes."""
import os
os.environ.update(OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2',TF_NUM_INTRAOP_THREADS='2',TF_NUM_INTEROP_THREADS='1',TF_CPP_MIN_LOG_LEVEL='3',TF_ENABLE_ONEDNN_OPTS='0')
import hashlib,json,subprocess,sys
from pathlib import Path
R=Path('research/wakeword_p4/results');PLAN=R/'4090-parallel-history-abi-plan-v1.json'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def lock():
 paths=[Path(__file__),Path(__file__).with_name('audit_c161_binary40_static_abi_v1.py'),R/'4090-c161-binary40-static-abi-original-v1.json',R/'4090-c161-binary40-static-abi-review-v1.json',R/'4090-frozen-history-teacher-teacher-export-original-v1.json',Path('.local/wakeword/frozen-history-teacher-int8-v1/teacher/int8.tflite')]
 for m in read(paths[2])['models'].values():paths.append(Path(m['path']))
 with PLAN.open('x') as f:json.dump(dict(schema='parallel-history-abi-plan-v1',files={str(p):sha(p) for p in paths},policy='Exact original frozen-history-teacher treatment ABI and original export agreement, independent host interpreter metadata check with zero invokes. Preserve original dual ABI evidence. Report cache input/output quantization separately, actual reset code, signatures and op versions; logical payload only, no arena/RAM/speed estimate, reexport, training or policy change.'),f,indent=2);f.write('\n')
def run():
 assert subprocess.check_output(['git','show','origin/research-hye-nyx-voice-gpu:'+str(PLAN)])==PLAN.read_bytes();p=read(PLAN)
 for path,h in p['files'].items():assert sha(path)==h,path
 from audit_c161_binary40_static_abi_v1 import inspect
 import numpy as np
 from ai_edge_litert.interpreter import Interpreter,OpResolverType
 path='.local/wakeword/frozen-history-teacher-int8-v1/teacher/int8.tflite';m=inspect(path,64800,'d92b95278334043ff70c20c6e8f267e13fe83acac7dbcf705cc2177df9130950');export=read(R/'4090-frozen-history-teacher-teacher-export-original-v1.json');sig=m['signatures'][0];pairs=[]
 assert not m['read_variable_states'] and len(m['subgraphs'])==1
 for side in ['inputs','outputs']:
  for name,t in sig[side].items():
   e=export[side][name];assert t['shape']==e['shape'] and t['scales']==[e['scale']] and t['zero_points']==[e['zero']] and t['dtype']=='INT8'
 for i in range(7):
  a=sig['inputs'][f'cache{i}'];b=sig['outputs'][f'cache{i}_out'];assert a['shape']==b['shape'];pairs.append(dict(name=f'cache{i}',input_tensor=a['index'],output_tensor=b['index'],shape=a['shape'],input_bytes=a['logical_bytes'],output_bytes=b['logical_bytes'],input_scale=a['scales'][0],output_scale=b['scales'][0],input_real_zero_code=a['zero_points'][0],output_real_zero_code=b['zero_points'][0],exact_byte_feedback=a['scales']==b['scales'] and a['zero_points']==b['zero_points']))
 it=Interpreter(model_path=path,num_threads=1,experimental_op_resolver_type=OpResolverType.BUILTIN_REF);it.allocate_tensors();details={t['index']:t for t in it.get_tensor_details()};count=0
 for t in m['subgraphs'][0]['tensors']:
  x=details[t['index']];assert list(x['shape'])==t['shape'];assert np.dtype(x['dtype']).itemsize*np.prod(x['shape'])==t['logical_bytes'];assert list(x['quantization_parameters']['scales'])==t['scales'] and list(x['quantization_parameters']['zero_points'])==t['zero_points'];count+=1
 runner=it.get_signature_runner()
 for side,details in [('inputs',runner.get_input_details()),('outputs',runner.get_output_details())]:
  for name,t in sig[side].items():assert details[name]['index']==t['index'] and details[name]['quantization']==(t['scales'][0],t['zero_points'][0])
 old=read(R/'4090-c161-binary40-static-abi-original-v1.json');assert read(R/'4090-c161-binary40-static-abi-review-v1.json')['all_passed']
 for model in old['models'].values():assert sha(model['path'])==model['sha256']
 retained=sum(x['input_bytes'] for x in pairs);assert retained==6544
 summary=dict(model_file_bytes=old['summary']['model_file_bytes']+64800,c161_internal_state_value_bytes=6020,binary40_cache_input_bytes=6544,history_cache_input_bytes=retained,distinct_retained_state_value_bytes=6020+6544+retained,explicit_cache_output_payload_bytes=6544+sum(x['output_bytes'] for x in pairs),both_explicit_cache_feedback_bytes_per_native_call=13088,history_byte_feedback_pairs=sum(x['exact_byte_feedback'] for x in pairs),history_requantized_feedback_pairs=sum(not x['exact_byte_feedback'] for x in pairs))
 out=dict(schema='parallel-history-abi-original-v1',plan_sha256=sha(PLAN),all_passed=True,history_model=m,history_cache_pairs=pairs,summary=summary,independent_host_review=dict(allocate_tensors_passed=True,tensor_sizes_quantizations_checked=count,signature_indices_quantizations_checked=True,invocations=0),operator_versions={**{n:model['operator_versions'] for n,model in old['models'].items()},'history_expert':m['operator_versions']},limits=p['policy'])
 with (R/'4090-parallel-history-abi-original-v1.json').open('x') as f:json.dump(out,f,indent=2);f.write('\n')
 print(json.dumps(dict(summary=summary,cache_pairs=pairs,independent_host_review=out['independent_host_review'])),flush=True)
if __name__=='__main__':{'lock':lock,'run':run}[sys.argv[1]]()
