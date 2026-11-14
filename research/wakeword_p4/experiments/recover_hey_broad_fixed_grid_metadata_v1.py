"""Read-only recovery after compiler report serialization failure; never recompiles/invokes."""
import json, hashlib
from pathlib import Path
import numpy as np
from ai_edge_litert import schema_py_generated as schema
from compile_hey_broad_fixed_grid_v1 import structural_hash, sha, P, R, OUT

def main():
 lock=json.loads(P.read_text()); failed=R/'4090-hey-broad-fixed-grid-compile-result-v1.json'
 assert failed.read_bytes()==b''
 log=Path('.local/wakeword/hey-broad-fixed-grid-compile-v1.log')
 assert 'TypeError: Object of type int32 is not JSON serializable' in log.read_text()
 arms=[]
 for arm in lock['arms']:
  path=OUT/f'{arm["target"]}-{arm["condition"]}'/'hey_nyx.tflite'
  template=Path(arm['template']['path']); assert sha(template)==arm['template']['sha256']
  assert sha(arm['weights'])==lock['files'][arm['weights']]
  a=schema.ModelT.InitFromObj(schema.Model.GetRootAsModel(template.read_bytes(),0))
  b=schema.ModelT.InitFromObj(schema.Model.GetRootAsModel(path.read_bytes(),0))
  assert structural_hash(a)==structural_hash(b)
  allowed=set(); constants=[]
  for op in arm['template']['ops']:
   if op['op'] not in ['CONV_2D','DEPTHWISE_CONV_2D','FULLY_CONNECTED']: continue
   for index in op['inputs'][1:]:
    t=b.subgraphs[0].tensors[index]; allowed.add(int(t.buffer)); raw=bytes(b.buffers[t.buffer].data)
    constants.append(dict(index=int(index),buffer=int(t.buffer),shape=[int(x) for x in t.shape],code_sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw)))
  initializers=[]
  for index,t in enumerate(b.subgraphs[1].tensors):
   raw=b.buffers[t.buffer].data
   if raw is None or len(raw)==0: continue
   assert t.type==schema.TensorType.INT8 and len(t.quantization.zeroPoint)==1
   zero=int(t.quantization.zeroPoint[0]); assert np.all(np.frombuffer(bytes(raw),np.int8)==zero)
   allowed.add(int(t.buffer)); initializers.append(dict(index=index,buffer=int(t.buffer),shape=[int(x) for x in t.shape],zero=zero,all_real_zero=True))
  assert len(constants)==20 and len(initializers)==6 and len(allowed)==26
  changed=[]
  for i,(x,y) in enumerate(zip(a.buffers,b.buffers,strict=True)):
   if not np.array_equal(x.data,y.data):
    assert i in allowed; changed.append(i)
  arms.append(dict(target=arm['target'],condition=arm['condition'],path=str(path),bytes=path.stat().st_size,sha256=sha(path),template_sha256=sha(template),final_h5_sha256=sha(arm['weights']),structure_sha256=structural_hash(b),original_grids_exact=True,only_allowed_buffers_changed=True,changed_buffers=changed,constants=constants,initializers=initializers))
 record=dict(status='recovered_metadata_not_original_success_report',plan_sha256=sha(P),original_failure=dict(path=str(failed),bytes=0,sha256=sha(failed),error='TypeError: Object of type int32 is not JSON serializable',log_sha256=sha(log)),arms=arms,limits='Read existing bytes only; no compiler/converter/model invoke. Independently verifies structure/grids, allowed buffer changes and six realzero initializers. Does not independently prove H5-to-code quantization, runtime parity or numerical/quality admission. Original run assertions remain log evidence, not reconstructed assertions.')
 dest=R/'4090-hey-broad-fixed-grid-recovered-metadata-v1.json'
 with dest.open('x') as f:json.dump(record,f,indent=2);f.write('\n')
 print(json.dumps(dict(recovered=str(dest),sha256=sha(dest),models=len(arms))))
if __name__=='__main__':main()
