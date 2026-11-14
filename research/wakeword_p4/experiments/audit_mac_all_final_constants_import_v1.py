"""Linux independent saved-array/FlatBuffer audit; never imports/runs Mac code."""
import hashlib,io,json,tarfile
from pathlib import Path
import numpy as np
from ai_edge_litert import schema_py_generated as schema
R=Path('research/wakeword_p4/results')
A=Path('.local/wakeword/mac-all-final-constant-capture-v2/original.tar.gz')
def digest(b):return hashlib.sha256(b).hexdigest()
def main():
 assert digest(A.read_bytes())=='46f535aee20e4a0889449a2c9a8885c8a0d83aa42a4c3c888b878a0be2b5fb92' and A.stat().st_size==7731984
 rows=[];datasets={}
 with tarfile.open(A) as t:
  read=lambda n:t.extractfile(n).read()
  mb=read('manifest.json');manifest=json.loads(mb)
  assert len(manifest['files'])==18 and len(t.getnames())==19 and set(t.getnames())=={'manifest.json'}|{x['name'] for x in manifest['files']}
  for x in manifest['files']:
   b=read(x['name']);assert t.getmember(x['name']).isfile() and len(b)==x['bytes'] and digest(b)==x['sha256']
  pb=read('source/results/mac-all-final-constant-capture-plan-v2.json');plan=json.loads(pb)
  assert digest(read('source/experiments/diagnose_all_final_constant_capture_mac_v2.py'))==plan['script_sha256']
  for name,h in plan['source_sha256'].items():assert digest(Path(name).read_bytes())==h,name
  templates=json.loads((R/'4090-hey-broad-consistency-fixed-grid-compile-plan-v1.json').read_text())['arms']
  for setting in ['default','no-fold']:
   report=json.loads(read(setting+'/report.json'));assert report['plan_sha256']==digest(pb) and report['model_invocations']==report['optimizer_updates']==0 and len(report['cases'])==4
   for arm in plan['arms']:
    key=arm['target'],arm['condition'];c,=[x for x in report['cases'] if (x['target'],x['condition'])==key]
    assert c['model_sha256']==arm['model_sha256'] and c['final_h5_sha256']==arm['weights_sha256']
    raw=Path(arm['model_path']).read_bytes();assert digest(raw)==arm['model_sha256'] and digest(Path(arm['weights_path']).read_bytes())==arm['weights_sha256']
    model=schema.ModelT.InitFromObj(schema.Model.GetRootAsModel(raw,0));template,=[x['template'] for x in templates if (x['target'],x['condition'])==key]
    ops=[x for x in template['ops'] if x['op'] in ['CONV_2D','DEPTHWISE_CONV_2D','FULLY_CONNECTED']];assert len(ops)==10
    b=read(setting+'/'+c['arrays']);assert digest(b)==c['arrays_sha256']
    with np.load(io.BytesIO(b),allow_pickle=False) as z:arrays={k:z[k] for k in z.files}
    assert len(arrays)==360 and all(np.isfinite(v).all() for v in arrays.values());datasets[(setting,*key)]=arrays
    count=0
    for method in plan['methods']:
     for op in ops:
      for bias,index in enumerate(op['inputs'][1:]):
       tensor=model.subgraphs[0].tensors[index];assert tensor.type==(schema.TensorType.INT32 if bias else schema.TensorType.INT8)
       actual=np.frombuffer(bytes(model.buffers[tensor.buffer].data),dtype='<i4' if bias else 'i1').reshape(tensor.shape)
       prefix=f'{method}_tensor{index}_';value,dequant,q,codes,compiled,savedscale=[arrays[prefix+x] for x in ['value','dequant','quotient','codes','compiled','scale']]
       scale=np.asarray(tensor.quantization.scale,np.float32);shape=[len(scale)] if bias else [1]*value.ndim
       if not bias:shape[-2 if op['op']=='DEPTHWISE_CONV_2D' else -1]=len(scale)
       scale=scale.reshape(shape);zero=np.asarray(tensor.quantization.zeroPoint,np.float64).reshape(shape);assert np.array_equal(scale,savedscale)
       from_d=np.rint(dequant.astype(float)/scale.astype(float)+zero).astype(np.int64);lo,hi=(-(2**31),2**31-1) if bias else (-127,127);from_q=np.clip(np.rint(q.astype(float))+zero,lo,hi).astype(np.int64)
       def layout(v):return v if bias else v.T if op['op']=='FULLY_CONNECTED' else np.transpose(v,(3,0,1,2))
       assert np.array_equal(codes,layout(from_d)) and np.array_equal(codes,layout(from_q)) and np.array_equal(codes,actual) and compiled.dtype==actual.dtype and np.array_equal(compiled,actual)
       observation,=[x for x in c['observations'] if (x['method'],x['index'])==(method,index)]
       assert observation['unequal']==0 and observation['differences']==[] and observation['values']==codes.size and observation['bias']==bool(bias);count+=1
    assert count==len(c['observations'])==60;rows.append(dict(setting=setting,target=key[0],condition=key[1],constant_comparisons=count,all_codes_equal_local_flatbuffer=True))
  changed={}
  for arm in plan['arms']:
   key=arm['target'],arm['condition'];a=datasets[('default',*key)];b=datasets[('no-fold',*key)];assert set(a)==set(b)
   for k in a:
    n=int(np.count_nonzero(a[k]!=b[k]))
    if n:assert k.startswith('graph_captured_scale_') and k.endswith('_quotient');changed['/'.join(key)+'/'+k]=n
  assert sum(changed.values())==39304
  for name,data in [('manifest',mb),('plan',pb)]:
   with (R/f'4090-mac-all-final-constants-original-{name}-v1.json').open('xb') as f:f.write(data)
 result=dict(archive_sha256=digest(A.read_bytes()),payloads_verified=18,constant_comparisons=sum(x['constant_comparisons'] for x in rows),cases=rows,cross_setting_changed_quotient_elements=sum(changed.values()),changed_arrays=changed,model_invocations=0,limits='Independent saved arrays and original Linux FlatBuffer bytes only. Does not independently verify unsaved repeats/variable snapshots or reproduce Mac execution. Explicit Tensor boundary/isolated arm64 graphs do not overturn original GPU full-network failure; no recognition/P4 qualification.')
 with (R/'4090-mac-all-final-constants-linux-cross-audit-v1.json').open('x') as f:f.write(json.dumps(result,indent=2)+'\n')
 print(json.dumps(dict(constant_comparisons=result['constant_comparisons'],changed_quotient_elements=sum(changed.values()),all_codes_exact=True)))
if __name__=='__main__':main()
