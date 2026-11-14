"""Instrument four fixed first-request witnesses, preserving original output identity."""
import hashlib,importlib.metadata,json,subprocess,sys,traceback
from pathlib import Path
import numpy as np
R=Path('research/wakeword_p4/results');LOCK=R/'4090-matched-internal-boundaries-execution-v1.json';OUT=Path('.local/wakeword/matched-internal-boundaries-v1')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def lock():
 parent=R/'4090-pooled-matched-screen-execution-v1.json';result=R/'4090-pooled-matched-screen-result-v1.json';p=json.loads(parent.read_text());r=json.loads(result.read_text());assert r['execution_sha256']==sha(parent)
 selected=[next(c for c in r['cases'] if (c['target'],c['condition'],c['request_order'])==(e['target'],e['condition'],0)) for e in p['exports']]
 files=dict(p['files'])
 for f in [parent,result,Path(__file__)]+[Path(c['arrays_path']) for c in selected]:files[str(f)]=sha(f)
 with LOCK.open('x') as f:f.write(json.dumps(dict(files=files,exports=p['exports'],cases=selected,versions={n:importlib.metadata.version(n) for n in ['numpy','ai-edge-litert']},policy='Exactly firstoriginalrequest0 for all4finalmodels, allstartup/completeinput from saved originalfeature arrays, builtinreference preservealltensors, twofresh interpreters eachbackend. Save9conv outputs, FCembedding/weights/bias, prelogit and finalprobability. Must reproduce originalFP32/UINT8 outputs byteexact; otherwise preservefailure. No tensorinjection/modelchange/training/threshold/newmedia. Mechanism witnesses only, not all540coverage or admission.'),indent=2)+'\n')
def metadata(d):
 q=d['quantization_parameters'];return dict(index=int(d['index']),name=d['name'],shape=d['shape'].tolist(),dtype=str(d['dtype']),scales=q['scales'].tolist(),zeros=q['zero_points'].tolist(),axis=int(q['quantized_dimension']))
def run():
 assert subprocess.check_output(['git','show','origin/research-hye-nyx-voice-gpu:'+str(LOCK)])==LOCK.read_bytes();p=json.loads(LOCK.read_text())
 for f,h in p['files'].items():assert sha(f)==h,f
 for n,v in p['versions'].items():assert importlib.metadata.version(n)==v
 OUT.mkdir(exist_ok=False)
 try:
  from ai_edge_litert.interpreter import Interpreter,OpResolverType
  rows=[]
  for c in p['cases']:
   e=next(e for e in p['exports'] if (e['target'],e['condition'])==(c['target'],c['condition']));name=c['target']+'-'+c['condition'];folder=OUT/name;folder.mkdir()
   with np.load(c['arrays_path'],allow_pickle=False) as a:saved={k:a[k].copy() for k in a.files}
   raw=saved['raw'];values=np.clip(np.pad(raw,((0,(-len(raw))%3),(0,0))).astype(np.float32)/25.6,0,26);arrays={};meta={};backend_rows=[]
   for kind,path in [('fp32',e['fp32_path']),('int8',e['path'])]:
    captures=[];expected=saved['fp32'] if kind=='fp32' else saved['uint8']
    for rep in range(2):
     it=Interpreter(model_path=path,num_threads=1,experimental_op_resolver_type=OpResolverType.BUILTIN_REF,experimental_preserve_all_tensors=True);it.allocate_tensors();details={int(d['index']):d for d in it.get_tensor_details()};ops=it._get_ops_details()
     conv=[o for o in ops if o['op_name'] in ['CONV_2D','DEPTHWISE_CONV_2D']];assert len(conv)==9
     fc,=[o for o in ops if o['op_name']=='FULLY_CONNECTED'];logistic,=[o for o in ops if o['op_name']=='LOGISTIC'];a=it.get_input_details()[0];b=it.get_output_details()[0]
     ids={f'encoder_{i:02}':int(o['outputs'][0]) for i,o in enumerate(conv)};ids.update(embedding=int(fc['inputs'][0]),logit=int(logistic['inputs'][0]),probability=int(b['index']))
     if rep==0:
      meta[kind]={k:metadata(details[index]) for k,index in ids.items()}
      for key,index in [('head_weight',int(fc['inputs'][1])),('head_bias',int(fc['inputs'][2]))]:arrays[kind+'_'+key]=it.get_tensor(index).copy();meta[kind][key]=metadata(details[index])
     traces={k:[] for k in ids}
     if kind=='fp32':feed=values
     else:
      scale,zero=a['quantization'];feed=np.clip(np.rint(values/scale)+zero,-128,127).astype(np.int8);assert feed.tobytes()==saved['quantized_input'].tobytes()
     for j in range(0,len(feed),3):
      it.set_tensor(a['index'],feed[None,j:j+3]);it.invoke()
      for key,index in ids.items():traces[key].append(it.get_tensor(index).copy())
     captures.append({k:np.asarray(v) for k,v in traces.items()})
    for key in ids:arrays[kind+'_'+key]=np.stack([x[key] for x in captures])
    repeat=all(x.tobytes()==captures[1][k].tobytes() for k,x in captures[0].items())
    actual=captures[0]['probability'].reshape(-1);identity=actual.dtype==expected.dtype and actual.tobytes()==expected.tobytes();backend_rows.append(dict(backend=kind,repeat_exact=repeat,original_output_exact=identity))
   path=folder/'captures.npz';np.savez(path,**arrays);(folder/'tensor-metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
   row=dict(target=c['target'],condition=c['condition'],request_order=0,source_arrays_sha256=c['arrays_sha256'],captures_path=str(path),captures_sha256=sha(path),metadata_path=str(folder/'tensor-metadata.json'),metadata_sha256=sha(folder/'tensor-metadata.json'),backends=backend_rows,passed=all(b['repeat_exact'] and b['original_output_exact'] for b in backend_rows));rows.append(row)
   (folder/'receipt.json').write_text(json.dumps(row,indent=2)+'\n');print(json.dumps(row),flush=True)
   assert row['passed'],'Instrumented capture differs from original outputs; do not retry'
  (OUT/'result.json').write_text(json.dumps(dict(execution_sha256=sha(LOCK),cases=rows,all_passed=all(r['passed'] for r in rows),limits=p['policy']),indent=2)+'\n')
 except Exception:
  (OUT/'failure.log').write_text(traceback.format_exc());raise
if __name__=='__main__':{'lock':lock,'run':run}[sys.argv[1]]()
