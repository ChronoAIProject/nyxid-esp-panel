"""Fixed complete-input changed-grid encoder parity; not training graph parity."""
import hashlib,json,subprocess,sys,traceback
from pathlib import Path
import numpy as np
from probe_hey_broad_integer_requant_v2 import fixed
R=Path('research/wakeword_p4/results');P=R/'4090-learned-grid-encoder-parity-execution-v1.json';OUT=Path('.local/wakeword/learned-grid-encoder-parity-v1')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def lock():
 cp=R/'4090-learned-grid-compile-witness-v2.json';sp=R/'4090-matched-internal-boundaries-execution-v1.json';compiled=json.loads(cp.read_text());source=json.loads(sp.read_text());files={str(p):sha(p) for p in (cp,sp,Path(__file__),Path(__file__).with_name('probe_hey_broad_integer_requant_v2.py'))}
 for c in compiled['cases']:files[c['path']]=sha(c['path'])
 for c in source['cases']:files[c['arrays_path']]=sha(c['arrays_path'])
 P.write_text(json.dumps(dict(files=files,compiled=compiled['cases'],sources=source['cases'],policy='All12 fourmodels x fixed factors1/.5/2, originalrequest0 complete quantized input/allstartup. Twofresh builtinref interpreters and independent recursive int64 histories each. Compare all9encoder tensors exact, save final UINT8 repeat and baseline originaloutput identity. Reuse original verified requant oracle, actual changedgrid constants. No FC/sigmoid oracle or traininggraph/recognition qualification claim; no optimization/recalibration/newdata.'),indent=2)+'\n')
def run():
 assert subprocess.check_output(['git','show','origin/research-hye-nyx-voice-gpu:'+str(P)])==P.read_bytes();p=json.loads(P.read_text())
 for f,h in p['files'].items():assert sha(f)==h
 from ai_edge_litert.interpreter import Interpreter,OpResolverType
 from ai_edge_litert import schema_py_generated as S
 OUT.mkdir(exist_ok=False);rows=[]
 try:
  for c in p['compiled']:
   source=next(x for x in p['sources'] if (x['target'],x['condition'])==(c['target'],c['condition']))
   with np.load(source['arrays_path'],allow_pickle=False) as a:q=a['quantized_input'].copy();baseline=a['uint8'].copy()
   meta=Interpreter(model_path=c['path'],experimental_op_resolver_type=OpResolverType.BUILTIN_REF,num_threads=1);meta.allocate_tensors();ds={x['index']:x for x in meta.get_tensor_details()};ops=[x for x in meta._get_ops_details() if x['op_name'] in ('CONV_2D','DEPTHWISE_CONV_2D')];assert len(ops)==9
   constants=[]
   for op in ops:
    ix,wi,bi=map(int,op['inputs']);oy=int(op['outputs'][0]);ins,iz=ds[ix]['quantization'];outs,oz=ds[oy]['quantization'];w=meta.get_tensor(wi).astype(np.int64);b=meta.get_tensor(bi).astype(np.int64);ws=ds[wi]['quantization_parameters']['scales'].astype(np.float64)
    constants.append((op['op_name'],oy,w,b,iz,oz,ins*ws/outs))
   model=S.Model.GetRootAsModel(Path(c['path']).read_bytes(),0);g=model.Subgraphs(1);initial={}
   for ti in range(g.TensorsLength()):
    t=g.Tensors(ti);buf=model.Buffers(t.Buffer())
    if buf.DataLength():
     shape=tuple(t.ShapeAsNumpy());initial[int(shape[1])]=buf.DataAsNumpy().view(np.int8).copy().reshape(shape[1],shape[-1]).astype(np.int64)
   assert set(initial)=={2,4,8,12,20,29}
   def calc(x,config):
    kind,oy,w,b,iz,oz,m=config;v=x-iz
    acc=(np.sum(v*w[0,:,0,:],axis=0) if kind=='DEPTHWISE_CONV_2D' else np.einsum('tc,otc->o',v,w[:,:,0,:]))+b
    assert acc.min()>=-(1<<31) and acc.max()<(1<<31)
    values=np.array([int(fixed(np.asarray(a,np.int64),s,False)) for a,s in zip(acc,m,strict=True)])+oz
    return np.clip(values,max(-128,oz) if kind=='CONV_2D' else -128,127)[None,:]
   runs=[];mismatch={str(x[1]):0 for x in constants}
   for repeat in range(2):
    state={k:v.copy() for k,v in initial.items()};it=Interpreter(model_path=c['path'],experimental_op_resolver_type=OpResolverType.BUILTIN_REF,num_threads=1,experimental_preserve_all_tensors=True);it.allocate_tensors();out=it.get_output_details()[0]['index'];trace={str(x[1]):[] for x in constants};trace['uint8']=[]
    for at in range(0,len(q),3):
     x=np.concatenate([state[2],q[at:at+3].astype(np.int64)]);state[2]=x[-2:].copy();x=calc(x,constants[0]);observed=[x]
     for j,h in enumerate((4,8,12,20)):
      v=np.concatenate([state[h],x]);state[h]=v[-h:].copy();x=calc(v,constants[1+2*j]);observed.append(x);x=calc(x,constants[2+2*j]);observed.append(x)
     it.set_tensor(0,q[None,at:at+3]);it.invoke();trace['uint8'].append(it.get_tensor(out).copy())
     for config,expected in zip(constants,observed,strict=True):
      index=config[1];actual=it.get_tensor(index).reshape(expected.shape);mismatch[str(index)]+=int(np.count_nonzero(actual!=expected));trace[str(index)].append(actual.copy());trace.setdefault('oracle_'+str(index),[]).append(expected.astype(np.int8))
    runs.append({k:np.stack(v) for k,v in trace.items()})
   repeat_exact=all(runs[0][k].tobytes()==runs[1][k].tobytes() for k in runs[0]);baseline_exact=c['factor']!=1 or runs[0]['uint8'].tobytes()==baseline.tobytes()
   dest=OUT/(c['target']+'-'+c['condition']+'-'+str(c['factor'])+'.npz');np.savez_compressed(dest,**{k:np.stack([r[k] for r in runs]) for k in runs[0]})
   row=dict(target=c['target'],condition=c['condition'],factor=c['factor'],model_sha256=c['sha256'],input_source_sha256=source['arrays_sha256'],mismatch_both_repeats=mismatch,repeat_exact=repeat_exact,baseline_output_exact=baseline_exact,arrays_path=str(dest),arrays_sha256=sha(dest),passed=not any(mismatch.values()) and repeat_exact and baseline_exact);rows.append(row);print(json.dumps(row),flush=True)
  result=dict(execution_sha256=sha(P),cases=rows,all_passed=all(x['passed'] for x in rows),limits=p['policy']);(R/'4090-learned-grid-encoder-parity-result-v1.json').write_text(json.dumps(result,indent=2)+'\n');assert result['all_passed']
 except Exception:
  (OUT/'failure.log').write_text(traceback.format_exc());raise
if __name__=='__main__':{'lock':lock,'run':run}[sys.argv[1]]()
