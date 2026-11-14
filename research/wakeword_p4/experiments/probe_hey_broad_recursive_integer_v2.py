"""Independent recursive integer encoder/FC against actual full270 runtime."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from probe_hey_broad_integer_requant_v2 import fixed
R=Path('research/wakeword_p4/results');P=R/'4090-hey-broad-recursive-integer-plan-v2.json';OUT=Path('.local/wakeword/hey-broad-recursive-integer-v2')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def plan():
 parent=R/'4090-hey-broad-quantized-stream-plan-v1.json';b=json.loads(parent.read_text());files=dict(b['files'])
 for p in [parent,Path(__file__),Path(__file__).with_name("probe_hey_broad_recursive_integer_v1.py"),R/"4090-hey-broad-recursive-integer-result-v1.json",R/"4090-hey-broad-scalar-fc-result-v2.json",*Path(".local/wakeword/litert-reference-v230").glob("*"),Path(__file__).with_name('probe_hey_broad_integer_requant_v2.py'),R/'4090-hey-broad-integer-requant-result-v2.json']:files[str(p)]=sha(p)
 P.write_text(json.dumps(dict(files=files,export=b['export'],actual=b['actual'],policy='All270 original complete waveforms, original saved inputcodes. Independent recursive int64 encoder/FC with actual constants, perchannel conv double rounding, source-reference scalarFC double effective scale with round away from zero, and original model initializer bytes; no injection of actual intermediates. Compare all9encoder outputs/1800embedding/boundedFC each step to actual builtinref preserveall interpreter. Actual finalUINT8 must reproduce saved original. Two fresh independent histories/interpreters per input; compare state hash and all traces exact repeats. Save every integer activation and final actualUINT8 including failures. No optimizer/export/newdata/gate change. Encoder/FC reference only: sigmoid/postFC integer kernels not reimplemented, no recognition/admission claim.'),indent=2)+'\n')
def run():
 import subprocess
 subprocess.run(['git','diff','--exit-code','HEAD','--',str(P)],check=True);assert subprocess.check_output(['git','ls-files','--',str(P)],text=True).strip()
 lock=json.loads(P.read_text())
 for p,h in lock['files'].items():assert sha(p)==h,p
 from ai_edge_litert.interpreter import Interpreter,OpResolverType
 from ai_edge_litert import schema_py_generated as schema
 exports=json.loads(Path(lock['export']).read_text())['arms'];rows=json.loads(Path(lock['actual']).read_text())['cases'];OUT.mkdir(exist_ok=False);cases=[]
 for arm in exports:
  target=arm['target'];meta=Interpreter(model_path=arm['path'],experimental_op_resolver_type=OpResolverType.BUILTIN_REF,num_threads=1);meta.allocate_tensors();details={d['index']:d for d in meta.get_tensor_details()}
  ops=[o for o in arm['ops'] if o['op'] in ['CONV_2D','DEPTHWISE_CONV_2D','FULLY_CONNECTED']];assert len(ops)==10
  model=schema.Model.GetRootAsModel(Path(arm['path']).read_bytes(),0);init=model.Subgraphs(1);initial={}
  for n in range(init.TensorsLength()):
   t=init.Tensors(n);buf=model.Buffers(t.Buffer())
   if buf.DataLength():
    shape=tuple(int(v) for v in t.ShapeAsNumpy());initial[shape[1]]=buf.DataAsNumpy().view(np.int8).copy().reshape(shape[1],shape[-1]).astype(np.int64)
  constants={}
  for o in ops:
   ix,wi,bi=map(int,o['inputs']);oy=int(o['outputs'][0]);w=meta.get_tensor(wi).astype(np.int64);bias=meta.get_tensor(bi).astype(np.int64);ins,inz=details[ix]['quantization'];outs,outz=details[oy]['quantization'];ws=details[wi]['quantization_parameters']['scales'].astype(np.float64)
   constants[oy]=(w,bias,inz,outz,ins*ws/outs)
  def calc(x,o):
   oy=int(o['outputs'][0]);w,b,iz,oz,m=constants[oy];v=x-iz
   if o['op']=='DEPTHWISE_CONV_2D':acc=np.sum(v*w[0,:,0,:],axis=0)+b
   elif o['op']=='FULLY_CONNECTED':acc=w@v.reshape(-1)+b
   else:acc=np.einsum('tc,otc->o',v,w[:,:,0,:])+b
   assert acc.min()>=-(1<<31) and acc.max()<(1<<31)
   value=np.asarray([int(fixed(np.asarray(a,np.int64),s,False)) for a,s in zip(acc,m,strict=True)])+oz
   if o['op']=='FULLY_CONNECTED':
    real=acc.astype(np.float64)*m
    value=np.copysign(np.floor(np.abs(real)+.5),real).astype(np.int64)+oz
   lo=max(-128,oz) if o['op']=='CONV_2D' else max(-128,oz) if o['op']=='FULLY_CONNECTED' else -128
   hi=min(127,int(np.rint(6/details[oy]['quantization'][0]))+oz) if o['op']=='FULLY_CONNECTED' else 127
   return np.clip(value,lo,hi).astype(np.int64)[None,:]
  for order,row in enumerate(r for r in rows if r['target']==target):
   with np.load(row['arrays_path']) as a:q=a['quantized_input'];baseline=a['uint8']
   runs=[];digests=[];metrics={}
   for repeat in range(2):
    states={k:v.copy() for k,v in initial.items()};digest=hashlib.sha256();i=Interpreter(model_path=arm['path'],experimental_op_resolver_type=OpResolverType.BUILTIN_REF,num_threads=1,experimental_preserve_all_tensors=True);i.allocate_tensors();out=i.get_output_details()[0]['index'];traces={};scores=[]
    for at in range(0,len(q),3):
     x=np.concatenate([states[2],q[at:at+3].astype(np.int64)]);states[2]=x[-2:].copy();x=calc(x,ops[0]);observed={50:x}
     for j,hist in enumerate([4,8,12,20]):
      joined=np.concatenate([states[hist],x]);states[hist]=joined[-hist:].copy();x=calc(joined,ops[1+2*j]);observed[int(ops[1+2*j]['outputs'][0])]=x;x=calc(x,ops[2+2*j]);observed[int(ops[2+2*j]['outputs'][0])]=x
     joined=np.concatenate([states[29],x]);states[29]=joined[-29:].copy();observed[64]=joined.reshape(1,-1);observed[65]=calc(joined,ops[-1])
     i.set_tensor(0,q[None,at:at+3]);i.invoke();scores.append(int(i.get_tensor(out)[0,0]))
     for n,v in observed.items():
      actual=i.get_tensor(n).reshape(v.shape).astype(np.int64);err=np.abs(v-actual);m=metrics.setdefault(str(n),dict(unequal=0,maximum=0));m['unequal']+=int(np.sum(err!=0));m['maximum']=max(m['maximum'],int(err.max()));traces.setdefault(str(n),[]).append(v.astype(np.int8))
      if n==65:traces.setdefault('actual_65',[]).append(actual.astype(np.int8))
     for h in [2,4,8,12,20,29]:digest.update(states[h].tobytes())
    assert np.array_equal(scores,baseline),'actual runtime changed original output'
    traces={k:np.stack(v) for k,v in traces.items()};traces['actual_uint8']=np.asarray(scores,np.uint8);runs.append(traces);digests.append(digest.hexdigest())
   assert digests[0]==digests[1] and all(np.array_equal(runs[0][k],runs[1][k]) for k in runs[0])
   path=OUT/f'{target}-{order:03}.npz';np.savez_compressed(path,**runs[0]);case=dict(target=target,request_order=row['request_order'],actual_output_exact=True,twice_output_state_exact=True,all_encoder_fc_exact=all(v['unequal']==0 for v in metrics.values()),metrics_both_repeats=metrics,arrays_path=str(path),arrays_sha256=sha(path),state_trace_sha256=digests[0]);cases.append(case)
   with (OUT/'cases.jsonl').open('a') as f:f.write(json.dumps(case)+'\n')
   print(json.dumps(dict(target=target,completed=order+1,exact=case['all_encoder_fc_exact'])),flush=True)
 result=dict(plan_sha256=sha(P),cases=cases,all_encoder_fc_exact=all(c['all_encoder_fc_exact'] for c in cases),limits=lock['policy']);assert len(cases)==270
 (R/'4090-hey-broad-recursive-integer-result-v2.json').write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['plan','run']);a=p.parse_args();plan() if a.action=='plan' else run()
