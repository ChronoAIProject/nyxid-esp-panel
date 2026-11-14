"""Saved same-input integer encoder reference, both rounding variants retained."""
import argparse,hashlib,json,math
from pathlib import Path
import numpy as np
R=Path('research/wakeword_p4/results');P=R/'4090-hey-broad-integer-requant-plan-v1.json'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def plan():
 parent=R/'4090-hey-broad-surrogate-residual-plan-v1.json';b=json.loads(parent.read_text());files=dict(b['files'])
 for p in [parent,Path(__file__),R/'4090-hey-broad-surrogate-residual-result-v1.json','.local/wakeword/hey-broad-surrogate-residual-v1/hey_willow-014.npz',*Path('.local/wakeword/litert-reference-v230').glob('*')]:files[str(p)]=sha(p)
 P.write_text(json.dumps(dict(files=files,export=b['export'],policy='All167 steps Willow14 nine encoder operators, actual saved inputs/history and actual exported integer weights/bias. Exact int64 accumulator with int32 bounds assertion. Compare mathematical nearest-even versus source-based double/single fixedpoint rounding, both reported without selecting by candidate gate. Reference sources public LiteRT v2.3.0 tag, not proof of installed binary build flags. No invoke, training, export, new data or admission changes.'),indent=2)+'\n')
def multiplier(m):
 f,s=math.frexp(float(m));q=int(math.floor(f*(1<<31)+.5))
 if q==(1<<31):q//=2;s+=1
 if s< -31:q=0;s=0
 return q,s
def fixed(x,m,single):
 q,s=multiplier(m)
 if single:
  assert -31<=s<=30
  return (x*q+(1<<(30-s)))>>(31-s)
 left=max(s,0);right=max(-s,0);prod=x*(1<<left)*q
 nudge=np.where(prod>=0,1<<30,1-(1<<30));v=prod+nudge
 high=np.where(v>=0,v//(1<<31),-((-v)//(1<<31)))
 mask=(1<<right)-1;rem=high&mask;threshold=(mask>>1)+(high<0)
 return (high>>right)+(rem>threshold)
def run():
 import subprocess
 subprocess.run(['git','diff','--exit-code','HEAD','--',str(P)],check=True);assert subprocess.check_output(['git','ls-files','--',str(P)],text=True).strip()
 lock=json.loads(P.read_text())
 for p,h in lock['files'].items():assert sha(p)==h,p
 from ai_edge_litert.interpreter import Interpreter,OpResolverType
 arm=next(a for a in json.loads(Path(lock['export']).read_text())['arms'] if a['target']=='hey_willow');i=Interpreter(model_path=arm['path'],experimental_op_resolver_type=OpResolverType.BUILTIN_REF,num_threads=1);i.allocate_tensors();d={x['index']:x for x in i.get_tensor_details()}
 with np.load('.local/wakeword/hey-broad-surrogate-residual-v1/hey_willow-014.npz') as a:actual={int(k.split('_')[1]):a[k].copy() for k in a.files if k.startswith('actual_')}
 ops=[o for o in arm['ops'] if o['op'] in ['CONV_2D','DEPTHWISE_CONV_2D']];assert len(ops)==9
 # Actual current inputs plus original zero history, never surrogate states.
 with np.load('.local/wakeword/hey-broad-equalized-screen-v1/hey_willow-014.npz') as a:inputq=a['quantized_input'].astype(np.int64)
 current={47:inputq.reshape(-1,3,40)}
 for n in [50,53,56,59,62]:current[n]=actual[n].reshape(167,-1,actual[n].shape[-1]).astype(np.int64)
 concats={48:(47,2),51:(50,4),54:(53,8),57:(56,12),60:(59,20)}
 reports=[];arrays={}
 for op in ops:
  ix,wi,bi=map(int,op['inputs']);oy=int(op['outputs'][0]);scale,zero=d[ix]['quantization'];os,oz=d[oy]['quantization'];weights=i.get_tensor(wi).astype(np.int64);bias=i.get_tensor(bi).astype(np.int64)
  if ix in concats:
   src,hist=concats[ix];v=current[src];flat=v.reshape(-1,v.shape[-1]);pad=np.pad(flat,((hist,0),(0,0)),constant_values=zero);stride=v.shape[1];x=np.stack([pad[t*stride:t*stride+hist+stride] for t in range(167)])
  else:x=actual[ix].reshape(167,-1,actual[ix].shape[-1]).astype(np.int64)
  x=x-zero
  if op['op']=='DEPTHWISE_CONV_2D':
   w=weights[0,:,0,:];acc=np.sum(x*w[None,:,:],axis=1)+bias
  else:
   w=weights[:,:,0,:];acc=np.einsum('ntc,otc->no',x,w)+bias
  assert acc.min()>=-(1<<31) and acc.max()<(1<<31)
  ws=d[wi]['quantization_parameters']['scales'].astype(np.float64);m=scale*ws/os
  assert len(m)==acc.shape[1]
  lo=max(-128,oz) if op['op']=='CONV_2D' else -128
  outputs={'nearest_even':np.rint(acc*m[None,:]).astype(np.int64)}
  for name,single in [('double_rounding',False),('single_rounding',True)]:outputs[name]=np.stack([fixed(acc[:,j],v,single) for j,v in enumerate(m)],axis=1)
  target=actual[oy].reshape(167,-1).astype(np.int64);metrics={}
  for name,v in outputs.items():
   value=np.clip(v+oz,lo,127);err=np.abs(value-target);metrics[name]=dict(maximum=int(err.max()),unequal=int(np.sum(err!=0)),total=int(err.size));arrays[f'{oy}_{name}']=value
  arrays[f'{oy}_accumulator']=acc;reports.append(dict(index=oy,kind=op['op'],metrics=metrics,accumulator_min=int(acc.min()),accumulator_max=int(acc.max())))
 out=Path('.local/wakeword/hey-broad-integer-requant-v1.npz');assert not out.exists();np.savez_compressed(out,**arrays)
 result=dict(plan_sha256=sha(P),operators=reports,arrays_path=str(out),arrays_sha256=sha(out),limits=lock['policy'],model_invocations=0)
 (R/'4090-hey-broad-integer-requant-result-v1.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(reports,indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['plan','run']);a=p.parse_args();plan() if a.action=='plan' else run()
