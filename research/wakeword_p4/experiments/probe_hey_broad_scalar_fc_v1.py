"""Source-derived scalarFC multiplier precision control on fixed first failure."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
from probe_hey_broad_integer_requant_v2 import fixed,multiplier
R=Path('research/wakeword_p4/results');P=R/'4090-hey-broad-scalar-fc-plan-v1.json'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def plan():
 parent=R/'4090-hey-broad-recursive-integer-plan-v1.json';b=json.loads(parent.read_text());files=dict(b['files']);cases=[json.loads(l) for l in Path('.local/wakeword/hey-broad-recursive-integer-v1/cases.jsonl').read_text().splitlines()];row=next(c for c in cases if not c['all_encoder_fc_exact'])
 for p in [parent,Path(__file__),row['arrays_path'],*map(lambda n:Path('.local/wakeword/litert-reference-v230')/n,['fully_connected.cc','kernel_util.cc','reference_integer_ops_fully_connected.h','common.cc'])]:files[str(p)]=sha(p)
 P.write_text(json.dumps(dict(files=files,case=row,export=b['export'],actual=b['actual'],policy='Fixed first recursive failedcase, original wholeinput twicefresh builtinref. Saved recursive embedding mustexact runtime embedding. Compare scalarFCsource float32 scaleproduct vs original doubleproduct, same original constants/accumulators/double-rounding. Preserve both variants/results. Original270 run not modified/restarted. No optimizer/export/newdata/gate change.'),indent=2)+'\n')
def run():
 import subprocess
 subprocess.run(['git','diff','--exit-code','HEAD','--',str(P)],check=True);assert subprocess.check_output(['git','ls-files','--',str(P)],text=True).strip();lock=json.loads(P.read_text())
 for p,h in lock['files'].items():assert sha(p)==h,p
 from ai_edge_litert.interpreter import Interpreter,OpResolverType
 row=lock['case'];arm=next(a for a in json.loads(Path(lock['export']).read_text())['arms'] if a['target']==row['target']);src=next(c for c in json.loads(Path(lock['actual']).read_text())['cases'] if (c['target'],c['request_order'])==(row['target'],row['request_order']))
 with np.load(row['arrays_path']) as a:embedding=a['64'].reshape(-1,1800).astype(np.int64);old=a['65'].reshape(-1)
 with np.load(src['arrays_path']) as a:q=a['quantized_input'];baseline=a['uint8']
 traces=[]
 for rep in range(2):
  i=Interpreter(model_path=arm['path'],experimental_op_resolver_type=OpResolverType.BUILTIN_REF,num_threads=1,experimental_preserve_all_tensors=True);i.allocate_tensors();d={v['index']:v for v in i.get_tensor_details()};out=i.get_output_details()[0]['index'];fc=[];scores=[]
  for at in range(0,len(q),3):
   i.set_tensor(0,q[None,at:at+3]);i.invoke();assert np.array_equal(i.get_tensor(64).reshape(-1),embedding[at//3]);fc.append(int(i.get_tensor(65)[0,0]));scores.append(int(i.get_tensor(out)[0,0]))
  assert np.array_equal(scores,baseline);traces.append(np.asarray(fc,np.int64))
 assert np.array_equal(*traces)
 ix,wi,bi=next(o for o in arm['ops'] if o['op']=='FULLY_CONNECTED')['inputs'];ins,iz=d[ix]['quantization'];ws,wz=d[wi]['quantization'];os,oz=d[65]['quantization'];assert wz==0
 acc=(embedding-iz)@i.get_tensor(wi).astype(np.int64).T+i.get_tensor(bi).astype(np.int64);acc=acc.reshape(-1)
 multipliers=dict(double_product=ins*ws/os,float32_product=float(np.float32(ins)*np.float32(ws))/os);variants={};arrays=dict(actual=traces[0],accumulator=acc)
 for name,m in multipliers.items():
  v=np.clip(fixed(acc,m,False)+oz,oz,min(127,int(np.rint(6/os))+oz));e=np.abs(v-traces[0]);variants[name]=dict(real_multiplier=m,quantized_multiplier_shift=multiplier(m),unequal=int(np.sum(e!=0)),maximum=int(e.max()),failed_steps=np.flatnonzero(e).tolist());arrays[name]=v
 assert np.array_equal(arrays['double_product'],old)
 outpath=Path('.local/wakeword/hey-broad-scalar-fc-v1.npz');assert not outpath.exists();np.savez(outpath,**arrays)
 result=dict(plan_sha256=sha(P),case=row,variants=variants,actual_twice_reset_exact=True,saved_recursive_embedding_exact=True,original_reference_reproduced=True,arrays_path=str(outpath),arrays_sha256=sha(outpath),limits=lock['policy']);(R/'4090-hey-broad-scalar-fc-result-v1.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(variants,indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['plan','run']);a=p.parse_args();plan() if a.action=='plan' else run()
