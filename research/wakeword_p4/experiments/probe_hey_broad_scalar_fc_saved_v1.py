"""Check scalar FC primitive on every original saved recursive embedding; no invoke."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
R=Path('research/wakeword_p4/results');P=R/'4090-hey-broad-scalar-fc-saved-plan-v1.json'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def plan():
 recursive=R/'4090-hey-broad-consistency-fixed-grid-recursive-result-v1.json';audit=R/'4090-hey-broad-consistency-fixed-grid-recursive-independent-audit-v1.json';compiled=R/'4090-hey-broad-consistency-fixed-grid-compile-result-v1.json';check=R/'4090-hey-broad-scalar-fc-ste-check-v1.json'
 from run_hey_broad_trained_quantization_v1 import published
 for p in (recursive,audit,compiled,check):published(p)
 data=json.loads(recursive.read_text());assert len(data['cases'])==540 and data['all_encoder_fc_exact'];assert json.loads(audit.read_text())['result_sha256']==sha(recursive)
 files={str(p):sha(p) for p in (recursive,audit,compiled,check,Path(__file__).resolve().relative_to(Path.cwd()),Path(__file__).with_name('hey_broad_scalar_fc_ste_v1.py'),Path(__file__).with_name('hey_broad_gpu_training_factory_v1.py'))}
 for c in data['cases']:assert sha(c['arrays_path'])==c['arrays_sha256'];files[c['arrays_path']]=c['arrays_sha256']
 for arm in json.loads(compiled.read_text())['arms']:assert sha(arm['path'])==arm['sha256'];files[arm['path']]=arm['sha256']
 with P.open('x') as f:json.dump(dict(files=files,recursive=str(recursive),compiled=str(compiled),policy='All540 original saved embedding64/actualFC65, all4final models. Metadata allocation only, no interpreter invoke. Reconstruct int64 centered accumulator including actual bias, int32 guard. New scalar primitive forward exact vs original actualFC65, twice repeat and weighted declared-gradient analytic equality. No model changes/optimizer/freshdev/selection or reinterpretation of original failures. Stop on mismatch; no tolerance or retry.'),f,indent=2);f.write('\n')
def run():
 from run_hey_broad_trained_quantization_v1 import published
 published(P);lock=json.loads(P.read_text())
 for p,h in lock['files'].items():assert sha(p)==h,p
 import hey_broad_gpu_training_factory_v1 as helper
 from hey_broad_scalar_fc_ste_v1 import scalar_fc_ste
 from ai_edge_litert.interpreter import Interpreter,OpResolverType
 tf=helper.tf;data=json.loads(Path(lock['recursive']).read_text());reports=[];seen=set()
 for arm in json.loads(Path(lock['compiled']).read_text())['arms']:
  i=Interpreter(model_path=arm['path'],experimental_op_resolver_type=OpResolverType.BUILTIN_REF,num_threads=1);i.allocate_tensors();d={v['index']:v for v in i.get_tensor_details()};op=next(o for o in i._get_ops_details() if o['op_name']=='FULLY_CONNECTED');xi,wi,bi=map(int,op['inputs']);yi=int(op['outputs'][0]);assert (xi,yi)==(64,65)
  ins,iz=d[xi]['quantization'];ws,wz=d[wi]['quantization'];os,oz=d[yi]['quantization'];assert ws>0
  weight=i.get_tensor(wi).astype(np.int64)-wz;bias=i.get_tensor(bi).astype(np.int64);multiplier=ins*ws/os;low=max(-128,oz);high=min(127,int(np.rint(6/os))+oz)
  for c in data['cases']:
   if (c['target'],c['condition'])!=(arm['target'],arm['condition']):continue
   key=(c['target'],c['condition'],c['request_order']);assert key not in seen;seen.add(key)
   with np.load(c['arrays_path'],allow_pickle=False) as a:
    embeddings=a['64'].reshape(-1,1800).astype(np.int64);expected=a['actual_65'].reshape(-1,1);acc=(embeddings-iz)@weight.T+bias;assert acc.min()>=-(1<<31) and acc.max()<(1<<31)
    upstream=np.linspace(-1,1,len(acc),dtype=np.float64).reshape(-1,1);identities=[]
    for repeat in range(2):
     value=tf.constant(acc.astype(np.float64),tf.float64)
     with tf.GradientTape() as tape:
      tape.watch(value);output=scalar_fc_ste(value,multiplier,oz,low,high,os,tf);loss=tf.reduce_sum(output*upstream)
     gradient=tape.gradient(loss,value).numpy();codes=np.rint(output.numpy()/os+oz).astype(np.int64);assert np.array_equal(codes,expected),(key,'forward')
     real=acc.astype(np.float64)*multiplier+oz;expected_gradient=upstream*(multiplier*os)*((real>=low)&(real<=high));assert np.array_equal(gradient,expected_gradient),(key,'gradient')
     identities.append(hashlib.sha256(codes.tobytes()+gradient.tobytes()).hexdigest())
    assert identities[0]==identities[1]
    reports.append(dict(target=c['target'],condition=c['condition'],request_order=c['request_order'],steps=len(acc),accumulator_sha256=hashlib.sha256(acc.tobytes()).hexdigest(),forward_exact=True,declared_gradient_exact=True,repeat_exact=True,signature=identities[0]))
 assert len(reports)==540
 with (R/'4090-hey-broad-scalar-fc-saved-result-v1.json').open('x') as f:json.dump(dict(plan_sha256=sha(P),cases=reports,all_passed=True,model_invocations=0,optimizer_updates=0,limits=lock['policy']),f,indent=2);f.write('\n')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['plan','run']);a=p.parse_args();plan() if a.action=='plan' else run()
