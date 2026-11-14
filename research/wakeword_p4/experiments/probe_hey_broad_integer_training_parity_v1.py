"""Full actual batch integer graph parity with independent NumPy convolution."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
from probe_hey_broad_integer_requant_v2 import fixed
R=Path('research/wakeword_p4/results');P=R/'4090-hey-broad-integer-training-parity-plan-v1.json';O=Path('.local/wakeword/hey-broad-integer-training-parity-v1')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def plan():
 parent=R/'4090-hey-broad-integer-encoder-training-plan-v1.json';b=json.loads(parent.read_text());files=dict(b['files']);result=R/'4090-hey-broad-integer-encoder-training-result-v1.json'
 for p in [parent,result,Path(__file__),Path(__file__).with_name('hey_broad_integer_training_observer_v1.py')]:files[str(p)]=sha(p)
 for c in json.loads(result.read_text())['cases']:files[c['arrays_path']]=c['arrays_sha256']
 P.write_text(json.dumps(dict(files=files,training_plan=str(parent),training_result=str(result),policy='Original four full128x225batches. Observational copy adds tensor returns only, mustexact reproduce saved rawlogits and diagnostic probability, parameters unchanged. All quantized modelweights/bias mustequal deployed integer constants. Independent NumPy int64 valid conv/depthwise/slidingwindows from actual inputcodes/actual weights, original perchannel double-rounding. Compare all9encoder layercodes allsamples/allpositions, no injection from traininggraph. Save graph/reference arrays and hashes including failures. No optimizer/export/newdata/gate changes. Offline225 valid convolution parity is separate from streaming initializer and phase controls.'),indent=2)+'\n')
def run():
 import subprocess
 subprocess.run(['git','diff','--exit-code','HEAD','--',str(P)],check=True);assert subprocess.check_output(['git','ls-files','--',str(P)],text=True).strip();lock=json.loads(P.read_text())
 for p,h in lock['files'].items():assert sha(p)==h,p
 import probe_hey_context_transfer_shapes_v1 as helper
 from hey_broad_integer_training_observer_v1 import operators,configuration
 from ai_edge_litert.interpreter import Interpreter,OpResolverType
 tf=helper.tf;parent=json.loads(Path(lock['training_plan']).read_text());exports=json.loads(Path(parent['export']).read_text())['arms'];cases=json.loads(Path(lock['training_result']).read_text())['cases'];reports=[];O.mkdir(exist_ok=False)
 for arm in parent['arms']:
  target=arm['target'];export=next(a for a in exports if a['target']==target);model=helper.create(225);model.load_weights(arm['path'])
  for l in model.layers:
   c=getattr(l,'cell',l)
   if isinstance(c,tf.keras.layers.BatchNormalization):l.trainable=c.trainable=False
  before=[v.numpy().copy() for v in model.variables];cfg=configuration(export);f=tf.function(operators(model,cfg,tf));i=Interpreter(model_path=export['path'],experimental_op_resolver_type=OpResolverType.BUILTIN_REF,num_threads=1);i.allocate_tensors();d={v['index']:v for v in i.get_tensor_details()};ops=[o for o in export['ops'] if o['op'] in ['CONV_2D','DEPTHWISE_CONV_2D']]
  for case in [c for c in cases if c['target']==target]:
   with np.load(case['arrays_path']) as a:x=a['x'];oldlogit=a['logits'];oldp=a['probability']
   logit,prob,observed=f(tf.constant(x));assert np.array_equal(logit.numpy(),oldlogit) and np.array_equal(prob.numpy(),oldp)
   # Authenticate every actual encoder/head quantized constant against shared H5.
   for op in [o for o in export['ops'] if o['op'] in ['CONV_2D','DEPTHWISE_CONV_2D','FULLY_CONNECTED']]:
    for ordinal,n in enumerate(op['inputs'][1:]):
     v=observed[str(n)].numpy();q=d[n]['quantization_parameters'];scale=q['scales'].astype(np.float64)
     if ordinal==0:
      if op['op']=='DEPTHWISE_CONV_2D':v=np.transpose(v,(3,0,1,2));scale=scale.reshape(1,1,1,-1)
      elif op['op']=='CONV_2D':v=np.transpose(v,(3,0,1,2));scale=scale.reshape(-1,1,1,1)
      else:v=v.T;scale=scale.reshape(-1,1)
     assert np.array_equal(np.rint(v.astype(np.float64)/scale),i.get_tensor(n)),(target,n)
   ins,iz=d[0]['quantization'];codes=np.clip(np.rint(x/np.float32(ins)+np.float32(iz)),-128,127).astype(np.int64);arrays={};layers=[]
   for j,op in enumerate(ops):
    ix,wi,bi=op['inputs'];oy=op['outputs'][0];sc,z=d[ix]['quantization'];os,oz=d[oy]['quantization'];w=i.get_tensor(wi).astype(np.int64);bias=i.get_tensor(bi).astype(np.int64);k=w.shape[1];stride=3 if j==0 else 1
    window=np.lib.stride_tricks.sliding_window_view(codes-z,k,axis=1)[:,::stride,:,:].transpose(0,1,3,2)
    if op['op']=='DEPTHWISE_CONV_2D':acc=np.sum(window*w[0,:,0,:][None,None,:,:],axis=2)+bias
    else:acc=np.einsum('btkc,okc->bto',window,w[:,:,0,:])+bias
    assert acc.min()>=-(1<<31) and acc.max()<(1<<31)
    m=sc*d[wi]['quantization_parameters']['scales'].astype(np.float64)/os
    v=np.stack([fixed(acc[:,:,n],factor,False) for n,factor in enumerate(m)],axis=-1)+oz;lo=-128 if op['op']=='DEPTHWISE_CONV_2D' else max(-128,oz);codes=np.clip(v,lo,127)
    graph=np.rint(observed[str(oy)].numpy().reshape(codes.shape).astype(np.float64)/os+oz).astype(np.int64);e=np.abs(graph-codes);layers.append(dict(index=oy,points=int(e.size),unequal=int(np.sum(e!=0)),maximum=int(e.max())));arrays[f'graph_{oy}']=graph.astype(np.int8);arrays[f'reference_{oy}']=codes.astype(np.int8)
   assert all(np.array_equal(v,a.numpy()) for v,a in zip(before,model.variables,strict=True))
   p=O/f'{target}-{case["batch"]:02}.npz';np.savez_compressed(p,**arrays);report=dict(target=target,batch=case['batch'],all_encoder_codes_exact=all(l['unequal']==0 for l in layers),quantized_constants_exact=True,original_logits_probability_exact=True,variables_unchanged=True,layers=layers,arrays_path=str(p),arrays_sha256=sha(p));reports.append(report);print(json.dumps(report),flush=True)
 with (R/'4090-hey-broad-integer-training-parity-result-v1.json').open('x') as out:out.write(json.dumps(dict(plan_sha256=sha(P),cases=reports,limits=lock['policy']),indent=2)+'\n')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['plan','run']);a=p.parse_args();plan() if a.action=='plan' else run()
