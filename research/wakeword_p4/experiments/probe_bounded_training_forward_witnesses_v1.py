"""Compare original QAT static contexts against existing integer wholewave traces."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
R=Path('research/wakeword_p4/results');E=Path('research/wakeword_p4/experiments');P=R/'4090-bounded-training-forward-witness-plan-v1.json';OUT=Path('.local/wakeword/bounded-training-forward-witness-v1')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def plan():
 names=['formal-execution','formal-result','formal-independent-audit','fixed-grid-compile-plan','fixed-grid-compile-result','fixed-grid-constant-audit','fixed-grid-screen-result','fixed-grid-screen-independent-audit','fixed-grid-recursive-result','fixed-grid-recursive-independent-audit'];paths={n:R/('4090-hey-broad-bounded-consistency-'+n+'-v1.json') for n in names}
 for p in paths.values():assert p.exists(),str(p)
 files={str(p):sha(p) for p in paths.values()};formal=json.loads(paths['formal-result'].read_text());compile=json.loads(paths['fixed-grid-compile-result'].read_text());screen=json.loads(paths['fixed-grid-screen-result'].read_text());recursive=json.loads(paths['fixed-grid-recursive-result'].read_text());assert len(screen['cases'])==len(recursive['cases'])==540
 assert json.loads(paths['fixed-grid-recursive-independent-audit'].read_text())['result_sha256']==sha(paths['fixed-grid-recursive-result'])
 assert json.loads(paths['fixed-grid-screen-independent-audit'].read_text())['result_sha256']==sha(paths['fixed-grid-screen-result'])
 cases=[]
 for target in ['hey_willow','hey_nyx']:
  for condition in ['raw_logit_control','bounded_logit_consistency']:
   order=14 if target=='hey_willow' else 48394;key=(target,condition,order);s,=[c for c in screen['cases'] if (c['target'],c['condition'],c['request_order'])==key];r,=[c for c in recursive['cases'] if (c['target'],c['condition'],c['request_order'])==key];a,=[a for a in compile['arms'] if (a['target'],a['condition'])==key[:2]];f,=[f['final_repeat'] for f in formal['cases'] if (f['target'],f['condition'])==key[:2]]
   for p,h in [(s['arrays_path'],s['arrays_sha256']),(r['arrays_path'],r['arrays_sha256']),(a['path'],a['sha256']),(f['weights_path'],f['weights_sha256'])]:assert sha(p)==h;files[p]=h
   cases.append(dict(target=target,condition=condition,request_order=order,screen=s,reference=r,arm=a,weights=f['weights_path']))
 for n in ['bounded_training_forward_witness_v1.md','probe_bounded_training_forward_witnesses_v1.py','hey_broad_scalar_fc_training_observer_v1.py','hey_broad_scalar_fc_ste_v1.py','hey_broad_fixedpoint_ste_v2.py','hey_broad_fake_quant_primitives_v1.py','hey_broad_gpu_training_factory_v1.py','probe_hey_context_transfer_shapes_v1.py']:files[str(E/n)]=sha(E/n)
 record=dict(files=files,versions=json.loads(paths['formal-execution'].read_text())['versions'],cases=cases,compile_plan=str(paths['fixed-grid-compile-plan']),policy='Fixed existing Willow14/Nyx48394 x raw/bounded; all fullsupport streamsteps>=74, no outcome selection. Original finalH5 and nofold/noarithmetic QAT observer; both nominal225 and original phase224+zero static contexts. Twice observe allconstants/intermediateencoder/embedding/scalarFC/rawlogits/probability, no gradient/optimizer/export or runtime model invoke. Compare saved actual wholewave recursive embedding/FC, preserve every discrepancy; comparison descriptive, not gate replacement. Require exact inputs/constants/repeats/modelunchanged, stop structural/hash/nonfinite faults no retry. No newdev/acceptance/promotion.')
 with P.open('x') as f:f.write(json.dumps(record,indent=2)+'\n')
def run():
 from run_hey_broad_trained_quantization_v1 import published
 published(P);lock=json.loads(P.read_text())
 for p,h in lock['files'].items():assert sha(p)==h,p
 import importlib.metadata
 for n,v in lock['versions'].items():assert importlib.metadata.version(n)==v,n
 import hey_broad_gpu_training_factory_v1 as helper
 from hey_broad_scalar_fc_training_observer_v1 import operators,configuration
 from ai_edge_litert.interpreter import Interpreter,OpResolverType
 tf=helper.tf;tf.config.optimizer.set_experimental_options({'constant_folding':False,'arithmetic_optimization':False});OUT.mkdir(exist_ok=False);reports=[];templates=json.loads(Path(lock['compile_plan']).read_text())['arms']
 for c in lock['cases']:
  target=c['target'];condition=c['condition'];arm=dict(c['arm']);arm['ops']=next(a['template']['ops'] for a in templates if (a['target'],a['condition'])==(target,condition));model=helper.create(225);model.load_weights(c['weights'])
  for layer in model.layers:
   cell=getattr(layer,'cell',layer)
   if isinstance(cell,tf.keras.layers.BatchNormalization):layer.trainable=cell.trainable=False
  before=[v.numpy().copy() for v in model.variables];infer=tf.function(operators(model,configuration(arm),tf));meta=Interpreter(model_path=arm['path'],experimental_op_resolver_type=OpResolverType.BUILTIN_REF,num_threads=1);meta.allocate_tensors();details={d['index']:d for d in meta.get_tensor_details()}
  with np.load(c['screen']['arrays_path']) as z:raw=z['raw'];q=z['quantized_input'];whole_float=z['float'];whole_uint8=z['uint8'];phase=z['offline_phase'];nominal=z['offline_nominal']
  values=np.clip(np.pad(raw,((0,(-len(raw))%3),(0,0))).astype(np.float32)/25.6,0,26);ins,inz=c['screen']['input_quantization'];assert np.array_equal(np.clip(np.rint(values/ins)+inz,-128,127).astype(np.int8),q);ends=list(range(225,len(values)+1,3));assert len(ends)==len(whole_float)-74
  with np.load(c['reference']['arrays_path']) as z:embedding=z['64'][74:].reshape(-1,1800);fc=z['actual_65'][74:].reshape(-1,1)
  arrays=dict(ends=np.asarray(ends),actual_embedding=embedding,actual_fc=fc,whole_float=whole_float[74:],whole_uint8=whole_uint8[74:],offline_phase=phase,offline_nominal=nominal);metrics={}
  for mode in ['nominal','phase']:
   x=np.stack([values[e-225:e] if mode=='nominal' else np.concatenate([values[e-224:e],np.zeros((1,40),np.float32)]) for e in ends]);arrays[mode+'_x']=x;runs=[]
   for repeat in range(2):
    z,p,o=infer(tf.constant(x));v=dict(rawlogit=z.numpy(),probability=p.numpy(),**{k:v.numpy() for k,v in o.items()});assert all(np.isfinite(a).all() for a in v.values());runs.append(v);arrays.update({f'{mode}_r{repeat}_{k}':a for k,a in v.items()})
   assert all(np.array_equal(runs[0][k],runs[1][k]) for k in runs[0]);v=runs[0]
   for op in [a for a in arm['ops'] if a['op'] in ['CONV_2D','DEPTHWISE_CONV_2D','FULLY_CONNECTED']]:
    for bias,index in enumerate(op['inputs'][1:]):
     a=v[str(index)];sc=details[index]['quantization_parameters']['scales'].astype(float)
     if not bias:
      if op['op']=='DEPTHWISE_CONV_2D':a=np.transpose(a,(3,0,1,2));sc=sc.reshape(1,1,1,-1)
      elif op['op']=='CONV_2D':a=np.transpose(a,(3,0,1,2));sc=sc.reshape(-1,1,1,1)
      else:a=a.T;sc=sc.reshape(-1,1)
     assert np.array_equal(np.rint(a.astype(float)/sc),meta.get_tensor(index)),(target,condition,index)
   es,ez=details[64]['quantization'];fs,fz=details[65]['quantization'];ec=np.rint(v['embedding_dequantized'].astype(float)/es+ez).astype(np.int64);fcodes=np.rint(v['scalar_fc_exact'].astype(float)/fs+fz).astype(np.int64);arrays[mode+'_embedding_codes']=ec;arrays[mode+'_fc_codes']=fcodes
   metrics[mode]=dict(contexts=len(x),embedding_unequal=int(np.sum(ec!=embedding)),embedding_maximum=int(np.max(np.abs(ec-embedding))),fc_unequal=int(np.sum(fcodes!=fc)),fc_maximum=int(np.max(np.abs(fcodes-fc))),saved_repeats_exact=True)
  assert all(np.array_equal(a,v.numpy()) for a,v in zip(before,model.variables,strict=True));dest=OUT/f'{target}-{condition}.npz';np.savez_compressed(dest,**arrays);reports.append(dict(target=target,condition=condition,request_order=c['request_order'],arrays_path=str(dest),arrays_sha256=sha(dest),model_unchanged=True,metrics=metrics));print(json.dumps(reports[-1]),flush=True)
 with (R/'4090-bounded-training-forward-witness-result-v1.json').open('x') as f:f.write(json.dumps(dict(plan_sha256=sha(P),cases=reports,optimizer_updates=0,runtime_model_invocations=0,limits=lock['policy']),indent=2)+'\n')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['plan','run']);a=p.parse_args();plan() if a.action=='plan' else run()
