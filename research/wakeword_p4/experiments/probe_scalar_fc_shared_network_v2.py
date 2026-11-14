"""Once-only shared-variable wiring/gradient diagnostic, no optimizer."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
R=Path('research/wakeword_p4/results');P=R/'4090-scalar-fc-shared-network-plan-v2.json';OUT=Path('.local/wakeword/scalar-fc-shared-network-v2')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def plan():
 names=['4090-hey-broad-consistency-fixed-grid-compile-result-v1.json','4090-hey-broad-consistency-fixed-grid-compile-plan-v1.json','4090-hey-broad-integer-encoder-training-result-v1.json','4090-hey-broad-scalar-fc-saved-result-v1.json','4090-hey-broad-consistency-formal-execution-v1.json']
 files={str(R/n):sha(R/n) for n in names}
 here=Path(__file__).parent
 for n in ['probe_scalar_fc_shared_network_v2.py','hey_broad_scalar_fc_training_observer_v1.py','scalar_fc_training_diagnostic_v1.md','hey_broad_scalar_fc_ste_v1.py','hey_broad_integer_training_observer_v1.py','hey_broad_integer_encoder_training_v1.py','hey_broad_gpu_training_factory_v1.py','hey_broad_fixedpoint_ste_v2.py','hey_broad_fake_quant_primitives_v1.py','probe_hey_context_transfer_shapes_v1.py','probe_hey_broad_integer_requant_v2.py']:files[str(here/n)]=sha(here/n)
 for c in json.loads((R/names[2]).read_text())['cases']:assert sha(c['arrays_path'])==c['arrays_sha256'];files[c['arrays_path']]=c['arrays_sha256']
 for a in json.loads((R/names[0]).read_text())['arms']:
  # H5 path belongs to the original complete formal receipt.
  formal=json.loads((R/'4090-hey-broad-consistency-formal-result-v1.json').read_text());c=next(c for c in formal['cases'] if (c['target'],c['condition'])==(a['target'],a['condition']));h=c['final_repeat']['weights_path'];assert sha(h)==a['final_h5_sha256'];files[h]=sha(h);files[a['path']]=sha(a['path'])
 files[str(R/'4090-hey-broad-consistency-formal-result-v1.json')]=sha(R/'4090-hey-broad-consistency-formal-result-v1.json')
 versions=json.loads((R/names[-1]).read_text())['versions']
 with P.open('x') as f:json.dump(dict(files=files,versions=versions,policy='All4 final models x original2 batches/target x twice repeat; no optimizer/freshdata/model export. Preserve original raw/probability; actual shared exactFC accumulator/code vs independent integer matmul, fixed sum exactFC gradients saved before assertions. Finite/nonNone/repeat/variables unchanged. Stop on mismatch, no retry.'),f,indent=2);f.write('\n')
def run():
 from run_hey_broad_trained_quantization_v1 import published
 published(P);lock=json.loads(P.read_text())
 for p,h in lock['files'].items():assert sha(p)==h,p
 import importlib.metadata
 for n,v in lock['versions'].items():assert importlib.metadata.version(n)==v
 import hey_broad_gpu_training_factory_v1 as helper
 from hey_broad_scalar_fc_training_observer_v1 import operators,configuration
 from hey_broad_integer_training_observer_v1 import operators as original
 from ai_edge_litert.interpreter import Interpreter,OpResolverType
 tf=helper.tf;OUT.mkdir(exist_ok=False);reports=[]
 arms=json.loads((R/'4090-hey-broad-consistency-fixed-grid-compile-result-v1.json').read_text())['arms'];templates=json.loads((R/'4090-hey-broad-consistency-fixed-grid-compile-plan-v1.json').read_text())['arms'];formal=json.loads((R/'4090-hey-broad-consistency-formal-result-v1.json').read_text())['cases'];batches=json.loads((R/'4090-hey-broad-integer-encoder-training-result-v1.json').read_text())['cases']
 for arm in arms:
  matching=next(c for c in formal if (c['target'],c['condition'])==(arm['target'],arm['condition']));model=helper.create(225);model.load_weights(matching['final_repeat']['weights_path'])
  for layer in model.layers:
   cell=getattr(layer,'cell',layer)
   if isinstance(cell,tf.keras.layers.BatchNormalization):layer.trainable=cell.trainable=False
  before=[v.numpy().copy() for v in model.variables];template=next(a for a in templates if (a['target'],a['condition'])==(arm['target'],arm['condition']));metadata=dict(arm,ops=template['template']['ops']);cfg=configuration(metadata);new=tf.function(operators(model,cfg,tf));old=tf.function(original(model,cfg,tf))
  i=Interpreter(model_path=arm['path'],experimental_op_resolver_type=OpResolverType.BUILTIN_REF,num_threads=1);i.allocate_tensors();details={d['index']:d for d in i.get_tensor_details()};fc=cfg['operators'][-1];wi=fc['weight']['index'];bi=fc['bias']['index'];weights=i.get_tensor(wi).astype(np.int64);bias=i.get_tensor(bi).astype(np.int64);ins,iz=details[64]['quantization'];os,oz=details[65]['quantization'];assert np.all(details[wi]['quantization_parameters']['zero_points']==0)
  for batch in (b for b in batches if b['target']==arm['target']):
   with np.load(batch['arrays_path'],allow_pickle=False) as a:x=a['x'].copy()
   oldz,oldp,_=old(tf.constant(x));runs=[];arrays={'x':x,'original_logits':oldz.numpy(),'original_probability':oldp.numpy()}
   for repeat in range(2):
    with tf.GradientTape() as tape:
     z,p,observed=new(tf.constant(x));loss=tf.reduce_sum(observed['scalar_fc_exact'])
    grads=tape.gradient(loss,model.trainable_variables);assert all(g is not None for g in grads)
    values={'logits':z.numpy(),'probability':p.numpy(),**{k:v.numpy() for k,v in observed.items()}}
    values.update({f'gradient_{n}':g.numpy() for n,g in enumerate(grads)});runs.append(values);arrays.update({f'r{repeat}_{k}':v for k,v in values.items()})
   embedding=runs[0]['embedding_dequantized'];codes=np.rint(embedding.astype(np.float64)/ins).astype(np.int64);acc=codes@weights.T+bias;real=acc.astype(np.float64)*(ins*float(details[wi]['quantization_parameters']['scales'][0])/os);expected=np.clip(np.copysign(np.floor(np.abs(real)+.5),real)+oz,max(-128,oz),min(127,int(np.rint(6/os))+oz)).astype(np.int64);arrays['independent_accumulator']=acc;arrays['independent_fc_codes']=expected
   dest=OUT/f'{arm["target"]}-{arm["condition"]}-batch{batch["batch"]}.npz';np.savez_compressed(dest,**arrays)
   for op in (o for o in metadata['ops'] if o['op'] in ('CONV_2D','DEPTHWISE_CONV_2D','FULLY_CONNECTED')):
    for ordinal,n in enumerate(op['inputs'][1:]):
     v=runs[0][str(n)];scales=details[n]['quantization_parameters']['scales'].astype(np.float64)
     if ordinal==0:
      if op['op']=='DEPTHWISE_CONV_2D':v=np.transpose(v,(3,0,1,2));scales=scales.reshape(1,1,1,-1)
      elif op['op']=='CONV_2D':v=np.transpose(v,(3,0,1,2));scales=scales.reshape(-1,1,1,1)
      else:v=v.T;scales=scales.reshape(-1,1)
     assert np.array_equal(np.rint(v.astype(np.float64)/scales),i.get_tensor(n)),(arm['target'],n)
   assert np.array_equal(oldz.numpy(),runs[0]['logits']) and np.array_equal(oldp.numpy(),runs[0]['probability'])
   assert all(np.array_equal(runs[0][k],runs[1][k]) and np.isfinite(runs[0][k]).all() for k in runs[0]);assert np.array_equal(acc,runs[0]['scalar_fc_accumulator']);assert np.array_equal(expected,np.rint(runs[0]['scalar_fc_exact']/os+oz).astype(np.int64))
   assert all(np.array_equal(a,v.numpy()) for a,v in zip(before,model.variables,strict=True))
   reports.append(dict(target=arm['target'],condition=arm['condition'],batch=batch['batch'],arrays_path=str(dest),arrays_sha256=sha(dest),all_passed=True,gradient_l2=[float(np.linalg.norm(runs[0][f'gradient_{n}'].astype(float))) for n in range(len(grads))]));print(arm['target'],arm['condition'],batch['batch'],'passed',flush=True)
 assert len(reports)==8
 with (R/'4090-scalar-fc-shared-network-result-v2.json').open('x') as f:json.dump(dict(plan_sha256=sha(P),cases=reports,all_passed=True,optimizer_updates=0,limits=lock['policy']),f,indent=2);f.write('\n')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['plan','run']);a=p.parse_args();plan() if a.action=='plan' else run()
