"""Optimizer-zero nominal bounded-C component-gradient preflight."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
R=Path('research/wakeword_p4/results');P=R/'4090-bounded-consistency-gradients-plan-v1.json';OUT=Path('.local/wakeword/bounded-consistency-gradients-v1')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def plan():
 parent=R/'4090-hey-broad-consistency-gradient-diagnostic-plan-v1.json';base=json.loads(parent.read_text());files=dict(base['files']);result=R/'4090-hey-broad-consistency-gradient-diagnostic-result-v1.json';files[str(result)]=sha(result);files[str(parent)]=sha(parent)
 for p in (Path(__file__).resolve().relative_to(Path.cwd()),*[Path(__file__).with_name(n) for n in ('hey_broad_bounded_consistency_loss_v1.py','hey_broad_scalar_fc_training_observer_v1.py','hey_broad_scalar_fc_ste_v1.py','hey_broad_bounded_consistency_preflight_v1.md','hey_broad_consistency_gradient_merge_v1.py')]):files[str(p)]=sha(p)
 for target in ('hey_willow','hey_nyx'):
  p=Path('.local/wakeword/hey-broad-consistency-gradient-diagnostic-v1')/target/'initial.weights.h5';files[str(p)]=sha(p)
 rows=json.loads(result.read_text())['cases'];assert len(rows)==8
 for c in rows:assert sha(c['arrays_path'])==c['arrays_sha256'];files[c['arrays_path']]=c['arrays_sha256']
 with P.open('x') as f:json.dump(dict(files=files,parent=str(parent),original_result=str(result),policy='Newpolicy constant_folding=False and arithmetic_optimization=False before allgraphs. Separate S/C backwards and canonical merge0/1, nominalquantbounded=clip((float32(exactFC)-3)*(float32(8/3)),-8,8). Same-side saturation/zero gradients retained. Original8 actualsaved x/y/w and actualsavedinitialH5, New graph policy; twice evaluate raw/bounded logits and S/C component gradients, no optimizer/newdata or finalmodel selection. Save logits/repeats/modelunchanged before descriptive decomposition. Same-side nominal saturation means both>=8 or both<=-8; opposed saturation separate, all other records retained. Nominal clipped sigmoid is not actual full quantized head. Actual historical logits never saved; this is new observation on authenticated originalinputs/state, not historical byte replay.'),f,indent=2);f.write('\n')
def run():
 from run_hey_broad_trained_quantization_v1 import published
 published(P);lock=json.loads(P.read_text())
 for p,h in lock['files'].items():assert sha(p)==h,p
 import hey_broad_gpu_training_factory_v1 as helper
 from hey_broad_folded_operators_v1 import operators as floating
 from hey_broad_scalar_fc_training_observer_v1 import operators as integer,configuration
 from hey_broad_bounded_consistency_loss_v1 import components
 from hey_broad_consistency_gradient_merge_v1 import merge
 tf=helper.tf;tf.config.optimizer.set_experimental_options({'constant_folding':False,'arithmetic_optimization':False});parent=json.loads(Path(lock['parent']).read_text());exports=json.loads(Path(parent['export']).read_text())['arms'];rows=json.loads(Path(lock['original_result']).read_text())['cases'];OUT.mkdir(exist_ok=False);reports=[]
 for target in ('hey_willow','hey_nyx'):
  model=helper.create(225);model.load_weights(Path('.local/wakeword/hey-broad-consistency-gradient-diagnostic-v1')/target/'initial.weights.h5')
  for layer in model.layers:
   cell=getattr(layer,'cell',layer)
   if isinstance(cell,tf.keras.layers.BatchNormalization):layer.trainable=cell.trainable=False
  before=[v.numpy().copy() for v in model.variables];ff=floating(model,tf);qf=integer(model,configuration(next(a for a in exports if a['target']==target)),tf)
  @tf.function
  def evaluate(x,y,w):
   with tf.GradientTape(persistent=True) as tape:
    zf=ff(x);zq,prob,observed=qf(x);bf=tf.clip_by_value(zf,-8.,8.);bq=tf.clip_by_value((tf.cast(observed['scalar_fc_exact'],tf.float32)-3)*tf.constant(8/3,tf.float32),-8.,8.);loss,s,c=components(zf,zq,bf,bq,y,w,1,tf)
   sg=tape.gradient(s,model.trainable_variables);cg=tape.gradient(c,model.trainable_variables)
   mg0=merge(sg,cg,0,tf);mg1=merge(sg,cg,1,tf)
   return dict(float_logits=zf,quant_logits=zq,float_bounded=bf,quant_bounded=bq,losses=tf.stack([s,c,loss]),**{f'{name}_{n}':v for name,group in [('supervised',sg),('consistency',cg),('merged0',mg0),('merged1',mg1)] for n,v in enumerate(group)},**{'observed_'+k:v for k,v in observed.items()})
  for row in (c for c in rows if c['target']==target):
   with np.load(row['arrays_path'],allow_pickle=False) as a:x=a['x'].copy();y=a['y'].copy();w=a['w'].copy();old=a['losses'].copy()
   runs=[{k:v.numpy() for k,v in evaluate(tf.constant(x),tf.constant(y),tf.constant(w)).items()} for _ in range(2)];dest=OUT/f'{target}-batch{row["batch"]}.npz';np.savez_compressed(dest,x=x,y=y,w=w,original_saved_losses=old,**runs[0],**{'repeat_'+k:v for k,v in runs[1].items()})
   assert all(np.array_equal(runs[0][k],runs[1][k]) and np.isfinite(runs[0][k]).all() for k in runs[0]);assert all(np.array_equal(a,v.numpy()) for a,v in zip(before,model.variables,strict=True));reports.append(dict(target=target,batch=row['batch'],arrays_path=str(dest),arrays_sha256=sha(dest),repeat_exact=True,model_unchanged=True,supervised_gradient_l2=float(np.sqrt(sum(np.sum(runs[0][f'supervised_{n}'].astype(float)**2) for n in range(15)))),consistency_gradient_l2=float(np.sqrt(sum(np.sum(runs[0][f'consistency_{n}'].astype(float)**2) for n in range(15)))),supervised_loss=float(runs[0]['losses'][0]),consistency_loss=float(runs[0]['losses'][1])));print(target,row['batch'],'saved',flush=True)
 with (R/'4090-bounded-consistency-gradients-result-v1.json').open('x') as f:json.dump(dict(plan_sha256=sha(P),cases=reports,optimizer_updates=0,limits=lock['policy']),f,indent=2);f.write('\n')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['plan','run']);a=p.parse_args();plan() if a.action=='plan' else run()
