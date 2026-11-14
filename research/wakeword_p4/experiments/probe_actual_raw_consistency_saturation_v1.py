"""Optimizer-zero raw-C saturation diagnostic on original saved initialization/batches."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
R=Path('research/wakeword_p4/results');P=R/'4090-actual-raw-consistency-saturation-plan-v1.json';OUT=Path('.local/wakeword/actual-raw-consistency-saturation-v1')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def plan():
 parent=R/'4090-hey-broad-consistency-gradient-diagnostic-plan-v1.json';base=json.loads(parent.read_text());files=dict(base['files']);result=R/'4090-hey-broad-consistency-gradient-diagnostic-result-v1.json';files[str(result)]=sha(result);files[str(parent)]=sha(parent)
 for p in (Path(__file__).resolve().relative_to(Path.cwd()),Path(__file__).with_name('hey_broad_consistency_loss_v1.py')):files[str(p)]=sha(p)
 for target in ('hey_willow','hey_nyx'):
  p=Path('.local/wakeword/hey-broad-consistency-gradient-diagnostic-v1')/target/'initial.weights.h5';files[str(p)]=sha(p)
 rows=json.loads(result.read_text())['cases'];assert len(rows)==8
 for c in rows:assert sha(c['arrays_path'])==c['arrays_sha256'];files[c['arrays_path']]=c['arrays_sha256']
 with P.open('x') as f:json.dump(dict(files=files,parent=str(parent),original_result=str(result),policy='Original8 actualsaved x/y/w and actualsavedinitialH5, default original graph optimizer policy. Twice evaluate raw float/quant precliplogits and S/C, no gradients/optimizer/newdata or finalmodel selection. Save logits/repeats/modelunchanged before descriptive decomposition. Same-side nominal saturation means both>=8 or both<=-8; opposed saturation separate, all other records retained. Nominal clipped sigmoid is not actual full quantized head. Actual historical logits never saved; this is new observation on authenticated originalinputs/state, not historical byte replay.'),f,indent=2);f.write('\n')
def run():
 from run_hey_broad_trained_quantization_v1 import published
 published(P);lock=json.loads(P.read_text())
 for p,h in lock['files'].items():assert sha(p)==h,p
 import hey_broad_gpu_training_factory_v1 as helper
 from hey_broad_folded_operators_v1 import operators as floating
 from hey_broad_integer_encoder_training_v1 import operators as integer,configuration
 from hey_broad_consistency_loss_v1 import components
 tf=helper.tf;parent=json.loads(Path(lock['parent']).read_text());exports=json.loads(Path(parent['export']).read_text())['arms'];rows=json.loads(Path(lock['original_result']).read_text())['cases'];OUT.mkdir(exist_ok=False);reports=[]
 for target in ('hey_willow','hey_nyx'):
  model=helper.create(225);model.load_weights(Path('.local/wakeword/hey-broad-consistency-gradient-diagnostic-v1')/target/'initial.weights.h5')
  for layer in model.layers:
   cell=getattr(layer,'cell',layer)
   if isinstance(cell,tf.keras.layers.BatchNormalization):layer.trainable=cell.trainable=False
  before=[v.numpy().copy() for v in model.variables];ff=floating(model,tf);qf=integer(model,configuration(next(a for a in exports if a['target']==target)),tf)
  @tf.function
  def evaluate(x,y,w):
   zf=ff(x);zq=qf(x)[0];loss,s,c=components(zf,zq,y,w,1,tf);return zf,zq,tf.stack([s,c,loss])
  for row in (c for c in rows if c['target']==target):
   with np.load(row['arrays_path'],allow_pickle=False) as a:x=a['x'].copy();y=a['y'].copy();w=a['w'].copy();old=a['losses'].copy()
   runs=[[v.numpy() for v in evaluate(tf.constant(x),tf.constant(y),tf.constant(w))] for _ in range(2)];dest=OUT/f'{target}-batch{row["batch"]}.npz';np.savez_compressed(dest,x=x,y=y,w=w,float_logits=runs[0][0],quant_logits=runs[0][1],losses=runs[0][2],repeat_float_logits=runs[1][0],repeat_quant_logits=runs[1][1],repeat_losses=runs[1][2],original_saved_losses=old)
   assert all(np.array_equal(a,b) and np.isfinite(a).all() for a,b in zip(*runs,strict=True));assert all(np.array_equal(a,v.numpy()) for a,v in zip(before,model.variables,strict=True));reports.append(dict(target=target,batch=row['batch'],arrays_path=str(dest),arrays_sha256=sha(dest),repeat_exact=True,model_unchanged=True,historical_losses_exact=np.array_equal(old,runs[0][2])));print(target,row['batch'],'saved',flush=True)
 with (R/'4090-actual-raw-consistency-saturation-result-v1.json').open('x') as f:json.dump(dict(plan_sha256=sha(P),cases=reports,optimizer_updates=0,limits=lock['policy']),f,indent=2);f.write('\n')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['plan','run']);a=p.parse_args();plan() if a.action=='plan' else run()
