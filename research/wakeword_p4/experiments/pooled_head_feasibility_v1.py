"""Original encoder transfer to existing average-pool option; no forward/export."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
R=Path('research/wakeword_p4/results')
PLAN=R/'4090-pooled-head-feasibility-plan-v1.json'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 import probe_hey_context_transfer_shapes_v1 as helper
 from microwakeword import mixednet
 p=json.loads(PLAN.read_text())
 for f,h in p['files'].items():assert sha(f)==h,f
 tf=helper.tf;tf.keras.utils.set_random_seed(p['seed']);rows=[]
 for m in p['models']:
  original=helper.create(225);original.load_weights(m['path'])
  parser=argparse.ArgumentParser();mixednet.model_parameters(parser)
  pooled=mixednet.model(parser.parse_args(helper.ARCH+['--pooled','1']),(225,40),None)
  cell=lambda l:getattr(l,'cell',l)
  source=[l for l in original.layers if l.weights and not isinstance(cell(l),tf.keras.layers.Dense)]
  dest=[l for l in pooled.layers if l.weights and not isinstance(cell(l),tf.keras.layers.Dense)]
  copied=[]
  assert len(source)==len(dest)
  for a,b in zip(source,dest,strict=True):
   assert type(cell(a))==type(cell(b))
   values=a.get_weights();assert [x.shape for x in values]==[x.shape for x in b.get_weights()]
   b.set_weights(values)
   if isinstance(cell(b),tf.keras.layers.BatchNormalization):b.trainable=cell(b).trainable=False
   assert all(x.dtype==y.dtype and x.tobytes()==y.tobytes() for x,y in zip(values,b.get_weights(),strict=True))
   copied.append(dict(kind=type(cell(a)).__name__,arrays=[dict(shape=list(x.shape),dtype=str(x.dtype),sha256=hashlib.sha256(x.tobytes()).hexdigest()) for x in values]))
  head,=[l for l in pooled.layers if isinstance(l,tf.keras.layers.Dense)]
  oldhead,=[l for l in original.layers if isinstance(l,tf.keras.layers.Dense)]
  assert tuple(head.kernel.shape)==(60,1) and tuple(oldhead.kernel.shape)==(1800,1)
  pools=[l for l in pooled.layers if isinstance(l,tf.keras.layers.AveragePooling2D)]
  assert len(pools)==1 and tuple(pools[0].pool_size)==(30,1)
  rows.append(dict(target=m['target'],condition=m['condition'],original_h5_sha256=m['sha256'],original_parameters=original.count_params(),pooled_parameters=pooled.count_params(),pool_size=list(pools[0].pool_size),new_head_shapes=[list(x.shape) for x in head.get_weights()],new_head_value_sha256=hashlib.sha256(b''.join(x.tobytes() for x in head.get_weights())).hexdigest(),copied_encoder_and_bn=copied))
 report=dict(plan_sha256=sha(PLAN),models=rows,model_forward_calls=0,optimizer_updates=0,exports=0,limits='Existing MixedNet average-pool architecture feasibility only. Original encoder/BN exact transfer; new random head is not trained or a preserved old classifier. No probability, quantization, streaming, quality, recognition, FAR or P4 claim. Pooling still retains the original temporal history and does not establish smaller arena usage.')
 with (R/'4090-pooled-head-feasibility-result-v1.json').open('x') as f:f.write(json.dumps(report,indent=2)+'\n')
 print(json.dumps([dict(target=x['target'],condition=x['condition'],parameters=x['pooled_parameters']) for x in rows]))
if __name__=='__main__':main()
