"""Prospectively locked original-step complete temporal TF integration probe."""
import hashlib,json,subprocess,sys,traceback
from pathlib import Path
import numpy as np
R=Path('research/wakeword_p4/results');P=R/'4090-learned-grid-stream-integration-execution-v1.json';OUT=Path('.local/wakeword/learned-grid-stream-integration-v1')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def lock():
 parent=R/'4090-learned-grid-encoder-parity-execution-v1.json';base=json.loads(parent.read_text());files=dict(base['files'])
 for f in [parent,Path(__file__),Path(__file__).with_name('learned_grid_stream_encoder_v1.py'),Path(__file__).with_name('learned_grid_requant_v1.py'),R/'4090-learned-grid-encoder-parity-result-v1.json']:files[str(f)]=sha(f)
 prior=json.loads((R/'4090-learned-grid-encoder-parity-result-v1.json').read_text())
 for c in prior['cases']:
  if c['factor']==1:files[c['arrays_path']]=sha(c['arrays_path'])
 P.write_text(json.dumps(dict(files=files,cases=[c for c in prior['cases'] if c['factor']==1],sources=base['sources'],models=[c for c in base['compiled'] if c['factor']==1],policy='Four originalrequest0 complete inputs/allstartup, original quantized kernels/realbias, nine zero-initialized logsteps; no optimizer. Offline differentiable temporal encoder only, twice repeat, all9forward codes exact vs saved originalactual. Finalembedding deterministic linear objective checks finite connected nine step gradients, not recognition/loss efficacy. No FC, trainedweights, changedgrid export or candidate qualification.'),indent=2)+'\n')
def run():
 assert subprocess.check_output(['git','show','origin/research-hye-nyx-voice-gpu:'+str(P)])==P.read_bytes();p=json.loads(P.read_text())
 for f,h in p['files'].items():assert sha(f)==h
 import tensorflow as tf
 from ai_edge_litert.interpreter import Interpreter,OpResolverType
 from learned_grid_stream_encoder_v1 import forward
 OUT.mkdir(exist_ok=False);rows=[]
 try:
  for c in p['cases']:
   model=next(x for x in p['models'] if (x['target'],x['condition'])==(c['target'],c['condition']));source=next(x for x in p['sources'] if (x['target'],x['condition'])==(c['target'],c['condition']))
   it=Interpreter(model_path=model['path'],experimental_op_resolver_type=OpResolverType.BUILTIN_REF,num_threads=1);it.allocate_tensors();ds={x['index']:x for x in it.get_tensor_details()};ops=[x for x in it._get_ops_details() if x['op_name'] in ('CONV_2D','DEPTHWISE_CONV_2D')];configs=[]
   for op in ops:
    ix,wi,bi=map(int,op['inputs']);oy=int(op['outputs'][0]);ins,iz=ds[ix]['quantization'];outs,oz=ds[oy]['quantization'];b=it.get_tensor(bi).astype(np.float64)*ds[bi]['quantization_parameters']['scales'].astype(np.float64)
    configs.append(dict(kind=op['op_name'],index=oy,input_step=ins,input_zero=iz,output_step=outs,output_zero=oz,kernel=it.get_tensor(wi).astype(np.float64),weight_steps=ds[wi]['quantization_parameters']['scales'].astype(np.float64),bias_real=b))
   with np.load(source['arrays_path'],allow_pickle=False) as a:q=a['quantized_input'].copy()
   runs=[];gradients=[];mismatch=[]
   for repeat in range(2):
    ratios=tf.Variable(np.zeros(9),dtype=tf.float64)
    with tf.GradientTape() as tape:
     values=forward(q,configs,ratios,tf);last=values[-1];objective=tf.reduce_sum(last*tf.reshape(tf.linspace(tf.constant(-1.,tf.float64),tf.constant(1.,tf.float64),tf.size(last)),tf.shape(last)))
    grad=tape.gradient(objective,ratios);assert grad is not None and np.isfinite(grad.numpy()).all();assert np.array_equal(ratios.numpy(),np.zeros(9))
    gradients.append(grad.numpy());runs.append([v.numpy() for v in values])
   with np.load(c['arrays_path'],allow_pickle=False) as old:
    for j,config in enumerate(configs):
     expected=old[str(config['index'])][0].reshape(runs[0][j].shape);mismatch.append(int(np.count_nonzero(np.rint(runs[0][j])!=expected)))
   repeat_exact=all(a.tobytes()==b.tobytes() for a,b in zip(runs[0],runs[1],strict=True)) and gradients[0].tobytes()==gradients[1].tobytes()
   path=OUT/(c['target']+'-'+c['condition']+'.npz');np.savez_compressed(path,**{f'layer_{j}':np.stack([r[j] for r in runs]) for j in range(9)},step_gradients=np.stack(gradients))
   row=dict(target=c['target'],condition=c['condition'],mismatch=mismatch,repeat_exact=repeat_exact,step_gradients=gradients[0].tolist(),arrays_path=str(path),arrays_sha256=sha(path),passed=not any(mismatch) and repeat_exact);rows.append(row);print(json.dumps(row),flush=True)
  result=dict(execution_sha256=sha(P),cases=rows,all_passed=all(c['passed'] for c in rows),optimizer_updates=0,limits=p['policy']);(R/'4090-learned-grid-stream-integration-result-v1.json').write_text(json.dumps(result,indent=2)+'\n');assert result['all_passed']
 except Exception:
  (OUT/'failure.log').write_text(traceback.format_exc());raise
if __name__=='__main__':{'lock':lock,'run':run}[sys.argv[1]]()
