"""Prospectively locked original-step complete temporal TF integration probe."""
import hashlib,json,subprocess,sys,traceback
from pathlib import Path
import numpy as np
R=Path('research/wakeword_p4/results');P=R/'4090-learned-grid-nonuniform-stream-execution-v1.json';OUT=Path('.local/wakeword/learned-grid-nonuniform-stream-v1')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def lock():
 parent=R/'4090-learned-grid-encoder-parity-execution-v1.json';base=json.loads(parent.read_text());files=dict(base['files'])
 for f in [parent,Path(__file__),Path(__file__).with_name('learned_grid_stream_encoder_v2.py'),Path(__file__).with_name('learned_grid_requant_v1.py'),R/'4090-learned-grid-encoder-parity-result-v1.json']:files[str(f)]=sha(f)
 extra=[Path(__file__).with_name('compile_learned_grid_nonuniform_v1.py'),R/'4090-learned-activation-grid-constraints-v1.json']
 for f in extra:files[str(f)]=sha(f)
 prior=json.loads((R/'4090-learned-grid-encoder-parity-result-v1.json').read_text())
 for c in prior['cases']:
  if c['factor']==1:files[c['arrays_path']]=sha(c['arrays_path'])
 P.write_text(json.dumps(dict(ratios=np.log([.73,1.31,.89,1.17,.67,1.41,.83,1.23,.91]).tolist(),files=files,cases=[c for c in prior['cases'] if c['factor']==1],sources=base['sources'],models=[c for c in base['compiled'] if c['factor']==1],policy='Nonuniform nine ratios fixed prospectively, no selection. Direct compiled INT8 actual vs TF all9 layers, allstartup/tworepeats. TF deterministic operations enabled. Four originalrequest0 complete inputs/allstartup, original quantized kernels/realbias, nine zero-initialized logsteps; no optimizer. Offline differentiable temporal encoder only, twice repeat, all9forward codes exact vs saved originalactual. Finalembedding deterministic linear objective checks finite connected nine step gradients, not recognition/loss efficacy. No FC reference, trainedweights or candidate qualification; original quantized realbias only.'),indent=2)+'\n')
def run():
 assert subprocess.check_output(['git','show','origin/research-hye-nyx-voice-gpu:'+str(P)])==P.read_bytes();p=json.loads(P.read_text())
 for f,h in p['files'].items():assert sha(f)==h
 import tensorflow as tf
 tf.config.experimental.enable_op_determinism()
 from ai_edge_litert.interpreter import Interpreter,OpResolverType
 from learned_grid_stream_encoder_v2 import forward
 from compile_learned_grid_nonuniform_v1 import compile_bytes
 mapping=json.loads((R/'4090-learned-activation-grid-constraints-v1.json').read_text())
 OUT.mkdir(exist_ok=False);rows=[]
 try:
  for c in p['cases']:
   model=next(x for x in p['models'] if (x['target'],x['condition'])==(c['target'],c['condition']));source=next(x for x in p['sources'] if (x['target'],x['condition'])==(c['target'],c['condition']))
   it=Interpreter(model_path=model['path'],experimental_op_resolver_type=OpResolverType.BUILTIN_REF,num_threads=1);it.allocate_tensors();ds={x['index']:x for x in it.get_tensor_details()};ops=[x for x in it._get_ops_details() if x['op_name'] in ('CONV_2D','DEPTHWISE_CONV_2D')];configs=[]
   for op in ops:
    ix,wi,bi=map(int,op['inputs']);oy=int(op['outputs'][0]);ins,iz=ds[ix]['quantization'];outs,oz=ds[oy]['quantization'];b=it.get_tensor(bi).astype(np.float64)*ds[bi]['quantization_parameters']['scales'].astype(np.float64)
    configs.append(dict(kind=op['op_name'],index=oy,input_step=ins,input_zero=iz,output_step=outs,output_zero=oz,kernel=it.get_tensor(wi).astype(np.float64),weight_steps=ds[wi]['quantization_parameters']['scales'].astype(np.float64),bias_real=b))
   with np.load(source['arrays_path'],allow_pickle=False) as a:q=a['quantized_input'].copy()
   maprow=next(x for x in mapping['models'] if (x['target'],x['condition'])==(c['target'],c['condition']))
   selected=[]
   for config in configs:
    selected.append(next(j for j,g in enumerate(maprow['groups']) if any(m['graph']==0 and m['tensor']==config['index'] for m in g['members'])))
   assert len(set(selected))==9
   factors={group:float(np.exp(p['ratios'][j])) for j,group in enumerate(selected)}
   compiled,biaschanges=compile_bytes(Path(model['path']).read_bytes(),maprow['groups'],selected,factors)
   compiled_path=OUT/(c['target']+'-'+c['condition']+'.tflite');compiled_path.write_bytes(compiled)
   actualruns=[]
   for repeat in range(2):
    actual=Interpreter(model_path=str(compiled_path),experimental_op_resolver_type=OpResolverType.BUILTIN_REF,num_threads=1,experimental_preserve_all_tensors=True);actual.allocate_tensors();out=actual.get_output_details()[0]['index'];traces={str(config['index']):[] for config in configs};traces['uint8']=[]
    for at in range(0,len(q),3):
     actual.set_tensor(0,q[None,at:at+3]);actual.invoke()
     for config in configs:traces[str(config['index'])].append(actual.get_tensor(config['index']).copy())
     traces['uint8'].append(actual.get_tensor(out).copy())
    actualruns.append({k:np.stack(v) for k,v in traces.items()})
   assert all(actualruns[0][k].tobytes()==actualruns[1][k].tobytes() for k in actualruns[0])
   runs=[];gradients=[];mismatch=[]
   for repeat in range(2):
    ratios=tf.Variable(p['ratios'],dtype=tf.float64)
    with tf.GradientTape() as tape:
     values=forward(q,configs,ratios,tf);last=values[-1];objective=tf.reduce_sum(last*tf.reshape(tf.linspace(tf.constant(-1.,tf.float64),tf.constant(1.,tf.float64),tf.size(last)),tf.shape(last)))
    grad=tape.gradient(objective,ratios);assert grad is not None and np.isfinite(grad.numpy()).all();assert np.array_equal(ratios.numpy(),np.array(p['ratios']))
    gradients.append(grad.numpy());runs.append([v.numpy() for v in values])
   for j,config in enumerate(configs):
    expected=actualruns[0][str(config['index'])].reshape(runs[0][j].shape)
    mismatch.append(int(np.count_nonzero(np.rint(runs[0][j])!=expected)));assert np.max(abs(runs[0][j]-np.rint(runs[0][j])))<1e-10
   repeat_exact=all(a.tobytes()==b.tobytes() for a,b in zip(runs[0],runs[1],strict=True)) and gradients[0].tobytes()==gradients[1].tobytes()
   path=OUT/(c['target']+'-'+c['condition']+'.npz');np.savez_compressed(path,**{f'layer_{j}':np.stack([r[j] for r in runs]) for j in range(9)},step_gradients=np.stack(gradients),**{'actual_'+k:np.stack([a[k] for a in actualruns]) for k in actualruns[0]})
   row=dict(compiled_path=str(compiled_path),compiled_sha256=sha(compiled_path),bias_changes=biaschanges,target=c['target'],condition=c['condition'],mismatch=mismatch,repeat_exact=repeat_exact,step_gradients=gradients[0].tolist(),arrays_path=str(path),arrays_sha256=sha(path),passed=not any(mismatch) and repeat_exact);rows.append(row);print(json.dumps(row),flush=True)
  result=dict(execution_sha256=sha(P),cases=rows,all_passed=all(c['passed'] for c in rows),optimizer_updates=0,limits=p['policy']);(R/'4090-learned-grid-nonuniform-stream-result-v1.json').write_text(json.dumps(result,indent=2)+'\n');assert result['all_passed']
 except Exception:
  (OUT/'failure.log').write_text(traceback.format_exc());raise
if __name__=='__main__':{'lock':lock,'run':run}[sys.argv[1]]()
