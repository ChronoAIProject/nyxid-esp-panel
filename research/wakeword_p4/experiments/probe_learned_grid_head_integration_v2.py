"""Prospectively locked original-step complete temporal TF integration probe."""
import hashlib,json,subprocess,sys,traceback
from pathlib import Path
import numpy as np
R=Path('research/wakeword_p4/results');P=R/'4090-learned-grid-head-integration-execution-v2.json';OUT=Path('.local/wakeword/learned-grid-head-integration-v2')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def lock():
 parent=R/'4090-learned-grid-encoder-parity-execution-v1.json';base=json.loads(parent.read_text());files=dict(base['files'])
 for f in [Path(__file__).with_name('learned_grid_head_v2.py'),R/'4090-matched-internal-boundaries-original-result-v1.json',parent,Path(__file__),Path(__file__).with_name('learned_grid_stream_encoder_v3.py'),Path(__file__).with_name('learned_grid_requant_v1.py'),R/'4090-learned-grid-encoder-parity-result-v1.json']:files[str(f)]=sha(f)
 prior=json.loads((R/'4090-learned-grid-encoder-parity-result-v1.json').read_text())
 for c in prior['cases']:
  if c['factor']==1:files[c['arrays_path']]=sha(c['arrays_path'])
 captures=json.loads((R/'4090-matched-internal-boundaries-original-result-v1.json').read_text())
 for cap in captures['cases']:
  for key in ('captures_path','metadata_path'):files[cap[key]]=sha(cap[key])
 P.write_text(json.dumps(dict(files=files,cases=[c for c in prior['cases'] if c['factor']==1],sources=base['sources'],models=[c for c in base['compiled'] if c['factor']==1],policy='Corrected first temporal history=kernel-3 (2), remaining histories kernel-1; TF deterministic operations enabled. Four originalrequest0 complete inputs/allstartup, 18 trainable folded FP32 kernel/bias tensors from actual FP32 export and nine zero logsteps; fixed weight grids; no optimizer. Offline differentiable temporal encoder only, twice repeat, all9forward codes exact vs saved originalactual. Correct signed raw-code pooling half ties before zero-point centering. Complete30history and pool/flat embedding plus rawFC codes vs originalcaptures; deterministic linear rawlogit objective checks20parameters+9stepgradients, not recognition/loss efficacy. No sigmoid kernel oracle, optimizer updates, post-update export or candidate qualification. Folded parameterization differs from raw BN parameterization.'),indent=2)+'\n')
def run():
 assert subprocess.check_output(['git','show','origin/research-hye-nyx-voice-gpu:'+str(P)])==P.read_bytes();p=json.loads(P.read_text())
 for f,h in p['files'].items():assert sha(f)==h
 import tensorflow as tf
 tf.config.experimental.enable_op_determinism()
 from ai_edge_litert.interpreter import Interpreter,OpResolverType
 from learned_grid_stream_encoder_v3 import forward
 from learned_grid_head_v2 import forward as head_forward
 OUT.mkdir(exist_ok=False);rows=[]
 try:
  for c in p['cases']:
   model=next(x for x in p['models'] if (x['target'],x['condition'])==(c['target'],c['condition']));source=next(x for x in p['sources'] if (x['target'],x['condition'])==(c['target'],c['condition']))
   it=Interpreter(model_path=model['path'],experimental_op_resolver_type=OpResolverType.BUILTIN_REF,num_threads=1);it.allocate_tensors();ds={x['index']:x for x in it.get_tensor_details()};ops=[x for x in it._get_ops_details() if x['op_name'] in ('CONV_2D','DEPTHWISE_CONV_2D')];configs=[]
   for op in ops:
    ix,wi,bi=map(int,op['inputs']);oy=int(op['outputs'][0]);ins,iz=ds[ix]['quantization'];outs,oz=ds[oy]['quantization'];b=it.get_tensor(bi).astype(np.float64)*ds[bi]['quantization_parameters']['scales'].astype(np.float64)
    configs.append(dict(kind=op['op_name'],index=oy,input_step=ins,input_zero=iz,output_step=outs,output_zero=oz,kernel=it.get_tensor(wi).astype(np.float64),weight_steps=ds[wi]['quantization_parameters']['scales'].astype(np.float64),bias_real=b))
   with np.load(source['arrays_path'],allow_pickle=False) as a:q=a['quantized_input'].copy()
   original=json.loads((R/'4090-matched-internal-boundaries-execution-v1.json').read_text())
   export=next(e for e in original['exports'] if (e['target'],e['condition'])==(c['target'],c['condition']))
   fp=Interpreter(model_path=export['fp32_path'],experimental_op_resolver_type=OpResolverType.BUILTIN_REF,num_threads=1);fp.allocate_tensors()
   fops=[o for o in fp._get_ops_details() if o['op_name'] in ('CONV_2D','DEPTHWISE_CONV_2D')]
   initial=[(fp.get_tensor(int(o['inputs'][1])).astype(np.float64),fp.get_tensor(int(o['inputs'][2])).astype(np.float64)) for o in fops]
   fc,=[o for o in it._get_ops_details() if o['op_name']=='FULLY_CONNECTED'];ffc,=[o for o in fp._get_ops_details() if o['op_name']=='FULLY_CONNECTED']
   ix,wi,bi=map(int,fc['inputs']);oy=int(fc['outputs'][0]);ws=ds[wi]['quantization_parameters']['scales'];assert len(ws)==1
   hc=dict(input_zero=ds[ix]['quantization'][1],weight_step=float(ws[0]),output_step=ds[oy]['quantization'][0],output_zero=ds[oy]['quantization'][1])
   head_initial=(fp.get_tensor(int(ffc['inputs'][1])).astype(np.float64),fp.get_tensor(int(ffc['inputs'][2])).astype(np.float64))
   caps=json.loads((R/'4090-matched-internal-boundaries-original-result-v1.json').read_text());cap=next(x for x in caps['cases'] if (x['target'],x['condition'])==(c['target'],c['condition']))
   with np.load(cap['captures_path'],allow_pickle=False) as ca:expected_embedding=ca['int8_embedding'].copy();expected_logit=ca['int8_logit'].copy()
   runs=[];gradients=[];parameter_gradients=[];head_runs=[];mismatch=[]
   for repeat in range(2):
    parameters=[(tf.Variable(w,dtype=tf.float64),tf.Variable(b,dtype=tf.float64)) for w,b in initial];head_parameters=[tf.Variable(v,dtype=tf.float64) for v in head_initial];flat=[v for pair in parameters for v in pair]+head_parameters
    ratios=tf.Variable(np.zeros(9),dtype=tf.float64)
    with tf.GradientTape() as tape:
     values=forward(q,configs,ratios,tf,parameters);embedding,logitcodes,rawlogit,deployedlogit=head_forward(values[-1],configs[-1]['output_step'],ratios[-1],hc,head_parameters,c['condition']=='pool',tf);head_runs.append((embedding,logitcodes));last=rawlogit;objective=tf.reduce_sum(last*tf.reshape(tf.linspace(tf.constant(-1.,tf.float64),tf.constant(1.,tf.float64),tf.size(last)),tf.shape(last)))
    allgrads=tape.gradient(objective,[ratios]+flat);grad=allgrads[0];assert all(g is not None and np.isfinite(g.numpy()).all() for g in allgrads);parameter_gradients.append([g.numpy() for g in allgrads[1:]]);assert all(np.array_equal(v.numpy(),a) for v,a in zip(flat,[a for pair in initial for a in pair]+list(head_initial),strict=True));assert grad is not None and np.isfinite(grad.numpy()).all();assert np.array_equal(ratios.numpy(),np.zeros(9))
    gradients.append(grad.numpy());runs.append([v.numpy() for v in values])
   with np.load(c['arrays_path'],allow_pickle=False) as old:
    for j,config in enumerate(configs):
     expected=old[str(config['index'])][0].reshape(runs[0][j].shape);mismatch.append(int(np.count_nonzero(np.rint(runs[0][j])!=expected)));assert np.max(abs(runs[0][j]-np.rint(runs[0][j])))<1e-10
   parameter_repeat=all(a.tobytes()==b.tobytes() for a,b in zip(parameter_gradients[0],parameter_gradients[1],strict=True))
   head_mismatch=[int(np.count_nonzero(np.rint(head_runs[0][j].numpy())!=expected[0].reshape(head_runs[0][j].shape))) for j,expected in enumerate((expected_embedding,expected_logit))]
   head_repeat=all(a.numpy().tobytes()==b.numpy().tobytes() for a,b in zip(head_runs[0],head_runs[1],strict=True))
   repeat_exact=head_repeat and parameter_repeat and all(a.tobytes()==b.tobytes() for a,b in zip(runs[0],runs[1],strict=True)) and gradients[0].tobytes()==gradients[1].tobytes()
   path=OUT/(c['target']+'-'+c['condition']+'.npz');np.savez_compressed(path,**{f'layer_{j}':np.stack([r[j] for r in runs]) for j in range(9)},step_gradients=np.stack(gradients),**{f'parameter_gradient_{j}':np.stack([g[j] for g in parameter_gradients]) for j in range(20)},embedding=np.stack([h[0].numpy() for h in head_runs]),logit_codes=np.stack([h[1].numpy() for h in head_runs]))
   row=dict(head_mismatch=head_mismatch,parameter_gradient_norms=[float(np.linalg.norm(g)) for g in parameter_gradients[0]],parameter_repeat_exact=parameter_repeat,target=c['target'],condition=c['condition'],mismatch=mismatch,repeat_exact=repeat_exact,step_gradients=gradients[0].tolist(),arrays_path=str(path),arrays_sha256=sha(path),passed=not any(mismatch+head_mismatch) and repeat_exact);rows.append(row);print(json.dumps(row),flush=True)
  result=dict(execution_sha256=sha(P),cases=rows,all_passed=all(c['passed'] for c in rows),optimizer_updates=0,limits=p['policy']);(R/'4090-learned-grid-head-integration-result-v2.json').write_text(json.dumps(result,indent=2)+'\n');assert result['all_passed']
 except Exception:
  (OUT/'failure.log').write_text(traceback.format_exc());raise
if __name__=='__main__':{'lock':lock,'run':run}[sys.argv[1]]()
