"""Prospectively locked original-step complete temporal TF integration probe."""
import hashlib,json,subprocess,sys,traceback,time
from pathlib import Path
import numpy as np
R=Path('research/wakeword_p4/results');P=R/'4090-learned-grid-paired-smoke-execution-v1.json';OUT=Path('.local/wakeword/learned-grid-paired-smoke-v1')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def lock():
 parent=R/'4090-learned-grid-encoder-parity-execution-v1.json';base=json.loads(parent.read_text());files=dict(base['files'])
 for f in [Path(__file__).with_name('learned_grid_head_v2.py'),R/'4090-matched-internal-boundaries-original-result-v1.json',parent,Path(__file__),Path(__file__).with_name('learned_grid_stream_encoder_v3.py'),Path(__file__).with_name('learned_grid_requant_v1.py'),R/'4090-learned-grid-encoder-parity-result-v1.json']:files[str(f)]=sha(f)
 for name in ['schedule_hey_broad_v1.py','hey_batch_schedule_v1.py','audit_hey_broad_batches_v1.py','learned_grid_head_batched_v1.py','learned_grid_stream_encoder_batched_v1.py','learned_grid_requant_batched_v1.py','compile_learned_grid_parameters_v1.py','compile_learned_grid_nonuniform_v1.py']:files[str(Path(__file__).with_name(name))]=sha(Path(__file__).with_name(name))
 for f in [R/'4090-learned-grid-smoke-original-batches-v1.json',R/'4090-learned-activation-grid-constraints-v1.json']:files[str(f)]=sha(f)
 batches=json.loads((R/'4090-learned-grid-smoke-original-batches-v1.json').read_text())
 for b in batches['cases']:files[b['arrays_path']]=sha(b['arrays_path'])
 prior=json.loads((R/'4090-learned-grid-encoder-parity-result-v1.json').read_text())
 for c in prior['cases']:
  if c['factor']==1:files[c['arrays_path']]=sha(c['arrays_path'])
 captures=json.loads((R/'4090-matched-internal-boundaries-original-result-v1.json').read_text())
 for cap in captures['cases']:
  for key in ('captures_path','metadata_path'):files[cap[key]]=sha(cap[key])
 for target in ('hey_willow','hey_nyx'):
  file=Path('.local/wakeword/hey-broad-schedules-v1')/(target+'.jsonl');files[str(file)]=sha(file)
 P.write_text(json.dumps(dict(files=files,cases=[dict(c,training_arm=arm) for c in prior['cases'] if c['factor']==1 and c['condition']=='flat' for arm in ('fixed_grid','learned_step')],sources=base['sources'],models=[c for c in base['compiled'] if c['factor']==1],policy='Disposable paired2updates Willow/Nyxflat x fixed_grid/learned_step, twicefresh. Identical originalFP32folded init/20Wb/fixedweightgrids/Adam1e-5 beta.9/.999 eps1e-7/causal225terminal224/negative20; only learned_step updates9logsteps; fixedgridratios0. First2originalfull128batches, exactalltensorSHA guard. Nocheckpoint selection. Directfinal2export compare9encoder/embedding/logit vs actualcomplete originalrequest0; parameters/Adam/modelbytes/outputs repeatexact. Timing measured mechanically, no throughput guarantee/formaltraining/recognition/freshdata.'),indent=2)+'\n')
def run():
 assert subprocess.check_output(['git','show','origin/research-hye-nyx-voice-gpu:'+str(P)])==P.read_bytes();p=json.loads(P.read_text())
 for f,h in p['files'].items():assert sha(f)==h
 import tensorflow as tf
 tf.config.experimental.enable_op_determinism()
 from ai_edge_litert.interpreter import Interpreter,OpResolverType
 from schedule_hey_broad_v1 import sampler
 from hey_batch_schedule_v1 import batch_digest
 from learned_grid_stream_encoder_batched_v1 import forward
 from learned_grid_stream_encoder_v3 import forward as single_forward
 from learned_grid_head_batched_v1 import forward as head_forward
 from learned_grid_head_v2 import forward as single_head
 from compile_learned_grid_parameters_v1 import compile_parameters
 batches=json.loads((R/'4090-learned-grid-smoke-original-batches-v1.json').read_text());mapping=json.loads((R/'4090-learned-activation-grid-constraints-v1.json').read_text())
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
   batch=next(b for b in batches['cases'] if b['target']==c['target'])
   with np.load(batch['arrays_path'],allow_pickle=False) as ba:
    sample_q=[np.clip(np.rint(ba['x'][index]/configs[0]['input_step'])+configs[0]['input_zero'],-128,127).astype(np.int8) for index in range(128)]
    labels=[float(ba['y'][index,0]) for index in range(128)];weights=[float(ba['weight'][index]) for index in range(128)]
   group=next(x['groups'] for x in mapping['models'] if (x['target'],x['condition'])==(c['target'],c['condition']))
   selected=[next(j for j,g in enumerate(group) if any(m['graph']==0 and m['tensor']==cfg['index'] for m in g['members'])) for cfg in configs]
   runs=[];modelbytes=[];metrics=[];losses=[]
   for repeat in range(2):
    parameters=[(tf.Variable(w,dtype=tf.float64),tf.Variable(b,dtype=tf.float64)) for w,b in initial];head_parameters=[tf.Variable(v,dtype=tf.float64) for v in head_initial];flat=[v for pair in parameters for v in pair]+head_parameters
    ratios=tf.Variable(np.zeros(9),dtype=tf.float64);variables=[ratios]+flat
    trainvars=variables if c['training_arm']=='learned_step' else flat
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5,beta_1=.9,beta_2=.999,epsilon=1e-7);optimizer.build(trainvars)
    env=sampler(c['target']);schedule=Path('.local/wakeword/hey-broad-schedules-v1')/(c['target']+'.jsonl');expected=[json.loads(l) for l in schedule.read_text().splitlines()[:2]];history=[];started=time.monotonic()
    for stepindex in range(2):
     bx,by,bw=env['batch'](mine=False,augment=False);digest=batch_digest(bx,by,bw);assert digest==expected[stepindex]['tensor_sha256']
     iq=np.clip(np.rint(bx/configs[0]['input_step'])+configs[0]['input_zero'],-128,127).astype(np.int8)
     with tf.GradientTape() as tape:
      val=forward(iq,configs,ratios,tf,parameters);_,_,raw,_=head_forward(val[-1],configs[-1]['output_step'],ratios[-1],hc,head_parameters,False,tf)
      loss=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.constant(by[:,0],tf.float64),logits=raw[:,-1,0])*tf.constant(bw,tf.float64))
     grads=tape.gradient(loss,trainvars);assert all(g is not None and np.isfinite(g.numpy()).all() for g in grads);assert np.isfinite(loss.numpy())
     optimizer.apply_gradients(zip(grads,trainvars));assert int(optimizer.iterations)==stepindex+1 and all(np.isfinite(v.numpy()).all() for v in variables)
     history.append(dict(step=stepindex+1,batch_sha256=digest,loss=float(loss.numpy())))
    elapsed=time.monotonic()-started
    assert (np.any(ratios.numpy()!=0) if c['training_arm']=='learned_step' else np.array_equal(ratios.numpy(),np.zeros(9)))
    assert any(not np.array_equal(v.numpy(),old) for v,old in zip(flat,[a for pair in initial for a in pair]+list(head_initial),strict=True))
    factor={g:float(np.exp(np.clip(ratios.numpy()[j],-np.log(16),np.log(16)))) for j,g in enumerate(selected)}
    pairs=[tuple(v.numpy() for v in pair) for pair in parameters]+[tuple(v.numpy() for v in head_parameters)]
    compiled,receipt=compile_parameters(Path(model['path']).read_bytes(),group,factor,pairs);modelbytes.append(compiled)
    compiled_path=OUT/(c['target']+'-'+c['condition']+'-'+c['training_arm']+'-'+str(repeat)+'.tflite');compiled_path.write_bytes(compiled)
    values=forward(q[None],configs,ratios,tf,parameters);embedding,logitcodes,rawlogit,_=head_forward(values[-1],configs[-1]['output_step'],ratios[-1],hc,head_parameters,c['condition']=='pool',tf)
    actual=Interpreter(model_path=str(compiled_path),experimental_op_resolver_type=OpResolverType.BUILTIN_REF,num_threads=1,experimental_preserve_all_tensors=True);actual.allocate_tensors();output=actual.get_output_details()[0]['index'];traces={str(cfg['index']):[] for cfg in configs};traces['embedding']=[];traces['logit']=[];traces['uint8']=[]
    for at in range(0,len(q),3):
     actual.set_tensor(0,q[None,at:at+3]);actual.invoke()
     for cfg in configs:traces[str(cfg['index'])].append(actual.get_tensor(cfg['index']).copy())
     traces['embedding'].append(actual.get_tensor(ix).copy());traces['logit'].append(actual.get_tensor(oy).copy());traces['uint8'].append(actual.get_tensor(output).copy())
    arrays={k:np.stack(v) for k,v in traces.items()};unequal=[]
    for j,cfg in enumerate(configs):
     tfcodes=values[j].numpy();arrays['tf_'+str(cfg['index'])]=tfcodes;unequal.append(int(np.count_nonzero(np.rint(tfcodes)!=arrays[str(cfg['index'])].reshape(tfcodes.shape))))
    for key,val in [('embedding',embedding),('logit',logitcodes)]:
     v=val.numpy();arrays['tf_'+key]=v;unequal.append(int(np.count_nonzero(np.rint(v)!=arrays[key].reshape(v.shape))))
    for j,v in enumerate(variables):arrays['variable_'+str(j)]=v.numpy()
    for j,v in enumerate(optimizer.variables):arrays['optimizer_'+str(j)]=v.numpy()
    path=OUT/(c['target']+'-'+c['condition']+'-'+c['training_arm']+'-'+str(repeat)+'.npz');np.savez_compressed(path,**arrays);runs.append(arrays);metrics.append(dict(history=history,elapsed_training_seconds=elapsed,repeat=repeat,mismatch=unequal,model_path=str(compiled_path),model_sha256=sha(compiled_path),arrays_path=str(path),arrays_sha256=sha(path),compiler=receipt));losses.append(float(loss.numpy()))
   exact=modelbytes[0]==modelbytes[1] and losses[0]==losses[1] and all(runs[0][k].tobytes()==runs[1][k].tobytes() for k in runs[0])
   row=dict(target=c['target'],condition=c['condition'],training_arm=c['training_arm'],losses=losses,labels=labels,weights=weights,repeats=metrics,repeat_exact=exact,passed=exact and all(not any(m['mismatch']) for m in metrics));rows.append(row);print(json.dumps(dict(target=c['target'],condition=c['condition'],training_arm=c['training_arm'],repeat_exact=exact,mismatch=[m['mismatch'] for m in metrics],passed=row['passed'])),flush=True)
  result=dict(execution_sha256=sha(P),cases=rows,all_passed=all(c['passed'] for c in rows),optimizer_updates_per_repeat=2,limits=p['policy']);(R/'4090-learned-grid-paired-smoke-result-v1.json').write_text(json.dumps(result,indent=2)+'\n');assert result['all_passed']
 except Exception:
  (OUT/'failure.log').write_text(traceback.format_exc());raise
if __name__=='__main__':{'lock':lock,'run':run}[sys.argv[1]]()
