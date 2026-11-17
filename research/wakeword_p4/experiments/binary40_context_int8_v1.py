"""Fixed INT8 confirmation of the original all30voice training factorial."""
import hashlib,json,sys,subprocess,time
from pathlib import Path
import numpy as np
R=Path('research/wakeword_p4/results');PLAN=R/'4090-binary40-context-int8-plan-v1.json';MODEL=Path('.local/wakeword/binary40-trained-int8-v1/int8.tflite');FLOAT=R/'4090-binary40-context-stratified-original-v1.json'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def plan():
 d=json.loads(FLOAT.read_text());files=[Path(__file__),MODEL,FLOAT,Path(d['trace']['path']),Path('research/wakeword_p4/experiments/binary40_stream_predict_v1.py')]
 assert sha(MODEL)=='454058039dd41e729ce82ac7f33a84edec4af9554615a632e3b33d47de38009c'
 PLAN.write_text(json.dumps(dict(files={str(p):sha(p) for p in files},policy='All90same training-source factorial rows, original saved features and split, four INT8 arms. BUILTIN_REF1thread; exact cache feedback. Diagonal byte parity against full replay. Fixed sum5>=1175; unchanged prefix-ledger metric, no optimizer or threshold/policy change. Counterfactual off-diagonals not deployable reset strategies.'),indent=2)+'\n')
def run():
 assert subprocess.check_output(['git','show','origin/research-hye-nyx-voice-gpu:'+str(PLAN)])==PLAN.read_bytes()
 p=json.loads(PLAN.read_text())
 for f,h in p['files'].items():assert sha(f)==h,f
 from ai_edge_litert.interpreter import Interpreter,OpResolverType
 from binary40_stream_predict_v1 import predict
 def runtime(**kw):return Interpreter(num_threads=1,experimental_op_resolver_type=OpResolverType.BUILTIN_REF,**kw)
 it=runtime(model_path=str(MODEL));it.allocate_tensors();runner=it.get_signature_runner();ins=runner.get_input_details();outs=runner.get_output_details()
 assert outs['probability']['quantization']==(1/256,-128)
 for i in range(7):assert ins[f'cache{i}']['quantization']==outs[f'cache{i}_out']['quantization']
 def reset():return {f'cache{i}':np.full(ins[f'cache{i}']['shape'],ins[f'cache{i}']['quantization'][1],np.int8) for i in range(7)}
 def infer(q,states):
  codes=[]
  for at in range(0,len(q),3):
   o=runner(chunk=q[None,at:at+3,None,:],**states);codes.append(int(o['probability'].item())+128);states={f'cache{i}':o[f'cache{i}_out'].copy() for i in range(7)}
  return np.array(codes,np.uint8),states
 d=json.loads(FLOAT.read_text());records=[];arrays={};start=time.monotonic();scale,zero=ins['chunk']['quantization']
 with np.load(d['trace']['path']) as z:
  for row in d['records']:
   idx=row['index'];k=row['split_feature'];states={};qs={};full={}
   for h in ['silence','speech']:
    x=z[f'{idx}_{h}_features'];qs[h]=np.clip(np.rint(x/scale)+zero,-128,127).astype(np.int8)
    _,states[h]=infer(qs[h][:k],reset());_,full[h],_=predict(x,str(MODEL),runtime)
   arms={}
   for front in ['silence','speech']:
    for cache in ['silence','speech']:
     codes,_=infer(qs[front][k:],{key:value.copy() for key,value in states[cache].items()})
     if front==cache:assert np.array_equal(codes,full[front][k//3:])
     support=(3*np.arange(k//3+1,k//3+len(codes)+1)+2)*160;valid=(support[4:]>=row['prefix_samples'])&(support[4:]<=row['prefix_samples']+row['source']['prefix_reference_samples']+4320)
     sums=np.convolve(codes.astype(np.int64),np.ones(5,dtype=np.int64),'valid');key=f'{front}/{cache}'
     arms[key]=dict(max_sum5=int(sums[valid].max()),crossing=bool(np.any(sums[valid]>=1175)),float_crossing=row['arms'][key]['crossing']);arrays[f'{idx}_{front}_{cache}']=codes
   records.append(dict(index=idx,source=row['source'],arms=arms));print(json.dumps(dict(index=idx,arms=arms)),flush=True)
 trace=R/'4090-binary40-context-int8-traces-v1.npz';np.savez_compressed(trace,**arrays)
 out=dict(plan_sha256=sha(PLAN),model_sha256=sha(MODEL),records=records,trace=dict(path=str(trace),sha256=sha(trace),bytes=trace.stat().st_size),elapsed_seconds=time.monotonic()-start,all180_diagonal_byte_checks_passed=True,optimizer_updates=0,policy=p['policy'])
 (R/'4090-binary40-context-int8-original-v1.json').write_text(json.dumps(out,indent=2)+'\n')
if __name__=='__main__':{'plan':plan,'run':run}[sys.argv[1]]()
