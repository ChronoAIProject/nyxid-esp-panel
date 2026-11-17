"""Training-only long-history mechanism diagnostic, fixed original INT8, no fitting."""
import os
os.environ.update(OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
import hashlib,json,subprocess,sys,time
from pathlib import Path
import numpy as np
R=Path('research/wakeword_p4/results');PLAN=R/'4090-binary40-long-history-factorial-plan-v1.json';OUT=Path('.local/wakeword/binary40-long-history-factorial-v1');MODEL=Path('.local/wakeword/binary40-trained-int8-v1/int8.tflite')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def plan():
 old=R/'4090-binary40-context-stratified-original-v1.json';short=R/'4090-binary40-context-int8-original-v1.json';d=json.loads(old.read_text());s=json.loads(short.read_text());assert len(d['records'])==len(s['records'])==90
 files=[Path(__file__),old,short,Path(s['trace']['path']),MODEL,Path('research/wakeword_p4/training/features.py'),Path('research/wakeword_p4/experiments/binary40_stream_predict_v1.py')];sources=[]
 for row in d['records']:
  source=row['source'];neg=source['context_negative'];assert source['role']==neg['role']=='training' and source['label']==1 and neg['label']==0
  for x in [source,neg]:assert sha(x['native_path'])==x['native_sha256'];files.append(Path(x['native_path']))
  sources.append(source)
 p=dict(schema='binary40-long-history-factorial-plan-v1',files={str(p):sha(p) for p in files},sources=sources,repetitions=16,gap_samples=8160,prime_samples=81600,tail_samples=16000,policy='All original90training-only stratified sources (30conditions xbare/time/light), unchanged samevoice first HeyNick negative repeated16times, each complete native wave padded to480 then8160zeros, after81600zero prime. Compare equal-length zero history. Original deployedB40INT8, four crossed frontend/cache arms, fresh explicit state only fullstream start. Artificial offdiagonal diagnostic, never deployment boundary resets. Save complete uint16 frontend and all rawcodes; physical diagonals must equal full original-adapter replay bytes. Existing single-prefix INT8 reference reused, not rerun. Fixed1175fivecode sum and original prefix support ledger. No optimizer, coefficient/length/threshold/source sweep, dev or acceptance audio, human/independent exposure or P4 claim. Repeated same negative is a controlled long-prefix stressor, not natural conversation or a proposed training recipe.')
 with PLAN.open('x') as f:json.dump(p,f,indent=2);f.write('\n')
def run():
 assert subprocess.check_output(['git','show','origin/research-hye-nyx-voice-gpu:'+str(PLAN)])==PLAN.read_bytes();p=json.loads(PLAN.read_text())
 for path,h in p['files'].items():assert sha(path)==h,path
 import soundfile as sf
 from ai_edge_litert.interpreter import Interpreter,OpResolverType
 from binary40_stream_predict_v1 import predict
 sys.path.insert(0,str(Path('research/wakeword_p4/training').resolve()));from features import features
 def runtime(**kw):return Interpreter(num_threads=1,experimental_op_resolver_type=OpResolverType.BUILTIN_REF,**kw)
 it=runtime(model_path=str(MODEL));it.allocate_tensors();runner=it.get_signature_runner();ins=runner.get_input_details();outs=runner.get_output_details()
 for i in range(7):assert ins[f'cache{i}']['quantization']==outs[f'cache{i}_out']['quantization']
 def reset():return {f'cache{i}':np.full(ins[f'cache{i}']['shape'],ins[f'cache{i}']['quantization'][1],np.int8) for i in range(7)}
 def infer(q,state):
  codes=[]
  for at in range(0,len(q),3):
   o=runner(chunk=q[None,at:at+3,None,:],**state);codes.append(int(o['probability'].item())+128);state={f'cache{i}':o[f'cache{i}_out'].copy() for i in range(7)}
  return np.asarray(codes,np.uint8),state
 OUT.mkdir(exist_ok=False);records=[];arrays={};scale,zero=ins['chunk']['quantization'];start=time.monotonic()
 for idx,source in enumerate(p['sources']):
  neg,rate=sf.read(source['context_negative']['native_path'],dtype='int16');assert rate==16000 and neg.ndim==1
  target,rate=sf.read(source['native_path'],dtype='int16');assert rate==16000 and target.ndim==1
  unit=np.r_[neg,np.zeros((-len(neg))%480+8160,np.int16)];prefix=np.r_[np.zeros(81600,np.int16),np.tile(unit,16)];assert len(prefix)%480==0;k=((len(prefix)//160-2)//3)*3
  qs={};states={};full={};feature_arrays={};feature_hashes={}
  for name in ['silence','speech']:
   pre=np.zeros_like(prefix) if name=='silence' else prefix;whole=np.r_[pre,target,np.zeros(16000,np.int16)];raw=features(whole);x=np.clip(np.pad(raw,((0,(-len(raw))%3),(0,0))).astype(np.float32)/25.6,0,26)
   feature_arrays[name]=raw;feature_hashes[name]=hashlib.sha256(raw.tobytes()).hexdigest();qs[name]=np.clip(np.rint(x/scale)+zero,-128,127).astype(np.int8)
   _,states[name]=infer(qs[name][:k],reset());_,full[name],_=predict(x,str(MODEL),runtime)
  feature_path=OUT/f'{idx:03d}-frontend.npz';np.savez_compressed(feature_path,**feature_arrays);arms={}
  for front in ['silence','speech']:
   for cache in ['silence','speech']:
    codes,_=infer(qs[front][k:],{name:value.copy() for name,value in states[cache].items()})
    if front==cache:assert np.array_equal(codes,full[front][k//3:])
    support=(3*np.arange(k//3+1,k//3+len(codes)+1)+2)*160;valid=(support[4:]>=len(prefix))&(support[4:]<=len(prefix)+source['prefix_reference_samples']+4320);sums=np.convolve(codes.astype(np.int64),np.ones(5,np.int64),'valid');assert valid.any()
    key=front+'/'+cache;arms[key]=dict(max_sum5=int(sums[valid].max()),crossing=bool(np.any(sums[valid]>=1175)));arrays[f'{idx}_{front}_{cache}']=codes
  row=dict(index=idx,source=source,prefix_samples=len(prefix),history_seconds=(len(prefix)-81600)/16000,split_feature=k,feature_file=dict(path=str(feature_path),bytes=feature_path.stat().st_size,sha256=sha(feature_path)),raw_frontend_sha256=feature_hashes,arms=arms);records.append(row);print(json.dumps(dict(index=idx,history_seconds=row['history_seconds'],arms=arms)),flush=True)
 trace=R/'4090-binary40-long-history-factorial-traces-v1.npz';np.savez_compressed(trace,**arrays);out=dict(schema='binary40-long-history-factorial-original-v1',plan_sha256=sha(PLAN),records=records,trace=dict(path=str(trace),bytes=trace.stat().st_size,sha256=sha(trace)),all180_diagonal_byte_checks_passed=True,optimizer_updates=0,elapsed_seconds=time.monotonic()-start,policy=p['policy'])
 with (R/'4090-binary40-long-history-factorial-original-v1.json').open('x') as f:json.dump(out,f,indent=2);f.write('\n')
if __name__=='__main__':{'plan':plan,'run':run}[sys.argv[1]]()
