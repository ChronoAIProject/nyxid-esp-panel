"""Original four raw FP32 models on complete reused development; no scoring."""
import hashlib,importlib.metadata,json,subprocess,sys,traceback
from pathlib import Path
import numpy as np
R=Path('research/wakeword_p4/results');LOCK=R/'4090-fp32-reused-development-execution-v1.json';OUT=Path('.local/wakeword/fp32-reused-development-v1')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def prepare(raw,samples):
 if type(samples)is not int or samples<=0:raise ValueError('Expected positive original sample count')
 if raw.dtype!=np.uint16 or raw.ndim!=2 or raw.shape[1]!=40 or len(raw)!=max(0,(samples+159)//160-2):raise ValueError('Incomplete native frontend')
 values=np.clip(np.pad(raw,((0,(-len(raw))%3),(0,0))).astype(np.float32)/25.6,0,26)
 support=(3*np.arange(1,len(values)//3+1,dtype=np.int64)+2)*160
 return values,support,np.minimum(support,samples)
def predict(values,model_path,interpreter):
 it=interpreter(model_path=model_path);it.allocate_tensors();a=it.get_input_details()[0];b=it.get_output_details()[0]
 assert a['dtype']==b['dtype']==np.float32 and list(a['shape'])==[1,3,40] and list(b['shape'])==[1,1]
 scores=np.empty(len(values)//3,np.float32)
 for i,j in enumerate(range(0,len(values),3)):
  it.set_tensor(a['index'],values[None,j:j+3]);it.invoke();scores[i]=it.get_tensor(b['index'])[0,0]
 if not np.isfinite(scores).all() or np.any(scores<0) or np.any(scores>1):raise ValueError('Invalid raw probability')
 return scores
def lock():
 scope=R/'4090-fp32-reused-development-prospective-v1.json';p=json.loads(scope.read_text())
 original=R/'4090-hey-development-custom-execution-v1.json';old=json.loads(original.read_text());files=dict(old['files']);files.update(p['files'])
 from collect_complete_development_v1 import runtime_identity_files
 paths=[scope,Path(__file__),Path('research/wakeword_p4/training/features.py'),Path('research/wakeword_p4/experiments/score_fp32_reused_development_v1.py')]+runtime_identity_files()
 materials={}
 for phase,key in [('synthetic','synthetic_manifest'),('background','background_manifest')]:
  path=Path(p[key]);rows=json.loads(path.read_text())['records'];materials[phase]=dict(path=str(path),sha256=sha(path),role='reused-development',records=len(rows));paths.append(path)
  for row in rows:
   wav=Path(row['path'] if phase=='synthetic' else row['wav_path']);expected=row['sha256'] if phase=='synthetic' else row['wav_sha256'];assert sha(wav)==expected;paths.append(wav)
   if 'frontend_path' in row:assert sha(row['frontend_path'])==row['frontend_sha256'];paths.append(Path(row['frontend_path']))
 assert materials['synthetic']['records']==360 and materials['background']['records']==12
 for m in p['models']:assert sha(m['path'])==m['sha256'];paths.append(Path(m['path']))
 for path in paths:files[str(path)]=sha(path)
 for f,h in files.items():assert sha(f)==h,f
 with LOCK.open('x') as f:f.write(json.dumps(dict(files=files,models=p['models'],materials=materials,grid=p['grid'],versions={n:importlib.metadata.version(n) for n in ['numpy','soundfile','ai-edge-litert','pymicro-features']},threshold_selected=False,acceptance_scored=False,retries=0,limits='All4 original raw FP32 exports x372 full consumed inputs, two fresh builtin-reference interpreters; no score selection, newdata or recognition admission.'),indent=2)+'\n')
def run():
 assert subprocess.check_output(['git','show','origin/research-hye-nyx-voice-gpu:'+str(LOCK)])==LOCK.read_bytes();p=json.loads(LOCK.read_text())
 for f,h in p['files'].items():assert sha(f)==h,f
 for n,v in p['versions'].items():assert importlib.metadata.version(n)==v
 OUT.mkdir(exist_ok=False)
 try:
  import soundfile as sf
  from ai_edge_litert.interpreter import Interpreter,OpResolverType
  sys.path.insert(0,str(Path('research/wakeword_p4/training').resolve()));from features import features
  def runtime(**kwargs):return Interpreter(num_threads=1,experimental_op_resolver_type=OpResolverType.BUILTIN_REF,**kwargs)
  models=[]
  for model in p['models']:
   name=model['target']+'-'+model['condition'];dest=OUT/name;dest.mkdir();records=[]
   for phase,material in p['materials'].items():
    rows=json.loads(Path(material['path']).read_text())['records'];group=dest/phase;group.mkdir()
    for i,row in enumerate(rows):
     wav=row['path'] if phase=='synthetic' else row['wav_path'];pcm,rate=sf.read(wav,dtype='int16');assert rate==16000 and pcm.ndim==1 and len(pcm)==row['samples'] and hashlib.sha256(pcm.astype('<i2').tobytes()).hexdigest()==row['pcm_sha256']
     raw=np.load(row['frontend_path'],allow_pickle=False) if 'frontend_path' in row else features(pcm)
     values,support,clocks=prepare(raw,row['samples']);first=predict(values,model['path'],runtime);second=predict(values,model['path'],runtime)
     trace=group/f'{i:06}.npz';np.savez(trace,probability=first,probability_repeat=second,sample_end=clocks,support_unclamped=support)
     exact=first.tobytes()==second.tobytes()
     receipt=dict(model=name,phase=phase,index=i,record=row,trace_path=str(trace),trace_sha256=sha(trace),input_sha256=hashlib.sha256(values.tobytes()).hexdigest(),raw_frontend_sha256=hashlib.sha256(raw.tobytes()).hexdigest(),steps=len(first),repeat_exact=exact)
     with (dest/'records.jsonl').open('a') as f:f.write(json.dumps(receipt)+'\n')
     assert exact,'Fresh repeat mismatch preserved';records.append(receipt);print(json.dumps({k:receipt[k] for k in ['model','phase','index','steps','repeat_exact']}),flush=True)
   report=dict(model=model,execution_sha256=sha(LOCK),records=records,all_passed=len(records)==372,threshold_selected=False)
   (dest/'report.json').write_text(json.dumps(report,indent=2)+'\n');models.append(dict(model=name,report_sha256=sha(dest/'report.json')))
  (OUT/'result.json').write_text(json.dumps(dict(execution_sha256=sha(LOCK),models=models,all_passed=len(models)==4,threshold_selected=False,acceptance_scored=False),indent=2)+'\n')
 except Exception:
  (OUT/'failure.log').write_text(traceback.format_exc());raise
if __name__=='__main__':{'lock':lock,'run':run}[sys.argv[1]]()
