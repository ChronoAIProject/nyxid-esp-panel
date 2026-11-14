"""Four new raw continuation models on complete reused development; no scoring."""
import hashlib,importlib.metadata,json,subprocess,sys,traceback
from pathlib import Path
import numpy as np
R=Path('research/wakeword_p4/results');LOCK=R/'4090-hey-common-carrier-raw-development-execution-v1.json';OUT=Path('.local/wakeword/hey-common-carrier-raw-development-v1')
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
def lock(check_only=False):
 required=[R/'4090-hey-common-carrier-raw-fp32-export-result-v1.json',R/'4090-hey-common-carrier-raw-fp32-waveform-result-v1.json',R/'4090-hey-common-carrier-raw-fp32-waveform-independent-audit-v1.json',R/'4090-hey-common-carrier-raw-recognition-protocol-v1.json']
 for path in required:
  if not path.exists():print(json.dumps(dict(status='not_ready',missing=str(path))));return 2
 export,screen,audit,protocol=[json.loads(path.read_text()) for path in required]
 assert export['all_static_passed'] and screen['all_passed'] and audit['all_passed'] and audit['all540_saved_cases_audited'] and audit['original_result_sha256']==sha(required[1])
 numerical_execution=R/'4090-hey-common-carrier-raw-fp32-waveform-execution-v1.json'
 assert screen['execution_sha256']==sha(numerical_execution)
 numerical=json.loads(numerical_execution.read_text())
 assert numerical['files'][str(required[0])]==sha(required[0])
 assert numerical['models']==export['models']
 assert len(protocol['recognition']['grid'])==138
 if check_only:print(json.dumps(dict(status='ready',models=4,inputs=372)));return 0
 from collect_complete_development_v1 import runtime_identity_files
 paths=[*required,numerical_execution,Path(__file__),Path('research/wakeword_p4/training/features.py')]+runtime_identity_files()
 for name in ('score_common_carrier_raw_development_v1.py','audit_common_carrier_raw_collection_v1.py','audit_common_carrier_raw_scoring_v1.py','report_common_carrier_raw_groups_v1.py','fp32_event_policy_v1.py','hey_development_metrics_v1.py'):paths.append(Path(__file__).with_name(name))
 models=[]
 for model in export['models']:
  artifact=model['artifacts']['fp32'];assert sha(artifact['path'])==artifact['sha256'];paths.append(Path(artifact['path']))
  models.append(dict(target=model['target'],condition=model['condition'],path=artifact['path'],sha256=artifact['sha256']))
 assert {(m['target'],m['condition']) for m in models}=={(t,c) for t in ('hey_willow','hey_nyx') for c in ('control','command_context')}
 materials=protocol['recognition']['materials']
 for phase,material in materials.items():
  path=Path(material['path']);assert sha(path)==material['sha256'];paths.append(path);rows=json.loads(path.read_text())['records'];assert len(rows)==material['records']
  for row in rows:
   wav=Path(row['path'] if phase=='synthetic' else row['wav_path']);assert sha(wav)==(row['sha256'] if phase=='synthetic' else row['wav_sha256']);paths.append(wav)
   if 'frontend_path' in row:assert sha(row['frontend_path'])==row['frontend_sha256'];paths.append(Path(row['frontend_path']))
 with LOCK.open('x') as f:f.write(json.dumps(dict(files={str(path):sha(path) for path in paths},models=models,materials=materials,grid=protocol['recognition']['grid'],versions={n:importlib.metadata.version(n) for n in ('numpy','soundfile','ai-edge-litert','pymicro-features')},threshold_selected=False,acceptance_scored=False,retries=0,limits='All4 newly audited rawFP32 finalmodels x372 complete consumed inputs, twofresh/reset, raw probabilities/clocks retained. Requires own540numeric/phase audit, no inheritance, selection or freshdata.'),indent=2)+'\n')
 return 0
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
if __name__=='__main__':
 import argparse
 ap=argparse.ArgumentParser();ap.add_argument('action',choices=['lock','run']);ap.add_argument('--check-only',action='store_true');a=ap.parse_args();raise SystemExit(lock(a.check_only) if a.action=='lock' else run())
