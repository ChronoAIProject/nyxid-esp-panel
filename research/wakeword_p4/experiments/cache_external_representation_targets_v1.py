"""All4320fixed physical training sources; reuse authenticated600 prefix vectors."""
import hashlib,json,os,subprocess,sys,time
from pathlib import Path
os.environ.update(USE_TF='0',OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false')
import numpy as np
R=Path('research/wakeword_p4/results');META=R/'4090-frozen-history-teacher-metadata-math-preflight-v1.json';REC=R/'4090-external-representation-teacher-recovered-v3.json';V=Path('.local/wakeword/external-representation-teacher-v2/vectors.npz');MODEL=Path('.local/wakeword/wav2vec2-base-960h-v1');LOCK=R/'4090-external-representation-target-cache-plan-v1.json';B=Path('.local/wakeword/external-representation-target-cache-v1')
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def normalized(raw,mean):
 centered=raw.astype(np.float64)-mean;norm=np.linalg.norm(centered,axis=1,keepdims=True);assert np.all(norm>1e-12);return (centered/norm).astype(np.float32)
def plan():
 rows=json.loads(META.read_text())['inventory'];old=json.loads(REC.read_text())['records'];by={r['source_id']:r for r in old};assert len(rows)==4320 and len(by)==600
 inherited=json.loads((R/'4090-external-representation-teacher-plan-v2.json').read_text())['files'];paths=[Path(__file__),META,REC,V,R/'4090-external-representation-teacher-plan-v2.json',Path(__file__).with_name('external_representation_teacher_findings_v1.md')]
 for path,h in inherited.items():assert sha(path)==h,path
 for r in rows:
  s=r['source'];assert s['role']=='training' and sha(s['native_path'])==s['native_sha256'] and 400<=r['reference_samples']<=s['native_samples'];paths.append(Path(s['native_path']))
  if s['id'] in by:assert by[s['id']]['end_sample']==r['reference_samples'] and by[s['id']]['native_sha256']==s['native_sha256']
 p=dict(schema='external-representation-target-cache-plan-v1',files={**inherited,**{str(p):sha(p) for p in paths}},sources=4320,reuse600=600,new_prefix_inferences=3720,past_samples=49440,anchor_rule='First physical student release sample (3*(zero_based_call+1)+2)*160 >= prefix_samples+reference_samples+480; <=reference end+4320. Same anchor for quiet/speech views; original metadata clocks, not measured phonetic ends.',normalization='Global unweighted training raw-vector mean in float64, row L2 normalize centered float64 and cast float32; no labels or validation statistics. Save raw float32 and mean float64, authenticate normalized rows by SHA.',optimizer_updates=0,policy='Only original4320training inventory; reuse600same source/reference vectors, infer remaining3720past-prefix teacher vectors once. No full-suffix teacher targets, new audio, student optimizer, layer/threshold search, acceptance or qualification. Causal cutoff and physical anchor availability proved for every source. CPU2/interop1, RTX4090 mandatory, >=100GiB free.')
 with LOCK.open('x') as f:json.dump(p,f,indent=2);f.write('\n')
def run():
 assert subprocess.check_output(['git','show','origin/research-hye-nyx-voice-gpu:'+str(LOCK)])==LOCK.read_bytes();p=json.loads(LOCK.read_text())
 for path,h in p['files'].items():assert sha(path)==h,path
 subprocess.run(['/usr/bin/python3','research/wakeword_p4/tools/check_disk_budget.py','--expected-gib','.1'],check=True)
 rows=json.loads(META.read_text())['inventory'];old=json.loads(REC.read_text())['records'];by={r['source_id']:r for r in old}
 with np.load(V,allow_pickle=False) as z:previous=z['prefix'].copy()
 B.mkdir(exist_ok=False)
 try:
  import torch,soundfile as sf
  from transformers import Wav2Vec2Config,Wav2Vec2ForCTC,Wav2Vec2Processor
  torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;assert torch.cuda.is_available() and '4090' in torch.cuda.get_device_name()
  config=Wav2Vec2Config.from_pretrained(MODEL,local_files_only=True);config.mask_time_prob=0.;config.mask_feature_prob=0.
  teacher,info=Wav2Vec2ForCTC.from_pretrained(MODEL,config=config,local_files_only=True,output_loading_info=True);assert not any(info[k] for k in ['missing_keys','unexpected_keys','mismatched_keys','error_msgs']) and not hasattr(teacher.wav2vec2,'masked_spec_embed')
  teacher=teacher.cuda().eval();teacher.requires_grad_(False);versions={n:v._version for n,v in teacher.state_dict().items()};processor=Wav2Vec2Processor.from_pretrained(MODEL,local_files_only=True)
  raw=np.empty((4320,768),np.float32);records=[];reused=0;inferred=0;started=time.monotonic()
  for i,row in enumerate(rows):
   s=row['source'];end=row['reference_samples'];start=max(0,end-49440)
   if s['id'] in by:
    r=by[s['id']];assert r['start_sample']==start and r['end_sample']==end;vec=previous[r['index']].copy();assert hashlib.sha256(vec.tobytes()).hexdigest()==r['prefix_vector_sha256'];kind='reused-authenticated-prefix';reused+=1
   else:
    wave,rate=sf.read(s['native_path'],dtype='float32');assert rate==16000 and wave.ndim==1 and len(wave)==s['native_samples'] and np.isfinite(wave).all();values=processor(wave[start:end],sampling_rate=16000,return_tensors='pt').input_values.cuda()
    with torch.inference_mode():h=teacher.wav2vec2(values).last_hidden_state[0];assert h.shape==((end-start-400)//320+1,768);vec=h.mean(0).cpu().numpy()
    kind='new-prefix-inference';inferred+=1
   assert vec.shape==(768,) and vec.dtype==np.float32 and np.isfinite(vec).all();raw[i]=vec
   support=(3*np.arange(1,row['frames']//3+1)+2)*160;absolute_end=row['prefix_samples']+end;indices=np.flatnonzero(support>=absolute_end+480);assert len(indices)>0;anchor=int(indices[0]);assert support[anchor]<=absolute_end+4320 and end<=support[anchor]-row['prefix_samples']
   records.append(dict(inventory_index=i,source_id=s['id'],native_sha256=s['native_sha256'],engine=s['engine'],speaker_id=s['speaker_id'],text=s['text'],label=s['label'],start_sample=start,end_sample=end,prefix_samples=row['prefix_samples'],frames=row['frames'],student_anchor_zero_based_call=anchor,student_anchor_release_sample=int(support[anchor]),teacher_future_samples=0,provenance=kind,raw_vector_sha256=hashlib.sha256(vec.tobytes()).hexdigest()))
   if (i+1)%500==0:print(json.dumps(dict(sources=i+1,reused=reused,inferred=inferred,elapsed_seconds=time.monotonic()-started)),flush=True)
  assert reused==600 and inferred==3720 and len(records)==4320 and versions=={n:v._version for n,v in teacher.state_dict().items()} and not teacher.training and all(not v.requires_grad for v in teacher.parameters())
  mean=raw.astype(np.float64).mean(0);targets=normalized(raw,mean);normerror=float(np.abs(np.linalg.norm(targets.astype(np.float64),axis=1)-1).max());assert normerror<1e-6
  for r,v in zip(records,targets,strict=True):r['normalized_target_sha256']=hashlib.sha256(v.tobytes()).hexdigest()
  path=B/'targets.npz';np.savez_compressed(path,raw_prefix=raw,training_mean=mean,labels=np.asarray([r['label'] for r in records],np.int8))
  result=dict(schema='external-representation-target-cache-original-v1',plan_sha256=sha(LOCK),metadata_sha256=sha(META),gpu=torch.cuda.get_device_name(),torch=torch.__version__,strict_loading_info={k:sorted(v) if isinstance(v,set) else v for k,v in info.items()},teacher_parameter_versions_unchanged=True,optimizer_updates=0,reused_prefix_vectors=reused,new_prefix_inferences=inferred,records=records,normalized_target_max_norm_error=normerror,arrays=dict(path=str(path),bytes=path.stat().st_size,sha256=sha(path)),elapsed_seconds=time.monotonic()-started,limits=p['policy'])
  with (R/'4090-external-representation-target-cache-original-v1.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
  print(json.dumps(dict(completed=True,reused=reused,inferred=inferred,arrays=result['arrays'],elapsed_seconds=result['elapsed_seconds'])),flush=True)
 except Exception:
  import traceback
  (B/'failure.log').write_text(traceback.format_exc());raise
if __name__=='__main__':{'plan':plan,'run':run}[sys.argv[1]]()
