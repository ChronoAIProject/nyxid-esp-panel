"""Fixed training-only external encoder feasibility; no student or optimizer."""
import hashlib,json,os,subprocess,sys,time
from pathlib import Path
os.environ.update(USE_TF='0',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false')
R=Path('research/wakeword_p4/results');LOCK=R/'4090-external-representation-teacher-plan-v2.json';OUT=Path('.local/wakeword/external-representation-teacher-v2');MODEL=Path('.local/wakeword/wav2vec2-base-960h-v1')
SOURCE=R/'4090-binary40-teacher-source-audit-plan-v1.json'
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def plan():
 source=json.loads(SOURCE.read_text());rows=source['positives'];assert len(rows)==600 and len({(s['engine'],s['speaker_id']) for s in rows})==30
 paths=[Path(__file__),Path(__file__).with_name('external_representation_loading_compatibility_v2.md'),Path(__file__).with_name('external_representation_teacher_preflight_v1.md'),SOURCE,MODEL/'source.json']
 meta=json.loads((MODEL/'source.json').read_text());assert meta['revision']=='22aad52d435eb6dbaf354bdad9b0da84ce7d6156'
 assert meta['files']['model.safetensors']['sha256']=='8aa76ab2243c81747a1f832954586bc566090c83a0ac167df6f31f0fa917d74a'
 for name,d in meta['files'].items():
  path=MODEL/name;assert path.stat().st_size==d['bytes'] and sha(path)==d['sha256'];paths.append(path)
 for s in rows:
  assert s['role']=='training' and sha(s['native_path'])==s['native_sha256'] and 400<=s['prefix_reference_samples']<=s['native_samples'];paths.append(Path(s['native_path']))
 import transformers,inspect
 from transformers.models.wav2vec2 import modeling_wav2vec2,feature_extraction_wav2vec2
 for m in [modeling_wav2vec2,feature_extraction_wav2vec2]:paths.append(Path(inspect.getfile(m)))
 p=dict(schema='external-representation-teacher-plan-v2',files={str(p):sha(p) for p in paths},sources=rows,teacher_revision=meta['revision'],encoder_layer='last',past_samples=49440,conv_receptive_samples=400,conv_stride_samples=320,optimizer_updates=0,transformers=transformers.__version__,policy='All600training sources, prefix-only and full-suffix diagnostic, final768encoder layer, original processor, fixed leave-one-stratum cosine-centroid/margin0 plus duration-only comparison. No student/model selection, text decoding, relabeling, development/acceptance or deployment/P4 claim.')
 with LOCK.open('x') as f:json.dump(p,f,indent=2);f.write('\n')
def run():
 assert subprocess.check_output(['git','show','origin/research-hye-nyx-voice-gpu:'+str(LOCK)])==LOCK.read_bytes();p=json.loads(LOCK.read_text())
 for path,h in p['files'].items():assert sha(path)==h,path
 subprocess.run(['/usr/bin/python3','research/wakeword_p4/tools/check_disk_budget.py','--expected-gib','.1'],check=True)
 OUT.mkdir(exist_ok=False)
 import numpy as np,soundfile as sf,torch,transformers
 from transformers import Wav2Vec2Config,Wav2Vec2ForCTC,Wav2Vec2Processor
 torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 assert torch.cuda.is_available() and '4090' in torch.cuda.get_device_name() and transformers.__version__==p['transformers']
 processor=Wav2Vec2Processor.from_pretrained(MODEL,local_files_only=True)
 config=Wav2Vec2Config.from_pretrained(MODEL,local_files_only=True)
 original_mask=dict(time=config.mask_time_prob,feature=config.mask_feature_prob)
 config.mask_time_prob=0.;config.mask_feature_prob=0.
 from safetensors import safe_open
 with safe_open(MODEL/'model.safetensors',framework='pt') as checkpoint:assert not any('masked_spec_embed' in key for key in checkpoint.keys())
 teacher,info=Wav2Vec2ForCTC.from_pretrained(MODEL,config=config,local_files_only=True,output_loading_info=True)
 assert not hasattr(teacher.wav2vec2,'masked_spec_embed')
 assert not any(info[k] for k in ['missing_keys','unexpected_keys','mismatched_keys','error_msgs']),info
 teacher=teacher.cuda().eval();teacher.requires_grad_(False);assert all(v.dtype==torch.float32 and not v.requires_grad for v in teacher.parameters())
 versions={n:v._version for n,v in teacher.state_dict().items()};vectors=[];fullvectors=[];records=[];start_time=time.monotonic()
 def embed(wave):
  v=processor(wave,sampling_rate=16000,return_tensors='pt').input_values.cuda()
  with torch.inference_mode():h=teacher.wav2vec2(v).last_hidden_state[0]
  assert h.shape[1]==768 and torch.isfinite(h).all();return h
 try:
  with (OUT/'records.jsonl').open('x') as ledger:
   for i,s in enumerate(p['sources']):
    wave,rate=sf.read(s['native_path'],dtype='float32');assert rate==16000 and wave.ndim==1 and len(wave)==s['native_samples'] and np.isfinite(wave).all()
    end=s['prefix_reference_samples'];start=max(0,end-p['past_samples']);prefix=embed(wave[start:end]);full=embed(wave[start:])
    count=(end-start-400)//320+1;assert len(prefix)==count and len(full)>=count
    a=prefix.mean(0).cpu().numpy();b=full[:count].mean(0).cpu().numpy();vectors.append(a);fullvectors.append(b)
    cosine=float(np.dot(a.astype(np.float64),b)/(np.linalg.norm(a.astype(np.float64))*np.linalg.norm(b.astype(np.float64))))
    if end==len(wave):assert np.array_equal(a,b)
    row=dict(index=i,source_id=s['id'],engine=s['engine'],speaker_id=s['speaker_id'],text=s['text'],label=s['label'],native_sha256=s['native_sha256'],start_sample=start,end_sample=end,native_samples=len(wave),prefix_encoder_frames=count,full_encoder_frames=len(full),prefix_full_cosine=cosine,prefix_vector_sha256=hashlib.sha256(a.tobytes()).hexdigest(),full_diagnostic_vector_sha256=hashlib.sha256(b.tobytes()).hexdigest())
    records.append(row);ledger.write(json.dumps(row)+'\n');ledger.flush()
    if (i+1)%50==0:print(json.dumps(dict(sources=i+1,elapsed_seconds=time.monotonic()-start_time)),flush=True)
  x=np.asarray(vectors);y=np.asarray([r['label'] for r in records],bool);groups=[(r['engine'],r['speaker_id']) for r in records]
  def margins(values):
   values=np.asarray(values,np.float64);result=np.empty(len(values))
   for group in sorted(set(groups)):
    test=np.asarray([g==group for g in groups]);train=~test;center=values[train].mean(0);z=values-center;z/=np.maximum(np.linalg.norm(z,axis=1,keepdims=True),1e-12)
    means=np.asarray([z[train&(y==label)].mean(0) for label in [False,True]]);means/=np.maximum(np.linalg.norm(means,axis=1,keepdims=True),1e-12)
    result[test]=z[test]@(means[1]-means[0])
   return result
  margin=margins(x);duration=margins([[r['end_sample']-r['start_sample']] for r in records])
  path=OUT/'vectors.npz';np.savez_compressed(path,prefix=x,full_suffix_diagnostic=np.asarray(fullvectors),labels=y,cosine_centroid_margin=margin,duration_only_margin=duration)
  assert versions=={n:v._version for n,v in teacher.state_dict().items()} and not teacher.training and all(not v.requires_grad for v in teacher.parameters())
  summaries={}
  for name,m in [('prefix_embedding',margin),('duration_only',duration)]:
   by={}
   for engine in ['all',*sorted({r['engine'] for r in records})]:
    mask=np.asarray([engine=='all' or r['engine']==engine for r in records]);by[engine]=dict(positive_count=int(np.sum(mask&y)),positive_hits=int(np.sum(mask&y&(m>0))),negative_count=int(np.sum(mask&~y)),negative_activations=int(np.sum(mask&~y&(m>0))))
   summaries[name]=by
  out=dict(schema='external-representation-teacher-original-v2',original_mask_config=original_mask,effective_mask_config=dict(time=0.,feature=0.),plan_sha256=sha(LOCK),gpu=torch.cuda.get_device_name(),torch=torch.__version__,transformers=transformers.__version__,strict_loading_info=info,optimizer_updates=0,teacher_parameter_versions_unchanged=True,teacher_parameter_count=sum(v.numel() for v in teacher.parameters()),summaries=summaries,records=[dict(r,centroid_margin=float(margin[i]),duration_only_margin=float(duration[i])) for i,r in enumerate(records)],files={str(f):dict(bytes=f.stat().st_size,sha256=sha(f)) for f in [path,OUT/'records.jsonl']},elapsed_seconds=time.monotonic()-start_time,limits=p['policy'])
  with (R/'4090-external-representation-teacher-original-v2.json').open('x') as f:json.dump(out,f,indent=2);f.write('\n')
  print(json.dumps(dict(completed=True,summaries=summaries,elapsed_seconds=out['elapsed_seconds'])),flush=True)
 except Exception:
  import traceback
  (OUT/'failure.log').write_text(traceback.format_exc());raise
if __name__=='__main__':{'plan':plan,'run':run}[sys.argv[1]]()
