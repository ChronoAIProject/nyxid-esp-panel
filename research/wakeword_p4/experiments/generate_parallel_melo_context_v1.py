"""Same48 contexts in two existing development-only Melo conditions; once each."""
import os
os.environ.update(NLTK_DATA=str(__import__('pathlib').Path('.local/wakeword/melo-nltk-v1').resolve()),USE_TF='0',USE_FLAX='0',HF_HUB_OFFLINE='1',OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
import hashlib,importlib.metadata as md,json,random,shutil,subprocess,sys,time
from pathlib import Path
sys.path.insert(0,str(Path('research/wakeword_p4/generation').resolve()))
from melo_english_adapter_v1 import EnglishFrontend,load_acoustic_model,SOURCE,BERT,MODEL
R=Path('research/wakeword_p4/results');PLAN=R/'4090-parallel-melo-context-language-plan-v1.json';LOCK=R/'4090-parallel-melo-context-generation-lock-v1.json';OUT=Path('.local/wakeword/parallel-melo-context-language-v1')
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def plan():
 parent=R/'4090-parallel-prospective-language-plan-v1.json';original=json.loads(parent.read_text());config=json.loads((MODEL/'config.json').read_text());rows=[]
 for voice in ['EN-US','EN-BR']:
  for i,r in enumerate(original['requests']):rows.append(dict(r,voice='melo-existing-development:'+voice,speaker_condition=voice,speaker_condition_id=config['data']['spk2id'][voice],text_index=i,role='targeted-new-context-development-not-acceptance',speed=1.0,seed=731))
 assert len(rows)==96
 p=dict(schema='parallel-melo-context-language-plan-v1',parent_plan_sha256=sha(parent),requests=rows,models=original['models'],operating_profile=original['operating_profile'],generation=dict(sdp_ratio=.2,noise_scale=.6,noise_scale_w=.8,speed=1.0,max_len=None,native_rate=44100),policy='Same48prospectively locked texts, once in each existing development-only EN-US/EN-BR condition; no independent new-person claim. Entire waveforms, engine alignment, nativefloat32 and full16kPCM retained; no trim/normalization/score-based retry/replacement. Fixed C161/originalB40/parallel and original81600prime/.91796875/5/67policy. No training, threshold/voice/speed sweep, protected acceptance access or background/FAR exposure. Intended text and internal alignment are not phonetic truth. Full96replay and whole96ASR planned, no result filtering.')
 with PLAN.open('x') as f:json.dump(p,f,indent=2);f.write('\n')
 paths=[Path(__file__),PLAN,Path('research/wakeword_p4/generation/melo_english_adapter_v1.py')]
 for folder in [MODEL,BERT]:paths.extend(p for p in folder.iterdir() if p.is_file())
 paths.extend(p for p in (SOURCE/'melo').rglob('*') if p.is_file() and p.suffix in ['.py','.txt','.rep','.pickle'])
 paths.extend(p for p in Path('.local/wakeword/melo-nltk-v1').rglob('*') if p.is_file())
 lock=dict(schema='parallel-melo-context-generation-lock-v1',files={str(p):sha(p) for p in paths},versions={n:md.version(n) for n in ['torch','transformers','numpy','scipy','soundfile','g2p-en','nltk']},source_commit=subprocess.check_output(['git','-C',str(SOURCE),'rev-parse','HEAD'],text=True).strip(),rows=96)
 with LOCK.open('x') as f:json.dump(lock,f,indent=2);f.write('\n')
def run():
 assert subprocess.check_output(['git','show','origin/research-hye-nyx-voice-gpu:'+str(LOCK)])==LOCK.read_bytes();lock=json.loads(LOCK.read_text());p=json.loads(PLAN.read_text())
 for path,h in lock['files'].items():assert sha(path)==h,path
 for n,v in lock['versions'].items():assert md.version(n)==v
 assert shutil.disk_usage('.').free>100*1024**3
 import numpy as np,soundfile as sf,torch
 from scipy.signal import resample_poly
 torch.set_num_threads(2);torch.set_num_interop_threads(1);assert torch.cuda.is_available() and '4090' in torch.cuda.get_device_name(0)
 OUT.mkdir(exist_ok=False);frontend=EnglishFrontend();config=json.loads((MODEL/'config.json').read_text());model=load_acoustic_model(config)
 assert config['data']['sampling_rate']==44100 and config['data']['hop_length']==512
 for index,r in enumerate(p['requests']):
  random.seed(r['seed']);np.random.seed(r['seed']);torch.manual_seed(r['seed']);torch.cuda.manual_seed_all(r['seed']);start=time.monotonic()
  with torch.inference_mode():
   info,(phones,tones,languages,bert,ja_bert)=frontend.tensors(r['text'],config,'cuda:0')
   output,attention,mask,_=model.infer(phones,torch.tensor([phones.shape[-1]],device='cuda:0'),torch.tensor([r['speaker_condition_id']],device='cuda:0'),tones,languages,bert,ja_bert,sdp_ratio=.2,noise_scale=.6,noise_scale_w=.8,length_scale=1.0,max_len=None)
  engine=output.detach().cpu().numpy();alignment=attention.detach().cpu().numpy();y_mask=mask.detach().cpu().numpy();assert engine.dtype==np.float32 and engine.shape[:2]==(1,1) and np.isfinite(engine).all()
  assert len(engine[0,0])==alignment.shape[2]*512 and mask.shape[-1]==alignment.shape[2]
  native=engine[0];native_path=OUT/f'{index:03d}-native.npy';np.save(native_path,native,allow_pickle=False)
  detail=OUT/f'{index:03d}-engine-alignment.npz';np.savez_compressed(detail,engine=engine,alignment=alignment,mask=y_mask)
  v=resample_poly(native[0],160,441);pcm=np.rint(np.clip(v,-1,32767/32768)*32768).astype(np.int16);wav=OUT/f'{index:03d}-16k.wav';sf.write(wav,pcm,16000,subtype='PCM_16')
  row=dict(index=index,request=r,native_rate=44100,native_shape=list(native.shape),seconds=len(native[0])/44100,peak=float(np.max(np.abs(native))),native_outside_pcm_range=int(np.count_nonzero((native<-1)|(native>32767/32768))),resampled_outside_pcm_range=int(np.count_nonzero((v<-1)|(v>32767/32768))),elapsed_seconds=time.monotonic()-start,files=[dict(path=str(x),bytes=x.stat().st_size,sha256=sha(x)) for x in [native_path,wav,detail]],input_frontend=info,alignment_shape=list(alignment.shape),full_returned_alignment=True,phoneme_timing=None,mechanical_checks_only=True)
  (OUT/f'{index:03d}.json').write_text(json.dumps(row,indent=2)+'\n');print(json.dumps(dict(index=index,voice=r['speaker_condition'],seconds=row['seconds'])),flush=True)
 (OUT/'completed.json').write_text(json.dumps(dict(requests=96,lock_sha256=sha(LOCK),pronunciation_validated=False))+'\n')
if __name__=='__main__':{'plan':plan,'run':run}[sys.argv[1]]()
