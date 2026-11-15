"""Locked development-only Melo complete native output and internal alignment."""
import os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
os.environ['NLTK_DATA']=str(ROOT/'.local/wakeword/melo-nltk-v1')
os.environ['USE_TF']='0'
os.environ['USE_FLAX']='0'
os.environ['HF_HUB_OFFLINE']='1'
import argparse
import hashlib
import importlib.metadata as metadata
import json
import random
import subprocess
import sys
from melo_english_adapter_v1 import EnglishFrontend,load_acoustic_model,SOURCE,BERT,MODEL
sys.path.insert(0,str(ROOT/'research/wakeword_p4/experiments'))
R=ROOT/'research/wakeword_p4/results'
PLAN=R/'4090-melo-prospective-request-plan-v1.json'
LOCK=R/'4090-melo-development-execution-lock-v1.json'
OUT=ROOT/'.local/wakeword/melo-development-native-v1'
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def identity():
 import torch
 assert torch.cuda.is_available() and '4090' in torch.cuda.get_device_name(0)
 files=[Path(__file__),Path(__file__).with_name('melo_english_adapter_v1.py'),PLAN,R/'4090-melo-english-input-frontend-review-v1.json',ROOT/'research/wakeword_p4/experiments/synthetic_output_quality_v1.py']
 for folder in [MODEL,BERT]:files.extend(p for p in sorted(folder.iterdir()) if p.is_file())
 files.extend(p for p in sorted((SOURCE/'melo').rglob('*')) if p.is_file() and p.suffix in ['.py','.txt','.rep','.pickle'])
 files.extend(p for p in sorted((ROOT/'.local/wakeword/melo-nltk-v1').rglob('*')) if p.is_file())
 assert (MODEL/'checkpoint.pth') in files and (BERT/'model.safetensors') in files
 return dict(schema='melo-development-execution-lock-v1',files={str(p.relative_to(ROOT)):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in files},
  packages={n:metadata.version(n) for n in ['torch','transformers','huggingface-hub','numpy','scipy','soundfile','g2p-en','inflect','nltk','numba','tokenizers','safetensors']},
  upstream_source_commit=subprocess.check_output(['git','-C',str(SOURCE),'rev-parse','HEAD'],text=True).strip(),device=torch.cuda.get_device_name(0),dtype='float32',native_rate=44100,
  generation=json.loads(PLAN.read_text())['generation'],requests=144,role='development',
  runtime_exception='Upstream requirements mention transformers4.27.4; isolated English runtime uses4.46.1 with actual versions bound here. Direct upstream SynthesizerTrn/English/BERT implementation, adapter only replaces local BERT ID and copies exact Japanese distribute_phone helper AST to avoid unrelated language imports. No fake torchaudio or sentence split/pad.',
  raw_policy='Save exact3Dengine float32 waveform before mono vector and complete resample; also save full returned attention and per-input-token duration path. Internal conditioning durations are not human phone truth.',
  stopping='max_len=None; require native sample count equals returned alignment frames*512 and y_mask frames. No autoregressive token cap applies.',failure_policy='retain-all-stop-pack-no-substitution',scoring_authorized=False)
def main():
 parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','run']);a=parser.parse_args();ident=identity()
 if a.action=='prepare':
  assert not LOCK.exists();LOCK.write_text(json.dumps(ident,indent=2)+'\n');print(LOCK,flush=True);return
 assert json.loads(LOCK.read_text())==ident,'Execution identity changed'
 assert subprocess.check_output(['git','-C',str(ROOT),'show','origin/research-hye-nyx-voice-gpu:'+str(LOCK.relative_to(ROOT))])==LOCK.read_bytes()
 assert not OUT.exists();OUT.mkdir();(OUT/'execution-lock.json').write_bytes(LOCK.read_bytes())
 import numpy as np
 import soundfile as sf
 import torch
 from scipy.signal import resample_poly
 from synthetic_output_quality_v1 import audit_output
 requests=[r for r in json.loads(PLAN.read_text())['requests'] if r['role']=='development'];assert len(requests)==144
 try:
  frontend=EnglishFrontend();config=json.loads((MODEL/'config.json').read_text());model=load_acoustic_model(config)
  assert config['data']['sampling_rate']==44100 and config['data']['hop_length']==512
  for r in requests:
   assert config['data']['spk2id'][r['voice_id'].split(':')[-1]]==r['speaker_condition_id']
   random.seed(r['seed']);np.random.seed(r['seed']);torch.manual_seed(r['seed']);torch.cuda.manual_seed_all(r['seed'])
   with torch.inference_mode():
    info,(phones,tones,languages,bert,ja_bert)=frontend.tensors(r['text'],config,'cuda:0')
    output,attention,mask,_=model.infer(phones,torch.tensor([phones.shape[-1]],device='cuda:0'),torch.tensor([r['speaker_condition_id']],device='cuda:0'),tones,languages,bert,ja_bert,
       sdp_ratio=.2,noise_scale=.6,noise_scale_w=.8,length_scale=1/r['speed'],max_len=None)
   wave=output.detach().cpu().numpy();name=r['request_id'];engine=OUT/(name+'.engine.npy');np.save(engine,wave,allow_pickle=False)
   assert wave.dtype==np.float32 and wave.ndim==3 and wave.shape[:2]==(1,1)
   native=wave[0,0];raw=OUT/(name+'.native.npy');np.save(raw,native,allow_pickle=False)
   alignment=attention.detach().cpu().numpy();ap=OUT/(name+'.alignment.npy');np.save(ap,alignment,allow_pickle=False)
   assert alignment.ndim==4 and alignment.shape[:2]==(1,1) and alignment.shape[-1]==phones.shape[-1]
   frames=alignment[0,0].sum(axis=0);full_alignment=bool(len(native)==alignment.shape[2]*512 and mask.shape[-1]==alignment.shape[2] and int(frames.sum())==alignment.shape[2])
   pcm=resample_poly(native,160,441);wav=OUT/(name+'.wav');sf.write(wav,pcm,16000,subtype='PCM_16')
   record=dict(r,raw_path=raw.name,raw_sha256=sha(raw),path=wav.name,sha256=sha(wav),engine_path=engine.name,engine_sha256=sha(engine),engine_shape=list(wave.shape),native_samples=len(native),samples=len(pcm),generated_reference_end_sample=len(pcm),alignment_path=ap.name,alignment_sha256=sha(ap),alignment_shape=list(alignment.shape),input_frontend=info,
    input_conditioning_symbols=[config['symbols'][i] for i in info['input_phone_ids']],generated_conditioning_duration_frames=frames.astype(int).tolist(),generated_conditioning_duration_samples=(frames*512).astype(int).tolist(),generated_alignment_scope='Upstream model internal duration path over input phones including blanks. Not observed acoustic phone truth; do not infer human keyword end.',generated_phoneme_truth=None,full_returned_alignment=full_alignment)
   record['quality']=audit_output(OUT,record);record['passed']=bool(record['quality']['passed'] and full_alignment)
   with (OUT/'records.jsonl').open('a') as f:f.write(json.dumps(record)+'\n');f.flush();os.fsync(f.fileno())
   print(name,record['passed'],len(pcm)/16000,flush=True)
   if not record['passed']:raise RuntimeError('Retained failed output; stop pack: '+name)
  (OUT/'completed.json').write_text(json.dumps(dict(requests=144,passed=True))+'\n')
 except BaseException as e:
  (OUT/'failure.json').write_text(json.dumps(dict(type=type(e).__name__,error=str(e)))+'\n');raise
if __name__=='__main__':main()
