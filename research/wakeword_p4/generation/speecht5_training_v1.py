"""GPU-only training-source feasibility, never development/acceptance data."""
import os
os.environ['USE_TF']='0'
os.environ['USE_FLAX']='0'
import argparse
import hashlib
import importlib.metadata as metadata
import io
import json
from pathlib import Path
import subprocess
import sys
import zipfile
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'research/wakeword_p4/experiments'))
PLAN=ROOT/'research/wakeword_p4/results/4090-speecht5-training-request-plan-v1.json'
LOCK=ROOT/'research/wakeword_p4/results/4090-speecht5-training-execution-lock-v1.json'
OUT=ROOT/'.local/wakeword/speecht5-training-native-v1'
TTS=ROOT/'.local/wakeword/speecht5-tts-pinned-v1'
VOC=ROOT/'.local/wakeword/speecht5-hifigan-pinned-v1'
ZIP=ROOT/'.local/wakeword/speecht5-xvectors-pinned-v1/spkrec-xvect.zip'
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def identity():
 import torch
 import transformers.models.speecht5.modeling_speecht5 as implementation
 assert torch.cuda.is_available() and '4090' in torch.cuda.get_device_name(0)
 return dict(plan_sha256=sha(PLAN),generator_sha256=sha(__file__),implementation_sha256=sha(implementation.__file__),
  files={str(p.relative_to(ROOT)):dict(bytes=p.stat().st_size,sha256=sha(p)) for folder in [TTS,VOC] for p in sorted(folder.iterdir()) if p.is_file()},
  embedding_zip_sha256=sha(ZIP),packages={n:metadata.version(n) for n in ['torch','transformers','numpy','scipy','soundfile','sentencepiece']},
  device=torch.cuda.get_device_name(0),dtype='float32',native_rate=16000,generation=dict(threshold=.5,minlenratio=0.,maxlenratio=20.),
  quality_code_sha256=sha(ROOT/'research/wakeword_p4/experiments/synthetic_output_quality_v1.py'),
  stopping='Capture every prob_out logit; reject when decoder steps reach input_tokens*20/reduction_factor or last summed sigmoid<0.5.',
  role='training-source-feasibility-only',failure_policy='retain-all-stop-pack-no-substitution')
def main():
 p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','run']);a=p.parse_args();ident=identity()
 if a.action=='prepare':
  assert not LOCK.exists();LOCK.write_text(json.dumps(ident,indent=2)+'\n');print(LOCK,flush=True);return
 assert json.loads(LOCK.read_text())==ident
 subprocess.run(['git','-C',str(ROOT),'diff','--exit-code','HEAD','--',str(LOCK),__file__],check=True)
 assert subprocess.check_output(['git','-C',str(ROOT),'show','origin/research-hye-nyx-voice-gpu:'+str(LOCK.relative_to(ROOT))])==LOCK.read_bytes()
 assert not OUT.exists();OUT.mkdir();(OUT/'execution-lock.json').write_bytes(LOCK.read_bytes())
 import numpy as np
 import soundfile as sf
 import torch
 from transformers import SpeechT5Processor,SpeechT5ForTextToSpeech,SpeechT5HifiGan,set_seed
 from synthetic_output_quality_v1 import audit_output
 plan=json.loads(PLAN.read_text());assert all(r['role']=='training' for r in plan['requests'])
 model=SpeechT5ForTextToSpeech.from_pretrained(str(TTS),local_files_only=True).to('cuda:0').eval()
 vocoder=SpeechT5HifiGan.from_pretrained(str(VOC),local_files_only=True).to('cuda:0').eval()
 processor=SpeechT5Processor.from_pretrained(str(TTS),local_files_only=True)
 assert vocoder.config.sampling_rate==16000
 captured=[]
 hook=model.speech_decoder_postnet.prob_out.register_forward_hook(lambda m,args,out:captured.append(out.detach().cpu().clone()))
 try:
  with zipfile.ZipFile(ZIP) as z:
   for r in plan['requests']:
    captured.clear();set_seed(r['seed']);embedding_bytes=z.read(r['embedding_member']);assert hashlib.sha256(embedding_bytes).hexdigest()==r['embedding_sha256']
    embedding=np.load(io.BytesIO(embedding_bytes),allow_pickle=False);assert embedding.shape==(512,) and np.isfinite(embedding).all()
    inputs=processor(text=r['text'],return_tensors='pt').to('cuda:0')
    with torch.inference_mode():
     output=model.generate_speech(inputs['input_ids'],torch.tensor(embedding,device='cuda:0').unsqueeze(0),attention_mask=inputs['attention_mask'],vocoder=vocoder,threshold=.5,minlenratio=0.,maxlenratio=20.)
    native=output.detach().cpu().numpy();name=r['request_id'];raw=OUT/(name+'.native.npy');np.save(raw,native,allow_pickle=False)
    assert native.ndim==1 and native.dtype==np.float32
    logits=torch.stack(captured).numpy();lp=OUT/(name+'.stop-logits.npy');np.save(lp,logits,allow_pickle=False)
    path=OUT/(name+'.wav');sf.write(path,native,16000,subtype='PCM_16')
    maximum=int(inputs['input_ids'].shape[1]*20/model.config.reduction_factor)
    final_probability=float(torch.sigmoid(captured[-1]).sum(-1).item())
    record=dict(r,raw_path=raw.name,raw_sha256=sha(raw),path=path.name,sha256=sha(path),native_rate=16000,native_samples=len(native),samples=len(native),engine_shape=list(native.shape),trimmed=False,generated_reference_end_sample=len(native),generated_phonemes=None,generated_phone_durations=None,input_token_ids=inputs['input_ids'].cpu().tolist(),decoder_steps=len(captured),decoder_max_steps=maximum,final_stop_probability_sum=final_probability,reached_decoder_cap=len(captured)>=maximum,stop_logits_path=lp.name,stop_logits_sha256=sha(lp))
    record['quality']=audit_output(OUT,record);record['passed']=bool(record['quality']['passed'] and not record['reached_decoder_cap'] and final_probability>=.5)
    with (OUT/'records.jsonl').open('a') as f:f.write(json.dumps(record)+'\n');f.flush();os.fsync(f.fileno())
    print(name,record['passed'],len(native)/16000,flush=True)
    if not record['passed']:raise RuntimeError('Retained failure; stop pack: '+name)
  (OUT/'completed.json').write_text(json.dumps(dict(requests=len(plan['requests']),passed=True))+'\n')
 except BaseException as e:
  (OUT/'failure.json').write_text(json.dumps(dict(type=type(e).__name__,error=str(e)))+'\n');raise
 finally:hook.remove()
if __name__=='__main__':main()
