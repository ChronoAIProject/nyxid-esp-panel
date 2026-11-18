"""Training-only missing Piper common-name command cells; no wake scoring."""
import hashlib,importlib.metadata,json,os,subprocess,sys,time
from pathlib import Path
import numpy as np
R=Path('research/wakeword_p4/results');BASE=Path('.local/wakeword');OUT=BASE/'piper-nick-command-material-v1';PLAN=R/'4090-piper-nick-command-material-plan-v1.json'
MODEL=BASE/'piper/models/en_US-libritts_r-medium.pt';CONFIG=Path(str(MODEL)+'.json')
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(p):return json.loads(Path(p).read_text())
def versions():return {n:importlib.metadata.version(n) for n in ['torch','numpy','scipy','soundfile','piper-phonemize']}
def schedule():
 old=read(R/'4090-hey-broad-piper-material-plan-v1.json');path=Path(old['schedule']['path']);assert sha(path)==old['schedule']['sha256']
 render=read(R/'4090-hey-broad-render-plan-v1.json');rp=Path(render['schedule']['path']);assert sha(rp)==render['schedule']['sha256']
 anchors={r['order']:r for r in map(json.loads,rp.read_text().splitlines()) if r['engine']=='piper' and r['kind']=='positive'}
 selected=[r for r in map(json.loads,path.read_text().splitlines()) if r['target']=='hey_nyx' and r['carrier']!='bare'];assert len(selected)==6048
 rows=[]
 for r in selected:
  a=anchors[r['order']];assert a['text']==r['text'] and a['pair_key']['conditioning']==str(r['speaker_id']) and a['pair_key']['scale']==r['length_scale']
  assert sha(a['source_path'])==a['source_sha256']
  rows.append(dict(order=len(rows),text=r['text'].replace('Hey Nicks,','Hey Nick,',1),original_request=r,positive_anchor=a,seed=r['seed']))
 assert len({(r['original_request']['speaker_id'],r['original_request']['length_scale'],r['original_request']['carrier']) for r in rows})==6048
 return rows,{str(path):sha(path),str(rp):sha(rp),str(R/'4090-hey-broad-piper-material-plan-v1.json'):sha(R/'4090-hey-broad-piper-material-plan-v1.json'),str(R/'4090-hey-broad-render-plan-v1.json'):sha(R/'4090-hey-broad-render-plan-v1.json')}
def plan():
 rows,files=schedule();sys.path.insert(0,str((BASE/'piper').resolve()));from generate_samples import get_phonemes
 config=read(CONFIG);texts={}
 for text in sorted({r['text'] for r in rows}):
  ids,clip=get_phonemes(config['espeak']['voice'],config,text);assert clip is None;texts[text]=ids
 for p in [Path(__file__),Path(__file__).with_name('piper_nick_command_material_protocol_v1.md'),MODEL,CONFIG,*sorted((BASE/'piper').rglob('*.py'))]:
  if '.git' not in p.parts:files[str(p)]=sha(p)
 d=dict(schema='piper-nick-command-material-plan-v1',files=files,versions=versions(),rows=rows,text_tokens=texts,cells=6048,training_enabled=False,policy='Complete fixed training-only Piper Nick commands paired by original conditioning/speed/carrier/seed to existing Nicks positives; stochastic/text duration outputs differ. No heldout or wake-score selection, trimming, normalization, retries or partial-pack admission. Save native float/duration/resampled float/PCM and all mechanical failures. CPU2/interop1 GPU4090 reserve100GiB; expected8GiB. This generates material only, not a training recipe or efficacy claim.')
 with PLAN.open('x') as f:json.dump(d,f,indent=2);f.write('\n')
 print(json.dumps(dict(cells=len(rows),plan_sha256=sha(PLAN))))
def run():
 assert subprocess.check_output(['git','show','origin/research-hye-nyx-voice-gpu:'+str(PLAN)])==PLAN.read_bytes();p=read(PLAN)
 assert versions()==p['versions']
 for path,h in p['files'].items():assert sha(path)==h,path
 subprocess.run(['/usr/bin/python3','research/wakeword_p4/tools/check_disk_budget.py','--expected-gib','8'],check=True)
 import torch,soundfile as sf
 from scipy.signal import resample_poly
 sys.path.insert(0,str((BASE/'piper').resolve()));from generate_samples import generate_audio
 torch.set_num_threads(2);torch.set_num_interop_threads(1)
 assert torch.cuda.is_available() and '4090' in torch.cuda.get_device_name()
 OUT.mkdir(exist_ok=False);model=torch.load(MODEL,map_location='cpu',weights_only=False).eval().cuda();failures=[];start=time.time()
 with (OUT/'records.jsonl').open('x') as log,torch.inference_mode():
  for row in p['rows']:
   i=row['order'];r=row['original_request'];torch.manual_seed(row['seed']);torch.cuda.manual_seed_all(row['seed'])
   audio,durations=generate_audio(model,torch.LongTensor([r['speaker_id']]),torch.LongTensor([r['speaker_id']]),[p['text_tokens'][row['text']]],r['slerp_weight'],r['noise_scale'],r['noise_scale_w'],r['length_scale'],None)
   native=audio.cpu().numpy()[0,0];duration=durations.cpu().numpy()[0,0];resampled=resample_poly(native,320,441)
   raw=OUT/f'{i:05}.npz';np.savez_compressed(raw,native=native,durations=duration,resampled=resampled)
   wav=OUT/f'{i:05}.wav';sf.write(wav,resampled,16000,subtype='PCM_16');pcm,rate=sf.read(wav,dtype='int16')
   rms=float(np.sqrt(np.mean(resampled.astype(np.float64)**2)))
   checks=dict(finite=bool(np.isfinite(native).all() and np.isfinite(duration).all() and np.isfinite(resampled).all()),format=rate==16000 and pcm.ndim==1 and sf.info(wav).subtype=='PCM_16',duration=.2<=len(pcm)/16000<=10,rms=rms>=.001,native_no_fullscale=bool(np.all(np.abs(native)<1)),resampled_no_fullscale=bool(np.all(np.abs(resampled)<1)),pcm_no_endpoint=not bool(np.any((pcm==-32768)|(pcm==32767))),full_length=len(pcm)==len(resampled)==(len(native)*320+440)//441,durations_integer=bool(np.isfinite(duration).all() and np.all(duration>=0) and np.array_equal(duration,np.rint(duration))),token_count=len(duration)==len(p['text_tokens'][row['text']]),native_duration=float(np.sum(duration,dtype=np.float64))==len(native))
   record=dict(order=i,request=row,checks=checks,quality_pass=all(checks.values()),rms=rms,samples=len(pcm),files={str(q):dict(bytes=q.stat().st_size,sha256=sha(q)) for q in [raw,wav]},pcm_sha256=hashlib.sha256(pcm.astype('<i2').tobytes()).hexdigest())
   log.write(json.dumps(record)+'\n');log.flush()
   if not record['quality_pass']:failures.append(dict(order=i,checks=checks))
   if (i+1)%100==0:
    print(json.dumps(dict(completed=i+1,total=6048,failures=len(failures),elapsed_seconds=time.time()-start)),flush=True)
    subprocess.run(['/usr/bin/python3','research/wakeword_p4/tools/check_disk_budget.py','--expected-gib','8'],check=True)
 result=dict(schema='piper-nick-command-material-original-v1',plan_sha256=sha(PLAN),cells=6048,failures=failures,all_mechanical_checks_passed=not failures,records_sha256=sha(OUT/'records.jsonl'),gpu=torch.cuda.get_device_name(),elapsed_seconds=time.time()-start,training_enabled=False,limits=p['policy'])
 with (R/'4090-piper-nick-command-material-original-v1.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
 print(json.dumps(result),flush=True)
if __name__=='__main__':{'plan':plan,'run':run}[sys.argv[1]]()
