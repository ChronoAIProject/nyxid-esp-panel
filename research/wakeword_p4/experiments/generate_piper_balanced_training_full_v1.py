"""Run the frozen8064 training-only attempts, preserving complete native outputs."""
import hashlib,importlib.metadata,json,subprocess,sys,time
from pathlib import Path
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
import torch
from piper_phonemize import phonemize_espeak
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
base=Path('.local/wakeword');source=base/'piper';out=base/'piper-balanced-training-full-v1'
plan=Path('research/wakeword_p4/experiments/piper_balanced_training_full_v1.md')
coverage=Path('research/wakeword_p4/results/4090-piper-training-phrase-coverage-v1.json')
meta=json.loads(coverage.read_text());ids=np.random.default_rng(726).permutation(904)[:672].tolist();assert ids==meta['training_ids']
modelpath=source/'models/en_US-libritts_r-medium.pt';configpath=Path(str(modelpath)+'.json')
assert sha(modelpath)=='e95ee53770bf598c354a6e6dbfc95ccb259aeeb501d35a86be8a767429ab0ff6'
assert sha(configpath)=='119118e510d0b8a7a0c8649a0668640d6db9b00239e874169d964853a8d15848'
revision=subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip();assert revision=='b5b36e266576693ea47db129fbfe6bbc06164762'
config=json.loads(configpath.read_text());assert config['audio']['sample_rate']==22050 and config['num_speakers']==904
assert torch.cuda.is_available();torch.set_num_threads(4)
sys.path.insert(0,str(source.resolve()));from generate_samples import generate_audio,get_phonemes
model=torch.load(modelpath,map_location='cpu',weights_only=False).eval().cuda()
out.mkdir(exist_ok=False)
phrases=[('hey_nyx','Hey Nicks.'),('hey_nick','Hey Nick.'),('hey_next','Hey Next.'),('hey_mix','Hey Mix.')]
run=dict(role='balanced-full-training-only',seed=197726,ids=ids,multipliers=[.9,1.,1.1],phrases=phrases,draws=1,device=torch.cuda.get_device_name(0),torch_cuda=torch.version.cuda,source_revision=revision,model_sha256=sha(modelpath),config_sha256=sha(configpath),plan_sha256=sha(plan),coverage_sha256=sha(coverage),script_sha256=sha(Path(__file__)),source_files={str(p.relative_to(source)):sha(p) for p in sorted(source.rglob('*.py')) if '.git' not in p.parts},source_diff=subprocess.check_output(['git','-C',str(source),'diff'],text=True),versions={n:importlib.metadata.version(n) for n in ['torch','numpy','scipy','soundfile','piper-phonemize']},python=sys.version,determinism_limit='Seed and order recorded, exact CUDA generation replay not asserted; no retries/normalization/trimming')
(out/'run.json').write_text(json.dumps(run,indent=2)+'\n');runsha=sha(out/'run.json');rows=[];started=time.monotonic();torch.manual_seed(197726)
with (out/'manifest.jsonl').open('x') as manifest,torch.inference_mode():
 for sid in ids:
  for speed in [.9,1.,1.1]:
   for target,text in phrases:
    phones=phonemize_espeak(text,config['espeak']['voice']);assert len(phones)==1 and all(p in config['phoneme_id_map'] for p in phones[0])
    tokens,clip=get_phonemes(config['espeak']['voice'],config,text);assert clip is None
    audio,duration=generate_audio(model,torch.LongTensor([sid]),torch.LongTensor([sid]),[tokens],.5,config['inference']['noise_scale'],config['inference']['noise_w'],speed,None)
    native=audio.detach().cpu().numpy()[0,0];durations=duration.detach().cpu().numpy()[0,0];assert native.dtype==np.float32 and len(durations)==len(tokens)
    processed=resample_poly(native,320,441);assert processed.dtype==np.float32
    stem=f'{sid}-{speed:.1f}-{target}';paths={k:out/f'{stem}{suffix}' for k,suffix in [('native','-native.npy'),('resampled','-16k.npy'),('durations','-durations.npy'),('wav','.wav')]}
    np.save(paths['native'],native);np.save(paths['resampled'],processed);np.save(paths['durations'],durations);sf.write(paths['wav'],processed,16000,subtype='PCM_16');pcm,rate=sf.read(paths['wav'],dtype='int16')
    rms=float(np.sqrt(np.mean(processed.astype(np.float64)**2))) if len(processed) else 0.
    checks=dict(finite=bool(np.isfinite(native).all() and np.isfinite(processed).all()),format_mono_pcm16_16k=pcm.ndim==1 and rate==16000 and sf.info(paths['wav']).subtype=='PCM_16',nonempty_nonzero=bool(len(native) and len(pcm) and np.any(pcm)),duration=.2<=len(pcm)/16000<=5,rms=rms>=.001,native_no_fullscale=bool(np.all(np.abs(native)<1)),resampled_no_fullscale=bool(np.all(np.abs(processed)<1)),pcm_no_endpoint=not bool(np.any((pcm==-32768)|(pcm==32767))),complete_length=len(pcm)==len(processed),duration_full_length=bool(np.isfinite(durations).all() and np.all(durations>=0) and durations.sum()==len(native)),resample_length=len(processed)==(len(native)*320+440)//441)
    row=dict(order=len(rows),split='training',role='balanced-full-training-only',speaker_id=sid,paired_speaker_ids=[sid,sid],target=target,text=text,phonemes=phones[0],token_ids=tokens,length_scale=speed,noise_scale=config['inference']['noise_scale'],noise_scale_w=config['inference']['noise_w'],slerp_weight=.5,draw=0,native_rate=22050,native_samples=len(native),samples=len(pcm),rms=rms,duration_samples=durations.tolist(),generated_reference_end_sample=len(pcm),trimmed=False,normalized=False,run_sha256=runsha,files={k:dict(path=p.name,sha256=sha(p),size_bytes=p.stat().st_size) for k,p in paths.items()},pcm_sha256=hashlib.sha256(pcm.astype('<i2').tobytes()).hexdigest(),checks=checks,quality_pass=all(checks.values()))
    manifest.write(json.dumps(row)+'\n');manifest.flush();rows.append(row)
  print(json.dumps(dict(completed=len(rows),total=8064,seconds=round(time.monotonic()-started,1))),flush=True)
assert len(rows)==8064
report=dict(role='All8064 frozen training-only attempts; no wake scoring or model training',clips=len(rows),quality_pass=sum(r['quality_pass'] for r in rows),failures=[dict(order=r['order'],checks={k:v for k,v in r['checks'].items() if not v}) for r in rows if not r['quality_pass']],run_sha256=runsha,manifest_sha256=sha(out/'manifest.jsonl'),script_sha256=sha(Path(__file__)),seconds=time.monotonic()-started,limits='Mechanical process checks and generation durations, not acoustic phonetic truth; no shared-pretraining/person independence or CUDA-bit-replay claim. All failures retained; no old-source repair or automatic training/promotion.')
(out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
