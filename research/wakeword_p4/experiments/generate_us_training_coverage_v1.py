"""Execute the preregistered108-attempt training-only grid; retain all outputs."""
import argparse,hashlib,importlib.metadata,json,math,sys
from pathlib import Path
import numpy as np
import onnxruntime as ort
from piper_phonemize import phonemize_espeak
from scipy.signal import resample_poly
import soundfile as sf
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--source-sha256',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
assert sha(a.source/'source.json')==a.source_sha256
source=json.loads((a.source/'source.json').read_text());assert source['revision']=='ae9d8352b73c1b1750748843e753a504b00e05b0' and source['voices']==['lessac','ryan','ljspeech']
for name,info in source['files'].items():assert sha(a.source/name)==info['sha256'] and (a.source/name).stat().st_size==info['size_bytes']
a.output.mkdir(exist_ok=False)
versions={name:importlib.metadata.version(name) for name in ['onnxruntime','piper-phonemize','numpy','scipy','soundfile']}
plan=[('hey_nyx','Hey Nicks.'),('hey_nick','Hey Nick.'),('hey_next','Hey Next.'),('hey_mix','Hey Mix.')]
runtime=dict(source_sha256=a.source_sha256,script_sha256=sha(Path(__file__)),python=sys.version,versions=versions,phonemizer_seed=None,stochastic_seed_status='ONNX internal randomness; no exposed controllable seed claimed',grid=dict(voices=source['voices'],multipliers=[.9,1.,1.1],phrases=plan,draws=3),purpose='training-only; no model scoring')
(a.output/'run.json').write_text(json.dumps(runtime,indent=2)+'\n');run_sha=sha(a.output/'run.json');rows=[]
with (a.output/'manifest.jsonl').open('x') as manifest:
 for voice in source['voices']:
  model=a.source/voice/f'en_US-{voice}-medium.onnx';config_path=Path(str(model)+'.json');config=json.loads(config_path.read_text());assert config['num_speakers']==1
  ids=config['phoneme_id_map'];rate=config['audio']['sample_rate'];options=ort.SessionOptions();options.intra_op_num_threads=3;options.inter_op_num_threads=1
  model_digest=sha(model);config_digest=sha(config_path)
  session=ort.InferenceSession(str(model),options,providers=['CPUExecutionProvider']);assert len(session.get_outputs())==1
  for multiplier in [.9,1.,1.1]:
   for target,text in plan:
    sentences=phonemize_espeak(text,config['espeak']['voice']);assert len(sentences)==1;phones=sentences[0];assert all(x in ids for x in phones)
    tokens=list(ids['^'])+list(ids['_'])
    for phone in phones:tokens+=list(ids[phone])+list(ids['_'])
    tokens+=list(ids['$'])
    scales=np.array([config['inference']['noise_scale'],config['inference']['length_scale']*multiplier,config['inference']['noise_w']],np.float32)
    for draw in range(3):
     native=session.run(None,dict(input=np.array([tokens],np.int64),input_lengths=np.array([len(tokens)],np.int64),scales=scales))[0].reshape(-1)
     divisor=math.gcd(16000,rate);processed=resample_poly(native,16000//divisor,rate//divisor) if rate!=16000 else native.copy()
     stem=f'{voice}-{multiplier:.1f}-{target}-{draw}';raw_path=a.output/(stem+'-native.npy');float_path=a.output/(stem+'-16k.npy');wav=a.output/(stem+'.wav')
     np.save(raw_path,native);np.save(float_path,processed);sf.write(wav,processed,16000,subtype='PCM_16');pcm,pcmrate=sf.read(wav,dtype='int16');info=sf.info(wav)
     rms=float(np.sqrt(np.mean(processed.astype(np.float64)**2))) if len(processed) else 0.
     checks=dict(finite=bool(np.isfinite(native).all() and np.isfinite(processed).all()),format_mono_pcm16_16k=pcm.ndim==1 and pcmrate==16000 and info.subtype=='PCM_16',nonempty_nonzero=bool(len(native) and len(pcm) and np.any(pcm)),duration=.2<=len(pcm)/16000<=5,rms=rms>=.001,native_no_fullscale=bool(np.all(np.abs(native)<1)),resampled_no_fullscale=bool(np.all(np.abs(processed)<1)),pcm_no_endpoint=not bool(np.any((pcm==-32768)|(pcm==32767))),complete_length=len(pcm)==len(processed))
     row=dict(order=len(rows),role='us-coverage-training-only',split='training',voice=voice,speaker_id=0,target=target,text=text,phonemes=phones,token_ids=tokens,scales=scales.tolist(),draw=draw,length_multiplier=multiplier,native_rate=rate,native_samples=len(native),native_path=raw_path.name,native_sha256=sha(raw_path),float16k_path=float_path.name,float16k_sha256=sha(float_path),path=wav.name,sha256=sha(wav),pcm_sha256=hashlib.sha256(pcm.astype('<i2').tobytes()).hexdigest(),samples=len(pcm),rms=rms,trimmed=False,normalized=False,generated_reference_end_sample=len(pcm),native_durations=None,duration_limit='Waveform-only ONNX output; token durations unavailable',model_sha256=model_digest,config_sha256=config_digest,run_sha256=run_sha,checks=checks,quality_pass=all(checks.values()))
     manifest.write(json.dumps(row)+'\n');manifest.flush();rows.append(row)
  print(json.dumps({'voice':voice,'completed':len(rows)}),flush=True)
assert len(rows)==108
report=dict(scope='Preregistered108 training-only synthesis attempts; no retries/filtering/scoring',clips=len(rows),quality_pass=sum(x['quality_pass'] for x in rows),run_sha256=run_sha,manifest_sha256=sha(a.output/'manifest.jsonl'),source_sha256=a.source_sha256,script_sha256=sha(Path(__file__)),versions=versions,limits='Mechanical process checks only, no acoustic phonetic truth or independence claim. All native/resampled floats retained; no new-development/final data or wake-model inference. No automatic dataset promotion or training.')
(a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
