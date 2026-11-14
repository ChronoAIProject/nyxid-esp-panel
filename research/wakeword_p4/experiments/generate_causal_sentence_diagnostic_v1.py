"""Fixed complete-sentence training-source diagnostic; never independent holdout."""
import argparse,hashlib,json,math,os,subprocess,time
from pathlib import Path
import numpy as np
import onnxruntime as ort
from scipy.signal import resample_poly
import soundfile as sf
from piper_phonemize import phonemize_espeak
from synthetic_output_quality_v1 import audit_output
from generate_multispeaker_coverage_native_v1 import versions
R=Path('research/wakeword_p4/results');LOCK=R/'4090-causal-sentence-diagnostic-native-plan-v1.json';OUT=Path('.local/wakeword/causal-sentence-diagnostic-native-v1');NATIVE=Path('.local/wakeword/multispeaker-coverage-native-v1')
PREFIXES=('Hey Nicks','Hey Muse','Hey Nick','Hey Next','Hello')
CARRIERS=('', 'turn on the light','what time is it','set a timer')
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def plan():
 parent=R/'4090-multispeaker-coverage-native-plan-v1.json';old=json.loads(parent.read_text());native=[r for r in map(json.loads,(NATIVE/'records.jsonl').read_text().splitlines()) if 'broad' in r['arms']]
 lookup={(r['voice'],r['speaker_id'],r['text'],r['length_scale'],r['draw']):r for r in native};requests=[]
 for m in old['models']:
  cfg=json.loads(Path(m['config_path']).read_text());speaker=min(m['speaker_id_map'].values());name=next(k for k,v in m['speaker_id_map'].items() if v==speaker)
  for prefix in PREFIXES:
   for carrier in CARRIERS:
    text=prefix+(', '+carrier if carrier else '')+('?' if carrier=='what time is it' else '.')
    phones,=phonemize_espeak(text,cfg['espeak']['voice']);mapping=cfg['phoneme_id_map'];tokens=list(mapping['^'])+list(mapping['_'])
    for phone in phones:tokens+=list(mapping[phone])+list(mapping['_'])
    tokens+=list(mapping['$'])
    for speed in [.9,1.,1.1]:
     reuse=lookup.get((m['voice'],speaker,text,speed,0))
     if reuse:assert reuse['phonemes']==phones and reuse['token_ids']==tokens and reuse['model_sha256']==m['sha256'] and reuse['config_sha256']==m['config_sha256']
     requests.append(dict(id=f'case{len(requests):04}',voice=m['voice'],speaker_id=speaker,speaker_name=name,text=text,prefix=prefix,carrier=carrier,length_scale=speed,draw=0,label=int(prefix=='Hey Nicks'),target='hey_nyx' if prefix=='Hey Nicks' else 'confusion',phonemes=phones,token_ids=tokens,reuse=reuse))
 assert len(requests)==120
 files=[Path(__file__),parent,NATIVE/'records.jsonl',Path(__file__).with_name('synthetic_output_quality_v1.py'),R/'4090-multispeaker-training-role-admission-v1.json']
 p=dict(schema='causal-sentence-diagnostic-native-plan-v1',files={str(p):sha(p) for p in files},models=old['models'],requests=requests,versions=versions(),noise_scale=.667,noise_w=.8,provider='CPUExecutionProvider',threads=2,expected_new_gib=.1,policy='Prospective diagnostic lattice: both existingtraining engines/minnumeric speakerID0/fiveprefixes/fourcarriers/threeoriginalspeeds/draw0. Reuse exact matching original request bytes by hardlink; generate each missingrequest once. Save fullnativefloat32 engine tensor immediately, then entiremono/resample16k, no crop/gain/substitution/scorebaseddraw or pronunciationtruth claim. Exact existing TTS model/config/runtime/phoneme/token/noise settings. Preserve everyfailure and stop pack if any qualitycheck fails. Training-source diagnostic only, not independent development/acceptance or training admission; no protectedrole access. CPU ONNX synthesis backend is existing source engine; detector training remains RTX4090. Futureactualprimed recognition needs separate fixedplan.')
 with LOCK.open('x') as f:f.write(json.dumps(p,indent=2)+'\n')
 print(json.dumps(dict(plan_sha256=sha(LOCK),requests=120,reused=sum(r['reuse'] is not None for r in requests),new=sum(r['reuse'] is None for r in requests))))
def run():
 assert subprocess.check_output(['git','show','origin/research-hye-nyx-voice-gpu:'+str(LOCK)])==LOCK.read_bytes();p=json.loads(LOCK.read_text());assert versions()==p['versions']
 for path,h in p['files'].items():assert sha(path)==h,path
 subprocess.run(['/usr/bin/python3','research/wakeword_p4/tools/check_disk_budget.py','--expected-gib',str(p['expected_new_gib'])],check=True)
 OUT.mkdir(exist_ok=False);launch=dict(pid=os.getpid(),proc_starttime=Path('/proc/self/stat').read_text().split()[21],plan_sha256=sha(LOCK));(OUT/'launch.json').write_text(json.dumps(launch,indent=2)+'\n');sessions={};models={m['voice']:m for m in p['models']}
 for m in p['models']:
  assert sha(m['model_path'])==m['sha256'] and sha(m['config_path'])==m['config_sha256'];opt=ort.SessionOptions();opt.intra_op_num_threads=2;opt.inter_op_num_threads=1;opt.execution_mode=ort.ExecutionMode.ORT_SEQUENTIAL;sessions[m['voice']]=ort.InferenceSession(m['model_path'],sess_options=opt,providers=[p['provider']])
 rows=[]
 with (OUT/'records.jsonl').open('x') as log:
  for q in p['requests']:
   m=models[q['voice']];cfg=json.loads(Path(m['config_path']).read_text());row={k:v for k,v in q.items() if k!='reuse'};row.update(role='training-source-diagnostic',model_sha256=m['sha256'],config_sha256=m['config_sha256'],native_rate=cfg['audio']['sample_rate'],trimmed=False,random_seed=None,phoneme_durations=None,duration_interface='Unmodified ONNX waveform output only; no exposed token durations',reused_original=q['reuse'] is not None)
   try:
    if q['reuse']:
     prior=q['reuse'];row['original_native_id']=prior['id']
     for key,hkey,suffix in [('engine_tensor_path','engine_tensor_sha256','.engine.npy'),('raw_path','raw_sha256','.npy'),('path','sha256','.wav')]:
      source=NATIVE/prior[key];assert sha(source)==prior[hkey];dest=OUT/(q['id']+suffix);os.link(source,dest);row[key]=dest.name;row[hkey]=prior[hkey]
     for key in ['engine_output_shape','native_samples','samples','generated_reference_end_sample']:row[key]=prior[key]
    else:
     tokens=np.asarray([q['token_ids']],np.int64);outputs=sessions[q['voice']].run(None,dict(input=tokens,input_lengths=np.asarray([tokens.shape[1]],np.int64),scales=np.asarray([p['noise_scale'],q['length_scale'],p['noise_w']],np.float32),sid=np.asarray([q['speaker_id']],np.int64)));assert len(outputs)==1;tensor=outputs[0];path=OUT/(q['id']+'.engine.npy');np.save(path,tensor,allow_pickle=False);row.update(engine_tensor_path=path.name,engine_tensor_sha256=sha(path),engine_output_shape=list(tensor.shape));assert tensor.dtype==np.float32 and tensor.ndim==4 and tensor.shape[:3]==(1,1,1)
     native=tensor[0,0,0,:];raw=OUT/(q['id']+'.npy');np.save(raw,native,allow_pickle=False);row.update(raw_path=raw.name,raw_sha256=sha(raw),native_samples=len(native));assert np.isfinite(native).all();g=math.gcd(row['native_rate'],16000);pcm=resample_poly(native,16000//g,row['native_rate']//g);wav=OUT/(q['id']+'.wav');sf.write(wav,pcm,16000,subtype='PCM_16');row.update(path=wav.name,sha256=sha(wav),samples=len(pcm),generated_reference_end_sample=len(pcm))
    row['audit']=audit_output(OUT,row)
   except Exception as e:row['error']=f'{type(e).__name__}: {e}'
   rows.append(row);log.write(json.dumps(row)+'\n');log.flush()
 report=dict(schema='causal-sentence-diagnostic-native-original-v1',launch=launch,plan_sha256=sha(LOCK),requests=len(rows),reused=sum(r['reused_original'] for r in rows),failures=[r['id'] for r in rows if not r.get('audit',{}).get('passed',False)],records_path=str(OUT/'records.jsonl'),records_sha256=sha(OUT/'records.jsonl'),detector_invocations=0,training_admitted=False,policy=p['policy'])
 with (R/'4090-causal-sentence-diagnostic-native-original-v1.json').open('x') as f:f.write(json.dumps(report,indent=2)+'\n')
 print(json.dumps(report));assert not report['failures'],'Failures preserved; no regeneration or substitution'
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('action',choices=['plan','run']);x=a.parse_args();globals()[x.action]()
