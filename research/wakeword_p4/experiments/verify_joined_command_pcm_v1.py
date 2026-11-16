"""Complete selected training PCM/native integrity admission, no inference."""
import hashlib,json
from pathlib import Path
import numpy as np
import soundfile as sf
R=Path('research/wakeword_p4/results');B=Path('.local/wakeword')
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
 joined=R/'4090-matchbox-joined-command-support-v1.json';p=json.loads(joined.read_text());aux=R/'4090-hey-broad-aux-material-plan-v1.json';roles=json.loads(aux.read_text())['kokoro'];allowed=set(roles['training_voices']);assert allowed.isdisjoint(roles['protected_roles']['validation']+roles['protected_roles']['testing'])
 posowner=B/'hey-broad-kokoro-native-v1/records.jsonl';nativepos={}
 for line in posowner.open():
  r=json.loads(line);w=Path(r['files']['wav']['path']);w=w if str(w).startswith('.local/') else posowner.parent/w;nativepos[str(w)]=r
 negplan=R/'4090-hey-common-carrier-render-execution-v1.json';npplan=json.loads(negplan.read_text());nativesneg={x['order']:x['native'] for x in npplan['requests']}
 renders={r['path']:r for f in [B/'hey-broad-training-rendered-v1/records.jsonl',B/'hey-common-carrier-rendered-v1/records.jsonl'] for r in map(json.loads,f.read_text().splitlines())}
 seen={};files={};count=0;maxpeak=0;metadata=[]
 for g in p['groups']:
  assert g['conditioning'] in allowed
  for label,row in g['classes'].items():
   assert sha(row['path'])==row['sha256'];wave,rate=sf.read(row['path'],dtype='int16');assert rate==16000 and wave.ndim==1 and len(wave)==row['samples'] and sf.info(row['path']).subtype=='PCM_16'
   assert hashlib.sha256(wave.astype('<i2').tobytes()).hexdigest()==row['pcm_sha256'];assert not np.any((wave==-32768)|(wave==32767));assert np.any(wave);maxpeak=max(maxpeak,int(np.abs(wave.astype(np.int32)).max()));count+=1
   render=renders[row['path']];assert render['quality_pass'] and all(render['checks'].values())
   if label=='1':
    nativepath=render['source']['source_path'];native=nativepos[nativepath];owner=posowner
   else:
    native=nativesneg[render['native_order']];nativepath=native['files']['wav']['path'];owner=None
   req=native['request'];assert req['split']=='training' and req['conditioning_id']==g['conditioning'] and float(req['speed'])==g['speed'] and req['carrier']==g['carrier'] and req['text']==row['text'];assert native['quality_pass'] and all(native['checks'].values()) and not native['trimmed'] and not native['normalized']
   if nativepath in seen:continue
   seen[nativepath]=True
   for kind,d in native['files'].items():
    f=Path(d['path']);f=f if str(f).startswith('.local/') else owner.parent/f;assert sha(f)==d['sha256'];files[str(f)]=d['sha256']
   w,rate=sf.read(nativepath,dtype='int16');assert rate==16000 and w.ndim==1 and len(w)==native['samples'];assert hashlib.sha256(w.astype('<i2').tobytes()).hexdigest()==native['pcm_sha256'];assert np.any(w) and not np.any((w==-32768)|(w==32767))
   md=native['metadata'];assert md.get('phonemes') and md.get('token_ids') and 'duration' in native['files'];metadata.append(dict(path=nativepath,text=req['text'],conditioning=req['conditioning_id'],phonemes=md['phonemes'],token_ids=md['token_ids'],duration=native['files']['duration'],native_record_sha256=hashlib.sha256(json.dumps(native,sort_keys=True).encode()).hexdigest()))
 assert count==3240 and len(seen)==1080
 paths=[Path(__file__),joined,aux,posowner,negplan,R/'4090-hey-broad-kokoro-generation-plan-v1.json',R/'4090-hey-common-carrier-kokoro-generation-execution-v1.json']
 out=dict(schema='joined-command-pcm-admission-v1',files={str(f):sha(f) for f in paths},native_files=files,native_metadata=metadata,rendered_pcm_verified=count,native_requests_verified=len(seen),native_file_count=len(files),maximum_rendered_pcm16_abs=maxpeak,all_selected_training_conditioning_ids_admitted=True,all_pcm_hashes_format_nonempty_nonclipping_full_length_passed=True,policy='Existing complete training sources only. Nativefiles including phones/tokens/duration retained andhashchecked; generatorintent/processquality, nothumanphonetictruth ornewindependentspeakers. No trimming, waveformrewrite, generation, recognition ortraining. Generationruntime/modelprovenance bound viaoriginal executionplans, no claim currentlyinstalledruntime equalsgeneration. Positive/negativeconditioning shared, upstreamphysical/pretrainingidentity unknown. Noheldoutaccess orqualification.')
 with (R/'4090-joined-command-pcm-admission-v1.json').open('x') as f:json.dump(out,f,indent=2);f.write('\n')
 print(json.dumps(dict(rendered=count,native_requests=len(seen),native_files=len(files))))
if __name__=='__main__':main()
