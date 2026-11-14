"""Actual primed whole-sentence recognition on fixed training-source diagnostics."""
import argparse,json,os,subprocess,sys
from pathlib import Path
import numpy as np
import soundfile as sf
from scipy.signal import fftconvolve
from collect_multispeaker_priming_v1 import sha,PRIME,prepare,predict
from uint8_event_policy_v1 import event_steps
R=Path('research/wakeword_p4/results');LOCK=R/'4090-causal-sentence-diagnostic-behavior-plan-v1.json';OUT=Path('.local/wakeword/causal-sentence-diagnostic-behavior-v1');NATIVE=Path('.local/wakeword/causal-sentence-diagnostic-native-v1')
def plan():
 original=R/'4090-causal-sentence-diagnostic-native-original-v1.json';n=json.loads(original.read_text());assert not n['failures'] and n['requests']==120 and sha(n['records_path'])==n['records_sha256'];rows=list(map(json.loads,Path(n['records_path']).read_text().splitlines()));models=[]
 for name in ['aligned-prefix','bare-positive']:
  p=R/f'4090-causal-{name}-int8-original-v1.json';m=json.loads(p.read_text())['model'];assert sha(m['path'])==m['sha256'];models.append(dict(name=name,**m))
 files=[Path(__file__),original,Path(n['records_path']),Path(__file__).with_name('collect_multispeaker_priming_v1.py'),Path(__file__).with_name('collect_multispeaker_formal_v1.py'),Path(__file__).with_name('uint8_event_policy_v1.py'),Path('research/wakeword_p4/training/features.py')]
 files += [NATIVE/r['path'] for r in rows]+[Path(m['path']) for m in models]
 p=dict(schema='causal-sentence-diagnostic-behavior-plan-v1',files={str(f):sha(f) for f in files},native_records=rows,models=models,prime_samples=PRIME,lead_samples=28800,tail_samples=16000,threshold=.91796875,variants=[0,1,2],rir_delays=[0,160,480,960],rir_amplitudes=[1.,.25,-.12,.06],snr_db={'1':25,'2':15},expected_new_gib=.15,policy='Both fixed final aligned-prefix/bare-only INT8 on identical120native training-source diagnostic requests ×3fixedquiet/noise/fullRIRconditions. CompletePCM/lead1.8s/tail1s, originalfullRIR/no source trim. FixednativeID-basednoise seed, preserve fullwav/hash/gain. Same81600continuousprime/frontend/model/app-event state. Fixedreference .91796875/smoothing5/67cooldown, no thresholdsearch/fit/newroles. Wholeutterance any-after-source recognition and falseevents only; no fabricated within-command prefix timing. Correlated variants and existing trainingvoices are not independentholdout. Allnativequalitypasses already required, no scorebasedfiltering/substitution.')
 with LOCK.open('x') as f:f.write(json.dumps(p,indent=2)+'\n')
 print(json.dumps(dict(plan_sha256=sha(LOCK),native=120,waveforms=360,models=2)))
def run():
 assert subprocess.check_output(['git','show','origin/research-hye-nyx-voice-gpu:'+str(LOCK)])==LOCK.read_bytes();p=json.loads(LOCK.read_text())
 for f,h in p['files'].items():assert sha(f)==h,f
 subprocess.run(['/usr/bin/python3','research/wakeword_p4/tools/check_disk_budget.py','--expected-gib',str(p['expected_new_gib'])],check=True)
 OUT.mkdir(exist_ok=False);(OUT/'wav').mkdir();launch=dict(pid=os.getpid(),proc_starttime=Path('/proc/self/stat').read_text().split()[21],plan_sha256=sha(LOCK));(OUT/'launch.json').write_text(json.dumps(launch,indent=2)+'\n')
 sys.path.insert(0,str(Path('research/wakeword_p4/training').resolve()));from features import features
 from ai_edge_litert.interpreter import Interpreter,OpResolverType
 def runtime(**kw):return Interpreter(num_threads=1,experimental_op_resolver_type=OpResolverType.BUILTIN_REF,**kw)
 rows=[]
 for native in p['native_records']:
  dry,rate=sf.read(NATIVE/native['path'],dtype='int16');assert rate==16000 and dry.ndim==1 and len(dry)==native['samples'];original=dry;dry=dry.astype(np.float64)/32768
  for variant in p['variants']:
   seed=int(__import__('hashlib').sha256(((native.get('original_native_id') or native['id'])+':nyx-coverage-render-v1:'+str(variant)).encode()).hexdigest()[:16],16);ir=np.zeros(961);ir[p['rir_delays']]=p['rir_amplitudes'];speech=fftconvolve(dry,ir,mode='full') if variant==2 else dry;values=np.pad(speech,(p['lead_samples'],p['tail_samples']))
   if variant:
    noise=np.random.default_rng(seed).normal(0,1,len(values));noise*=np.sqrt(np.mean(dry**2))*10**(-p['snr_db'][str(variant)]/20)/np.sqrt(np.mean(noise**2));values+=noise
   gain=min(1.,.95/max(float(np.max(np.abs(values))),1e-8)) if variant else 1.;values*=gain;wave=OUT/'wav'/f'{native["id"]}-v{variant}.wav';sf.write(wave,values,16000,subtype='PCM_16');pcm,rate=sf.read(wave,dtype='int16');end=p['lead_samples']+len(dry)+(960 if variant==2 else 0)
   assert len(pcm)==end+p['tail_samples'] and rate==16000 and np.isfinite(values).all() and np.min(pcm)>-32768 and np.max(pcm)<32767
   assert np.max(np.abs(pcm.astype(np.float64)/32768-values))<=1/32768+1e-7
   assert variant!=0 or np.array_equal(pcm[p['lead_samples']:p['lead_samples']+len(original)],original)
   raw=features(np.concatenate((np.zeros(PRIME,np.int16),pcm)));inputs,support,clock=prepare(raw,PRIME+len(pcm))
   for model in p['models']:
    scores,codes,q=predict(inputs,model['path'],runtime);total=PRIME+len(pcm);events=[dict(step=s,sample=min(((3*s+2)*160+479)//480*480,total)-PRIME) for s in event_steps(codes,p['threshold'])];prime=[e for e in events if e['sample']<=0];app=[dict(e,whole_container_signed_ms=(e['sample']-end)/16) for e in events if e['sample']>0];trace=OUT/f'{model["name"]}-{native["id"]}-v{variant}.npz';np.savez_compressed(trace,uint8=codes,support_unclamped=support,sample_end=clock)
    rows.append(dict(model=model['name'],native_id=native['id'],voice=native['voice'],prefix=native['prefix'],carrier=native['carrier'],text=native['text'],length_scale=native['length_scale'],variant=variant,label=native['label'],wave_path=str(wave),wave_sha256=sha(wave),native_sha256=native['sha256'],seed=seed,gain=gain,samples=len(pcm),container_end_samples=end,trace_path=str(trace),trace_sha256=sha(trace),quantization=q,prime_events=prime,app_events=app,any_after_source=any(e['sample']>=p['lead_samples'] for e in app),pre_source_events=sum(e['sample']<p['lead_samples'] for e in app)))
 groups=[]
 for key in sorted({(r['model'],r['voice'],r['prefix'],r['carrier']) for r in rows}):
  a=[r for r in rows if (r['model'],r['voice'],r['prefix'],r['carrier'])==key];groups.append(dict(model=key[0],voice=key[1],prefix=key[2],carrier=key[3],clips=len(a),label=a[0]['label'],any_after_source=sum(r['any_after_source'] for r in a),activated_clips=sum(bool(r['app_events']) for r in a),pre_source_events=sum(r['pre_source_events'] for r in a)))
 report=dict(schema='causal-sentence-diagnostic-behavior-original-v1',launch=launch,plan_sha256=sha(LOCK),records=rows,groups=groups,threshold_selected=False,training_run=False,independent_acceptance=False,policy=p['policy'])
 with (R/'4090-causal-sentence-diagnostic-behavior-original-v1.json').open('x') as f:f.write(json.dumps(report,indent=2)+'\n')
 print(json.dumps(dict(completed=True,waveforms=360,streams=len(rows),groups=groups)),flush=True)
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('action',choices=['plan','run']);x=a.parse_args();globals()[x.action]()
