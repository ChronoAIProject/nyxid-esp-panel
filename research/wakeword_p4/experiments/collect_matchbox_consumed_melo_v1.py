"""One fixed-threshold full-waveform diagnostic; all Melo is now consumed."""
import argparse
import hashlib
import importlib.metadata as metadata
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import numpy as np
from finite_pcm_window_v1 import contexts
from uint8_event_policy_v1 import event_steps
from select_melo_joint_development_v2 import STRATA
R=Path('research/wakeword_p4/results');BASE=Path('.local/wakeword');LOCK=R/'4090-matchbox-consumed-melo-plan-v1.json';OUT=BASE/'matchbox-consumed-melo-v1';PRIME=81600
sys.path.insert(0,str(Path('research/wakeword_p4/training').resolve()))
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def plan():
 export=R/'4090-matchbox-window-int8-original-v1.json';e=json.loads(export.read_text());assert e['all_static_and_runtime_abi_passed'];model=e['model'];assert sha(model['path'])==model['sha256']
 parent=R/'4090-melo-development-int8-execution-v2.json';p=json.loads(parent.read_text());assert len(p['records'])==474
 contract=R/'4090-finite-pcm-window-analytic-check-v1.json';assert json.loads(contract.read_text())['all_passed']
 files={str(Path(row['path'])):row['sha256'] for row in p['records']}
 paths=[Path(__file__),Path(__file__).with_name('finite_pcm_window_v1.py'),Path(__file__).with_name('uint8_event_policy_v1.py'),Path(__file__).with_name('select_melo_joint_development_v2.py'),Path(__file__).with_name('matchbox_window_protocol_v1.md'),Path('research/wakeword_p4/training/matchboxnet_pretrained_v1.py'),Path('research/wakeword_p4/tools/check_disk_budget.py'),BASE/'matchbox-window-v1/frontend.pt',Path(model['path']),export,parent,contract,R/'4090-melo-joint-development-findings-v2.json']
 spec=importlib.util.find_spec('ai_edge_litert');paths.extend(x for x in Path(spec.origin).parent.rglob('*') if x.is_file() and x.suffix in ['.py','.so'])
 import torch
 paths.extend([Path(torch.__file__),Path(torch._C.__file__),Path(torch.__file__).parent/'lib/libtorch_cuda.so'])
 files.update({str(x):sha(x) for x in paths})
 for f,h in files.items():assert sha(f)==h,f
 result=dict(schema='matchbox-consumed-melo-plan-v1',files=files,model=model,records=p['records'],versions={n:metadata.version(n) for n in ['torch','numpy','soundfile','ai-edge-litert','scipy']},threshold=.91796875,prime_samples=PRIME,batch_frontend_windows=64,expected_new_gib=1,
  policy='Consumed diagnostic only, no winner/threshold selection or qualification. All432sameMelo and42wholebackground PCM, same81600continuous zeros; finite past20480PCM window and480cadence, no transition reset/readiness censoring. Fixed0.91796875/fiveUINT8/67refractory/halfthreshold rearm, preserve allprime/finish/events. CUDA float32MFCC is the exact saved-buffer bounded-window training frontend; offline batches contain only each window\'s available pastPCM. Actual exportedINT8 neural bytes execute CPU builtin optimized/defaultdelegate singlethread, one fresh interpreter per recording. No neuralCPU/GPU equality gate or FP32recognition. HostCUDA frontend is not a P4 frontend port/performance proof; same exact algorithm remains to port. Capture source/input/runtime/trace hashes. Never open acceptance, reinterpret consumedMelo as fresh, retry thresholds/calibration or promote; v30/Mac unchanged.')
 with LOCK.open('x') as f:f.write(json.dumps(result,indent=2)+'\n')
 print(json.dumps(dict(recordings=474,plan_sha256=sha(LOCK),threshold=.91796875)),flush=True)
def run():
 assert subprocess.check_output(['git','show','origin/research-hye-nyx-voice-gpu:'+str(LOCK)])==LOCK.read_bytes();p=json.loads(LOCK.read_text())
 for f,h in p['files'].items():assert sha(f)==h,f
 for n,v in p['versions'].items():assert metadata.version(n)==v
 subprocess.run(['/usr/bin/python3','research/wakeword_p4/tools/check_disk_budget.py','--expected-gib','1'],check=True);OUT.mkdir(exist_ok=False)
 launch=dict(pid=os.getpid(),starttime=Path('/proc/self/stat').read_text().split()[21],plan_sha256=sha(LOCK));(OUT/'launch.json').write_text(json.dumps(launch)+'\n')
 try:
  import soundfile as sf
  import torch
  from ai_edge_litert.interpreter import Interpreter,OpResolverType
  from scipy.stats import chi2
  from matchboxnet_pretrained_v1 import LogMFCC,FRONTEND_KEYS
  assert torch.cuda.is_available();torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
  state=torch.load(BASE/'matchbox-window-v1/frontend.pt',map_location='cpu',weights_only=True)
  front=LogMFCC({key:state[name] for key,name in zip(FRONTEND_KEYS,['dct','window','mel'],strict=True)}).cuda().eval()
  metrics=dict(threshold=p['threshold'],positive={s:dict(clips=0,hits=0) for s in STRATA},confusion_clips=0,confusion_activated_clips=0,cross_target_clips=0,cross_target_activated_clips=0,background_events=0,priming_events=0);hours=0.;start=time.monotonic();latencies=[]
  with (OUT/'records.jsonl').open('x') as log:
   for index,row in enumerate(p['records']):
    pcm,rate=sf.read(row['path'],dtype='int16');assert rate==16000 and pcm.ndim==1 and len(pcm)==row['samples'] and hashlib.sha256(pcm.astype('<i2').tobytes()).hexdigest()==row['pcm_sha256']
    combined=np.concatenate((np.zeros(PRIME,np.int16),pcm));windows,nominal,actual,finish=contexts(combined);codes=np.empty(len(windows),np.uint8);features_hash=hashlib.sha256();input_hash=hashlib.sha256()
    it=Interpreter(model_path=p['model']['path'],num_threads=1,experimental_op_resolver_type=OpResolverType.BUILTIN);it.allocate_tensors();a=it.get_input_details()[0];b=it.get_output_details()[0]
    assert a['dtype']==np.int8 and b['dtype']==np.uint8 and list(a['shape'])==[1,128,1,64] and list(b['shape'])==[1,1] and b['quantization']==(1/256,0)
    scale,zero=a['quantization'];assert scale>0
    for offset in range(0,len(windows),p['batch_frontend_windows']):
     raw=np.ascontiguousarray(windows[offset:offset+p['batch_frontend_windows']],dtype=np.float32)/32768
     with torch.no_grad():values=front(torch.from_numpy(raw).cuda()).permute(0,2,1).unsqueeze(2).cpu().numpy()
     values=np.ascontiguousarray(values);assert np.isfinite(values).all();features_hash.update(values.tobytes());q=np.clip(np.rint(values/scale)+zero,-128,127).astype(np.int8);input_hash.update(q.tobytes())
     for j in range(len(q)):
      it.set_tensor(a['index'],q[j:j+1]);it.invoke();codes[offset+j]=it.get_tensor(b['index'])[0,0]
    trace=OUT/f'{index:04}.npz';np.savez_compressed(trace,uint8=codes,nominal_sample_end=nominal,actual_sample_end=actual,finish_call=finish)
    events=[];real=[];prime=0
    for step in event_steps(codes,p['threshold']):
     clock=int(actual[step-1])-PRIME;event=dict(step=step,sample=clock,nominal_sample=int(nominal[step-1])-PRIME,finish_call=bool(finish[step-1]),priming=clock<=0)
     if clock<=0:prime+=1
     else:real.append(clock)
     if row['phase']=='synthetic':
      for boundary in ['generated_prefix_end_sample','full_rir_prefix_end_sample','generated_container_end_sample','full_rir_container_end_sample']:event[boundary+'_signed_ms']=(clock-row[boundary])/16
     events.append(event)
    metrics['priming_events']+=prime
    if row['phase']=='background':metrics['background_events']+=len(real);hours+=row['samples']/16000/3600
    elif row['phrase_id']=='nyx':
     stratum=row['voice_id'].split(':')[-1]+('/bare' if row['carrier']=='bare' else '/command');metrics['positive'][stratum]['clips']+=1;metrics['positive'][stratum]['hits']+=int(any(t>=row['lead_samples'] for t in real))
     if real:latencies.append((real[0]-row['generated_prefix_end_sample'])/16)
    elif row['phrase_id']=='official':metrics['cross_target_clips']+=1;metrics['cross_target_activated_clips']+=int(bool(real))
    else:metrics['confusion_clips']+=1;metrics['confusion_activated_clips']+=int(bool(real))
    receipt=dict(index=index,record=row,model_sha256=p['model']['sha256'],trace_path=str(trace),trace_sha256=sha(trace),calls=len(codes),features_sha256=features_hash.hexdigest(),quantized_inputs_sha256=input_hash.hexdigest(),input_quantization=[scale,zero],events=events,continuous_raw_context=True,transition_reset=False,prime_samples=PRIME)
    log.write(json.dumps(receipt)+'\n');log.flush();print(json.dumps(dict(completed=index+1,total=474,phase=row['phase'],elapsed_seconds=time.monotonic()-start)),flush=True)
  assert {s:v['clips'] for s,v in metrics['positive'].items()}==STRATA and metrics['confusion_clips']==288 and metrics['cross_target_clips']==72
  report=dict(schema='matchbox-consumed-melo-original-v1',plan_sha256=sha(LOCK),launch=launch,model=p['model'],metrics=metrics,background_hours=hours,background_far_upper95=float(chi2.ppf(.95,2*(metrics['background_events']+1))/(2*hours)),first_detected_target_generated_prefix_signed_ms=dict(zip(['min','p25','median','p75','max'],map(float,np.percentile(latencies,[0,25,50,75,100])))) if latencies else None,all474streams_complete=True,elapsed_seconds=time.monotonic()-start,records_sha256=sha(OUT/'records.jsonl'),frontend_device=torch.cuda.get_device_name(),neural_runtime='CPU builtin/defaultdelegate singlethread actualINT8',consumed_diagnostic_only=True,threshold_selected=False,acceptance_accessed=False,promotion_qualified=False,policy=p['policy'])
  with (R/'4090-matchbox-consumed-melo-original-v1.json').open('x') as f:f.write(json.dumps(report,indent=2)+'\n')
  print(json.dumps(report),flush=True)
 except BaseException:
  (OUT/'failure.log').write_text(traceback.format_exc());raise
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('action',choices=['plan','run']);a=parser.parse_args();plan() if a.action=='plan' else run()
