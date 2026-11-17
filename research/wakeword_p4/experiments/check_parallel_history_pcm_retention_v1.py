"""Production PCM FIFO helpers with actual triple event clocks and full source PCM."""
import os
os.environ.update(OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2')
import ctypes as C,hashlib,json,shutil,subprocess,sys
from pathlib import Path
import numpy as np
from run_parallel_context_streams_v1 import material
R=Path('research/wakeword_p4/results');PLAN=R/'4090-parallel-history-pcm-retention-plan-v1.json';OUT=Path('.local/wakeword/parallel-history-pcm-retention-v1')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def digest(a):return hashlib.sha256(a.astype('<i2',copy=False).tobytes()).hexdigest()
def lock():
 source=R/'4090-parallel-context-streams-plan-v1.json';summary=R/'4090-parallel-history-expert-runtime-original-v1.json';review=R/'4090-parallel-history-expert-runtime-review-v1.json';files=[Path(__file__),Path(__file__).with_name('run_parallel_context_streams_v1.py'),source,summary,review,Path(read(summary)['cohorts']['context']['path']),Path('components/panel_services/panel_voice.c'),Path('components/panel_services/panel_voice_audio.c'),Path('components/panel_services/panel_voice_audio.h')];assert read(review)['all_passed']
 for stream in read(source)['streams']:
  for item in stream['items']:files.append(Path(item['wav']))
 text=Path('components/panel_services/panel_voice.c').read_text();assert '#define WAKE_BUFFER_SAMPLES (16000 * 60)' in text and '#define WAKE_LOOKBACK_SAMPLES (16000 * 2)' in text and 'for (unsigned burst=0;burst<4;++burst)' in text
 out=dict(schema='parallel-history-pcm-retention-plan-v1',files={str(p):sha(p) for p in files},profile=dict(rate=16000,lookback_samples=32000,capture_capacity_samples=960000,capture_block_samples=160,sender_frame_samples=320,max_sender_burst=4,first_connect_delay_samples=48000,probe_after_event_samples=160000),scope='All3whole context streams/all emitted triple events, counterfactual independent handoff per event; all36targets including5misses accounted. Compile unchanged production PCM helper, continuous wholePCM preroll, original wrapped2segment copy, first-connect gate false3seconds then ready,4x320sender bursts per160capture tick. Compare every emitted PCM sample/hash to original contiguous retained suffix; check wholeclip containment and queue peaks. Synthetic scheduling fixture, not measured task timing or actual active-call simulation. Raw corpus PCM substitutes for postAEC/AGC input; no I2S/AGC/Opus/network/P4/phonetic/ASR boundary truth. No model inference, model/policy/buffer resizing, training, waveform disk copies or new exposure.')
 with PLAN.open('x') as f:json.dump(out,f,indent=2);f.write('\n')
class FIFO(C.Structure):_fields_=[('samples',C.POINTER(C.c_int16)),('capacity',C.c_size_t),('head',C.c_size_t),('count',C.c_size_t)]
def run():
 assert subprocess.check_output(['git','show','origin/research-hye-nyx-voice-gpu:'+str(PLAN)])==PLAN.read_bytes();p=read(PLAN)
 for path,h in p['files'].items():assert sha(path)==h,path
 assert shutil.disk_usage('.').free>100*1024**3;OUT.mkdir(exist_ok=False);library=OUT/'pcm.so';cmd=['cc','-std=c11','-O2','-Wall','-Wextra','-Werror','-shared','-fPIC','components/panel_services/panel_voice_audio.c','-lm','-o',str(library)];b=subprocess.run(cmd,capture_output=True);(OUT/'compile.log').write_bytes(b.stdout+b.stderr);assert b.returncode==0,b.stderr
 lib=C.CDLL(str(library.resolve()));ptr=C.POINTER(C.c_int16)
 for name in ['push','pop','preroll_push']:
  f=getattr(lib,'panel_voice_pcm_'+name);f.argtypes=[C.POINTER(FIFO),ptr,C.c_size_t];f.restype=None if name=='preroll_push' else C.c_bool
 lib.panel_voice_pcm_gate.argtypes=[C.POINTER(FIFO),C.POINTER(C.c_bool),C.c_bool,C.c_bool];lib.panel_voice_pcm_gate.restype=C.c_bool
 lib.panel_voice_pcm_clear.argtypes=[C.POINTER(FIFO)];lib.panel_voice_pcm_clear.restype=None
 def address(a,offset=0):return C.cast(a.ctypes.data+2*offset,ptr)
 def fifo(cap):
  a=np.zeros(cap,np.int16);return a,FIFO(address(a),cap,0,0)
 source=read(R/'4090-parallel-context-streams-plan-v1.json');rows=read(read(R/'4090-parallel-history-expert-runtime-original-v1.json')['cohorts']['context']['path'])['records'];reports=[];targets=[]
 for stream,row in zip(source['streams'],rows,strict=True):
  assert stream['name']==row['name'];pcm,timeline=material(stream,8000);assert digest(pcm)==row['pcm_sha256'] and timeline==row['timeline'];whole=np.r_[np.zeros(81600,np.int16),pcm,np.zeros(160000,np.int16)];memory,tail=fifo(32000);cursor=0
  for e in row['emitted']:
   at=e['relative_sample']+81600;assert at>=cursor and at%160==0
   while cursor<at:
    n=min(160,at-cursor);lib.panel_voice_pcm_preroll_push(C.byref(tail),address(whole,cursor),n);cursor+=n
   assert tail.count==32000;oldest=at-tail.count;target=next(t for t in timeline if t['start']<e['relative_sample']<=t['end']);assert target['kind']=='clip' and target['group']=='nyx_connected_command'
   capmem,capture=fifo(960000);first=min(tail.count,tail.capacity-tail.head);assert lib.panel_voice_pcm_push(C.byref(capture),address(memory,tail.head),first);assert lib.panel_voice_pcm_push(C.byref(capture),address(memory),tail.count-first);assert capture.count==32000
   started=C.c_bool(False);output=np.empty(320,np.int16);sent=[];peak=capture.count;end=at+160000;assert end>=target['end']+81600 and end<=len(whole)
   caught_up=None
   for pos in range(at,end,160):
    ready=pos-at>=48000;assert lib.panel_voice_pcm_gate(C.byref(capture),C.byref(started),ready,False);assert lib.panel_voice_pcm_push(C.byref(capture),address(whole,pos),160);peak=max(peak,capture.count)
    if ready:
     for _ in range(4):
      if capture.count<320:break
      assert lib.panel_voice_pcm_pop(C.byref(capture),address(output),320);sent.append(output.copy())
     if capture.count<320 and caught_up is None:caught_up=pos+160-at
   while capture.count>=320:assert lib.panel_voice_pcm_pop(C.byref(capture),address(output),320);sent.append(output.copy())
   assert capture.count==0 and started.value;actual=np.concatenate(sent);expected=whole[oldest:end];assert np.array_equal(actual,expected);assert digest(actual)==digest(expected)
   reports.append(dict(condition=row['name'],event=e,text_index=target['text_index'],oldest_retained_relative_sample=oldest-81600,target_start_relative_sample=target['start'],target_end_relative_sample=target['end'],full_target_clip_retained=oldest-81600<=target['start'],whole_clip_start_margin_samples=target['start']-(oldest-81600),preroll_head=tail.head,preroll_first_segment_samples=first,retained_output_samples=len(actual),output_sha256=digest(actual),reference_sha256=digest(expected),all_samples_exact=True,queue_peak_samples=int(peak),queue_peak_bytes=int(peak*2),caught_up_samples_after_event=caught_up,source_after_end_zero_padding_samples=max(0,end-(81600+len(pcm)))))
  for t in timeline:
   if t['kind']=='clip' and t['group']=='nyx_connected_command':targets.append(dict(condition=row['name'],text_index=t['text_index'],hit=any(t['start']<e['relative_sample']<=t['end'] for e in row['emitted'])))
 assert len(reports)==31 and len(targets)==36 and sum(t['hit'] for t in targets)==31
 result=dict(schema='parallel-history-pcm-retention-original-v1',plan_sha256=sha(PLAN),all_passed=True,compile_actual_exit_code=b.returncode,compile_args=cmd,helper_library_sha256=sha(library),targets=targets,records=reports,summary=dict(targets=36,missed_targets=5,event_handoffs=len(reports),byte_exact_handoffs=sum(x['all_samples_exact'] for x in reports),whole_source_clip_retained=sum(x['full_target_clip_retained'] for x in reports),minimum_whole_clip_start_margin_samples=min(x['whole_clip_start_margin_samples'] for x in reports),max_queue_peak_samples=max(x['queue_peak_samples'] for x in reports),max_queue_peak_bytes=max(x['queue_peak_bytes'] for x in reports)),limits=p['scope'])
 with (R/'4090-parallel-history-pcm-retention-original-v1.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
 print(json.dumps(result['summary']),flush=True)
if __name__=='__main__':{'lock':lock,'run':run}[sys.argv[1]]()
