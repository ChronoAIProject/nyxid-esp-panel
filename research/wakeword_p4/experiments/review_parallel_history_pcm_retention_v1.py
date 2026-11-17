"""Independent PCM/source/event and analytic fixture accounting, no helper rerun."""
import os
os.environ.update(OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2')
import hashlib,json
from pathlib import Path
import numpy as np
import soundfile as sf
R=Path('research/wakeword_p4/results')
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 pp=R/'4090-parallel-history-pcm-retention-plan-v1.json';rp=R/'4090-parallel-history-pcm-retention-original-v1.json';p=read(pp);d=read(rp);assert d['all_passed'] and d['plan_sha256']==sha(pp) and read(R/'4090-parallel-history-pcm-retention-execution-v1.json')['actual_exit_code']==0
 for path,h in p['files'].items():assert sha(path)==h,path
 sources=read(R/'4090-parallel-context-streams-plan-v1.json')['streams'];events=read(read(R/'4090-parallel-history-expert-runtime-original-v1.json')['cohorts']['context']['path'])['records'];verified=0;target_count=0;miss=[]
 for stream,reference in zip(sources,events,strict=True):
  assert stream['name']==reference['name'];parts=[np.zeros(81600,np.int16)];timeline=[];position=0
  for i,item in enumerate(stream['items']):
   pcm,rate=sf.read(item['wav'],dtype='int16');assert rate==16000 and hashlib.sha256(pcm.astype('<i2').tobytes()).hexdigest()==item['pcm_sha256'];timeline.append((position,position+len(pcm),item));parts.append(pcm);position+=len(pcm)
   if i+1<len(stream['items']):parts.append(np.zeros(8000,np.int16));position+=8000
  parts.append(np.zeros(160000,np.int16));whole=np.concatenate(parts);rows=[x for x in d['records'] if x['condition']==reference['name']];assert len(rows)==len(reference['emitted'])
  for event,row in zip(reference['emitted'],rows,strict=True):
   assert event==row['event'];start,end,item=next(t for t in timeline if t[0]<event['relative_sample']<=t[1]);assert item['request']['group']=='nyx_connected_command' and row['text_index']==item['text_index'];oldest=event['relative_sample']-32000;expected=whole[oldest+81600:event['relative_sample']+81600+160000];h=hashlib.sha256(expected.astype('<i2').tobytes()).hexdigest()
   assert row['all_samples_exact'] and row['output_sha256']==row['reference_sha256']==h and row['retained_output_samples']==len(expected)==192000;assert row['oldest_retained_relative_sample']==oldest and row['target_start_relative_sample']==start and row['target_end_relative_sample']==end;assert row['whole_clip_start_margin_samples']==start-oldest>=0 and row['full_target_clip_retained'];assert row['queue_peak_samples']==32000+48000+160==80160 and row['queue_peak_bytes']==160320;assert row['preroll_head']==(event['relative_sample']+81600)%32000 and row['preroll_first_segment_samples']==32000-row['preroll_head'];verified+=1
  for start,end,item in timeline:
   if item['request']['group']=='nyx_connected_command':
    target_count+=1;hit=any(start<e['relative_sample']<=end for e in reference['emitted']);saved=next(x for x in d['targets'] if x['condition']==reference['name'] and x['text_index']==item['text_index']);assert saved['hit']==hit
    if not hit:miss.append(dict(condition=reference['name'],text_index=item['text_index']))
 assert verified==31 and target_count==36 and len(miss)==5;assert d['summary']['minimum_whole_clip_start_margin_samples']==min(x['whole_clip_start_margin_samples'] for x in d['records'])==9347
 out=dict(schema='parallel-history-pcm-retention-review-v1',all_passed=True,report_sha256=sha(rp),script_sha256=sha(__file__),all31_original_output_reference_hashes_rederived=True,all36_targets_accounted=True,misses=miss,minimum_source_clip_start_margin_ms=9347/16,queue_peak_samples=80160,queue_peak_bytes=160320,allocated_preroll_capacity_bytes=64000,allocated_call_capture_capacity_bytes=1920000,limits='Independent source reconstruction/hash and analytic fixture accounting only; output hashes authenticate production-helper execution receipt.31counterfactual independent handoffs, not actual single active-call/multitrigger schedule or P4/AEC/AGC/Opus/network/first-command acoustic timing. No waveform disk copies.')
 with (R/'4090-parallel-history-pcm-retention-review-v1.json').open('x') as f:json.dump(out,f,indent=2);f.write('\n')
 print(json.dumps(out),flush=True)
if __name__=='__main__':main()
