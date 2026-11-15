"""Read existing INT8 traces only; no inference, selection or policy changes."""
import hashlib,json,gzip,collections
from pathlib import Path
import numpy as np
B=Path('.local/wakeword/matchbox-long-training-stream-v1');R=Path('research/wakeword_p4/results')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 report=json.loads((R/'4090-matchbox-long-training-stream-original-v1.json').read_text());plan=R/'4090-matchbox-long-training-stream-plan-v1.json';p=json.loads(plan.read_text());assert sha(plan)==report['plan_sha256']
 results=[]
 for arm in ['control','pooled']:
  log=B/f'{arm}.jsonl';assert sha(log)==report['summary'][arm]['records_sha256']
  for line in log.read_text().splitlines():
   r=json.loads(line);assert sha(r['trace'])==r['trace_sha256'];row=r['record'];a=np.load(r['trace']);raw=a['uint8'].astype(np.int64);ends=a['actual_sample_end']-p['prime_samples'];smooth=np.convolve(raw,np.ones(5,dtype=np.int64),'valid');se=ends[4:]
   source_end=row['source_lead']+row['complete_source_and_rir_samples'];post=ends-source_end;spost=se-source_end
   # Exact full-source endpoint offsets admitted in sampler are256,512,1024;
   # cadence rarely lands exactly there. Envelope is descriptive, not equivalence.
   envelope=(post>=256)&(post<=1024);senvelope=(spost>=256)&(spost<=1024);real=ends>=row['source_lead'];sreal=se>=row['source_lead']
   threshold_sum=p['threshold']*1280
   runs=[];run=0
   for v,valid in zip(raw,real):
    run=run+1 if valid and v/256>=p['threshold'] else 0;runs.append(run)
   item=dict(arm=arm,index=r['index'],engine=row['engine'],conditioning=row['conditioning'],class_id=row['class_id'],path=row['path'],trace_sha256=r['trace_sha256'],source_end_sample=source_end,activated=any(e['sample']>=row['source_lead'] for e in r['events']),raw_peak=int(raw[real].max()),raw_peak_post_source_samples=int(post[np.flatnonzero(real)[np.argmax(raw[real])]]),smooth_peak_sum=int(smooth[sreal].max()),smooth_peak_post_source_samples=int(spost[np.flatnonzero(sreal)[np.argmax(smooth[sreal])]]),longest_raw_above_threshold_calls=max(runs),trained_offset_envelope_calls=int(envelope.sum()),trained_offset_envelope_raw_peak=int(raw[envelope].max()) if envelope.any() else None,trained_offset_envelope_smooth_peak_sum=int(smooth[senvelope].max()) if senvelope.any() else None,any_smooth_crossing=bool(np.any(smooth[sreal]>=threshold_sum)))
   assert item['activated']==item['any_smooth_crossing'],'Unexpected event suppression; inspect history'
   results.append(item)
 material=Path('.local/wakeword/matchbox-long-material-v1/records.json');assert sha(material)==p['files'][str(material)]
 rows=json.loads(material.read_text());indices={r['path']:i for i,r in enumerate(rows)}
 batch=Path('.local/wakeword/matchbox-long-pooled-recovery-v1/batches.jsonl.gz');exposure=collections.Counter();steps=[]
 with gzip.open(batch,'rt') as f:
  for line in f:
   record=json.loads(line);steps.append(record['step'])
   for d in record['descriptors']:
    if d['kind']=='full-bare':exposure[d['index']]+=1
 assert steps==list(range(1,5001))
 for item in results:
  if item['arm']=='pooled':item['exact_record_training_draws']=exposure[indices[item['path']]]
 output=dict(pooled_batch_log_sha256=sha(batch),schema='matchbox-long-training-trace-analysis-v1',source_report_sha256=sha(R/'4090-matchbox-long-training-stream-original-v1.json'),script_sha256=sha(__file__),threshold=p['threshold'],threshold_five_code_sum=p['threshold']*1280,records=results,policy='Existing consumed training traces only. No inference/training/new data or thresholds. Source end is full generated source plus RIR container, not phonetic endpoint. Offset envelope256..1024 is descriptive; it is not proof that cadence windows equal trained offsets. Rawpeak and five-code sums do not establish FP32/INT8 distortion or causal mechanism.')
 with (R/'4090-matchbox-long-training-trace-analysis-v1.json').open('x') as f:json.dump(output,f,indent=2);f.write('\n')
 for x in results:
  if x['class_id']==1 and not x['activated']:print(json.dumps(x))
if __name__=='__main__':main()
