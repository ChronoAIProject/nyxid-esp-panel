"""Audit original saved batches, no recognition or data consumption."""
import collections,gzip,hashlib,json
from pathlib import Path
import numpy as np
R=Path('research/wakeword_p4/results');B=Path('.local/wakeword')
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
 material=B/'matchbox-long-material-v1/records.json';rows=json.loads(material.read_text());binding=json.loads((R/'4090-matchbox-long-training-material-v1.json').read_text());assert sha(material)==binding['files'][str(material)]
 batch=B/'matchbox-long-pooled-recovery-v1/batches.jsonl.gz';counts=collections.Counter();offsets=collections.Counter();steps=[];minmargin=10**9
 with gzip.open(batch,'rt') as f:
  for line in f:
   r=json.loads(line);steps.append(r['step'])
   for d in r['descriptors']:
    if d['kind']!='full-bare':continue
    row=rows[d['index']];assert row['role']=='training' and row['class_id']==d['label'] and row['engine']==d['engine']
    source_end=row['source_lead']+row['complete_source_and_rir_samples'];assert d['start']<=row['source_lead'] and d['start']+44800==source_end+d['after'];assert d['after'] in [256,512,1024]
    minmargin=min(minmargin,row['source_lead']-d['start']);counts[d['index']]+=1;offsets[(row['engine'],d['after'])]+=1
 assert steps==list(range(1,5001));summaries={}
 for engine in ['speecht5','piper','kokoro','qwen']:
  for label in range(1,7):
   ids=[i for i,r in enumerate(rows) if r.get('engine')==engine and r['class_id']==label];draws=[counts[i] for i in ids]
   summaries[f'{engine}/class{label}']=dict(records=len(ids),total_draws=sum(draws),unseen_records=sum(v==0 for v in draws),at_most_three_draws=sum(v<=3 for v in draws),draw_quantiles=dict(zip(['min','p25','median','p75','max'],map(float,np.percentile(draws,[0,25,50,75,100])))))
 out=dict(schema='matchbox-long-sampling-support-audit-v1',script_sha256=sha(__file__),material_sha256=sha(material),batch_log_sha256=sha(batch),steps=5000,synthetic_draws=sum(counts.values()),minimum_full_source_left_margin_samples=minmargin,all_full_sources_and_rir_inside_training_windows=True,offset_counts={f'{e}/{o}':v for (e,o),v in sorted(offsets.items())},summary=summaries,policy='Original training-only batch audit. Exposure counts are exact rendered-record draws, not independent voices, acoustic diversity, native request coverage or learning guarantees. No neural calls, new PCM, threshold changes or heldout access. Three endpoint offsets256/512/1024 give no explicit five-call480sample-cadence trajectory supervision; this is a recipe fact, not proof that such supervision would solve recognition.')
 with (R/'4090-matchbox-long-sampling-support-audit-v1.json').open('x') as f:json.dump(out,f,indent=2);f.write('\n')
 print(json.dumps(summaries,indent=2))
if __name__=='__main__':main()
