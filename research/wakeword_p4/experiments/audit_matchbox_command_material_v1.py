"""Metadata-only existing training command support, no audio/inference."""
import collections,hashlib,json
from pathlib import Path
import numpy as np
R=Path('research/wakeword_p4/results');B=Path('.local/wakeword')
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
 features=B/'hey-broad-training-features-v1/hey_nyx/features.json';rendered=B/'hey-broad-training-rendered-v1/records.jsonl';binding=json.loads((R/'4090-matchbox-long-training-material-v1.json').read_text())
 for p in [features,rendered]:assert sha(p)==binding['files'][str(p)]
 selected={r['path']:r for r in json.loads(features.read_text()) if r['source_pair_key']['carrier']!='bare'};groups=collections.defaultdict(list);texts=collections.Counter()
 for line in rendered.open():
  row=json.loads(line)
  if row['path'] not in selected:continue
  f=selected[row['path']];assert f['split']=='training' and row['quality_pass'] and all(row['checks'].values());assert row['sha256']==f['sha256'] and row['pcm_sha256']==f['pcm_sha256']
  span=row['generated_container_end_sample']-row['lead_samples'];assert span==row['source']['source_samples']+row['rir_tail_samples'];assert span>0
  end=row['generated_container_end_sample'];first=((end+256+479)//480)*480;gridfits=first<=end+1024 and first-44800<=row['lead_samples']
  groups[(f['engine'],f['source_pair_key']['carrier'],f['text'])].append((span,gridfits));texts[(f['engine'],f['text'])]+=1
 result=[]
 for (engine,carrier,text),values in sorted(groups.items()):
  spans=[v[0] for v in values];result.append(dict(engine=engine,carrier=carrier,text=text,records=len(values),conditioning_ids=len({f['conditioning_id'] for f in selected.values() if f['engine']==engine and f['text']==text}),complete_source_rir_samples_quantiles=dict(zip(['min','median','max'],map(float,np.percentile(spans,[0,50,100])))),fits_original_all_three_offsets=sum(s+1024<=44800 for s in spans),fits_some_original_deployment_grid_endpoint=sum(v[1] for v in values)))
 out=dict(schema='matchbox-command-material-support-v1',files={str(p):sha(p) for p in [Path(__file__),features,rendered,R/'4090-matchbox-long-training-material-v1.json']},records=len(selected),groups=result,metadata_only=True,pcm_hashes_reverified=False,recognition_calls=0,limitations='Existing role/quality metadata binding only, no new audio or phonetic truth. Selected command labels include Nicks and Willow but no matched Nick/Next/Mix/Muse/HiESP command negative carriers in this pack. Cannot treat positive-command availability as balanced lexical task coverage. Full-source+RIR bound, no shortening/cropping/filtering performed. Some commands exceed44800; increasing context or filtering changes source support/cost and needs independent lock. Prior oldencoder command/prefix failures retained. No qualification.')
 with (R/'4090-matchbox-command-material-support-v1.json').open('x') as f:json.dump(out,f,indent=2);f.write('\n')
 print(json.dumps(result,indent=2))
if __name__=='__main__':main()
