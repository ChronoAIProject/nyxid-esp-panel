"""Actual completed batch population census, metadata only."""
import collections,gzip,hashlib,json
from pathlib import Path
R=Path('research/wakeword_p4/results')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
material=Path('.local/wakeword/causal-onset-material-v1/records.jsonl');rows=[json.loads(x) for x in material.read_text().splitlines()];lookup={r['order']:r for r in rows}
probe=R/'4090-multiscale-factorial-control-plan-v1.json';selected=json.loads(probe.read_text())['positives'];ids={r['id'] for r in selected};counts=collections.Counter();positive=collections.Counter();sourcecounts=collections.Counter();variants=collections.Counter();leads=collections.Counter();files={str(material):sha(material),str(probe):sha(probe)}
p=Path('.local/wakeword/binary40-multiscale-training-v1/control/batches.jsonl.gz');files[str(p)]=sha(p)
with gzip.open(p,'rt') as f:
 for step,line in enumerate(f,1):
  b=json.loads(line);assert b['step']==step and len(b['descriptors'])==128
  for d in b['descriptors']:
   counts[d['kind']]+=1
   if d['label']:positive[d['kind']]+=1
   if d['kind']=='synthetic' and d['label']:
    r=lookup[d['index']];sourcecounts[r['source']['id']]+=1
    if r['source']['id'] in ids:variants[str(r['variant'])]+=1;leads[str(r['lead_samples'])]+=1
assert step==20000 and sum(counts.values())==2560000
selected_rows=[r for r in rows if r['source']['id'] in ids]
out={'schema':'multiscale-training-presentation-census-v1','steps':step,'descriptor_rows':sum(counts.values()),'kind_counts':dict(counts),'positive_kind_counts':dict(positive),'selected90_native_sources':{'sources':len(ids),'rendered_material_rows':len(selected_rows),'actual_positive_draws':sum(sourcecounts[k] for k in ids),'draws_per_source':{k:sourcecounts[k] for k in sorted(ids)},'variant_draws':dict(variants),'lead_sample_draws':dict(leads)},'input_sha256':files,'limits':'Actual control ledger population; complete matched batch identities already verified separately. Metadata coverage does not establish acoustic correctness, effective learning, generalization or mechanism. Native90probe is not augmented sampler tensor reproduction. No new inference/training/exposure.'}
(R/'4090-multiscale-training-presentation-census-v1.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:v for k,v in out.items() if k not in ['input_sha256','selected90_native_sources']}));print({k:v for k,v in out['selected90_native_sources'].items() if k!='draws_per_source'})
