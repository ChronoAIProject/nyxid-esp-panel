"""Saved raw-window origin relative to deployment MFCC hop/cadence."""
import collections,gzip,hashlib,json
from pathlib import Path
R=Path('research/wakeword_p4/results');B=Path('.local/wakeword')
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
 material=B/'matchbox-long-material-v1/records.json';rows=json.loads(material.read_text());binding=json.loads((R/'4090-matchbox-long-training-material-v1.json').read_text());assert sha(material)==binding['files'][str(material)]
 batch=B/'matchbox-long-pooled-recovery-v1/batches.jsonl.gz';phase=collections.defaultdict(collections.Counter);cadence=collections.defaultdict(collections.Counter);offsetsets=collections.Counter();steps=[]
 with gzip.open(batch,'rt') as f:
  for line in f:
   r=json.loads(line);steps.append(r['step'])
   for d in r['descriptors']:
    if d['kind']!='full-bare':continue
    row=rows[d['index']];e=row['engine'];end=row['source_lead']+row['complete_source_and_rir_samples'];assert d['start']+44800==end+d['after']
    phase[e][d['start']%160]+=1;cadence[e][(d['start']+44800)%480]+=1
 assert steps==list(range(1,5001))
 for row in rows:
  if row['kind']!='synthetic-full-bare':continue
  end=row['source_lead']+row['complete_source_and_rir_samples'];first=((end+256+479)//480)*480;ends=list(range(first,end+1024+1,480));assert ends
  assert all(x-44800>=0 and x-44800<=row['source_lead'] and x>=end for x in ends)
  offsetsets[(row['engine'],len(ends))]+=1
 files=[Path(__file__),material,batch,Path('research/wakeword_p4/experiments/finite_pcm_long_window_v1.py'),Path('research/wakeword_p4/training/matchboxnet_long_window_v1.py'),Path('research/wakeword_p4/training/matchboxnet_pretrained_v1.py'),Path('research/wakeword_p4/experiments/v38_temporal_support.md'),Path('research/wakeword_p4/experiments/v39_five_score_loss.md')]
 result=dict(schema='matchbox-mfcc-phase-audit-v1',files={str(f):sha(f) for f in files},steps=5000,training_start_phase_mod160={e:dict(sorted(c.items())) for e,c in phase.items()},training_end_phase_mod480={e:dict(sorted(c.items())) for e,c in cadence.items()},training_draws_with_deployment_mfcc_phase={e:c[0] for e,c in phase.items()},training_draws_with_deployment_cadence={e:c[0] for e,c in cadence.items()},aligned_window_options_all_material={f'{e}/{n}':v for (e,n),v in offsetsets.items()},all_material_has_full_source_aligned_window_in_original_offset_envelope=True,deployment_start_phase_mod160=0,policy='Relative to each original recording origin,81600prime is multiple480 and44800window is multiple160. Every actual finite context starts at480*k-44800, hence phase0 mod160; exact sampler starts need not. Finish pads retain same grid. This is frontend-frame-origin difference, not convolution stride aliasing (all stride1), phonetic truth, proof of causal mechanism or evidence against free-running device onset distributions. Training-window phase alignment changes within-envelope end offsets and cannot isolate phase from endpoint timing by itself. No neural calls/new data/qualification.')
 with (R/'4090-matchbox-mfcc-phase-audit-v1.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
 print(json.dumps({k:result[k] for k in ['training_draws_with_deployment_mfcc_phase','training_draws_with_deployment_cadence','aligned_window_options_all_material']}))
if __name__=='__main__':main()
