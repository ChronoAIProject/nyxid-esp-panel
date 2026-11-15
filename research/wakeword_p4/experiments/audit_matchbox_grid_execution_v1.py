"""Post-completion audit of actual descriptors and initialization, no inference."""
import gzip,hashlib,json
from pathlib import Path
import torch
from matchbox_grid_aligned_batch_v1 import Sampler
R=Path('research/wakeword_p4/results');B=Path('.local/wakeword');NEW=B/'matchbox-grid-aligned-v1';OLD=B/'matchbox-long-pooled-recovery-v1'
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
 report=R/'4090-matchbox-grid-aligned-training-original-v1.json';p=json.loads(report.read_text());assert p['steps']==p['optimizer_iterations']==5000 and p['all_training_checks_passed']
 for name in ['initial.pt','final.pt','batches.jsonl.gz']:assert sha(NEW/name)==p['files'][name]['sha256']
 a=torch.load(NEW/'initial.pt',map_location='cpu',weights_only=True);b=torch.load(OLD/'initial.pt',map_location='cpu',weights_only=True);assert a.keys()==b.keys() and all(torch.equal(a[k],b[k]) for k in a)
 sampler=Sampler('pooled');steps=0;changed=0
 with gzip.open(NEW/'batches.jsonl.gz','rt') as nf,gzip.open(OLD/'batches.jsonl.gz','rt') as of:
  for new,old in zip(nf,of,strict=True):
   n=json.loads(new);o=json.loads(old);steps+=1;assert n['step']==o['step']==steps
   assert n['descriptors']==sampler.descriptors();assert n['labels_sha256']==o['labels_sha256']
   for x,y in zip(n['descriptors'],o['descriptors'],strict=True):
    if x['kind']=='full-bare':
     assert {k:v for k,v in x.items() if k not in ['start','after','original_after']}=={k:v for k,v in y.items() if k not in ['start','after']}
     assert x['original_after']==y['after'];assert (x['start']+44800)%480==0;changed+=int(x['start']!=y['start'])
    else:assert x==y
 assert steps==5000
 out=dict(schema='matchbox-grid-execution-audit-v1',files={str(f):sha(f) for f in [Path(__file__),report,NEW/'initial.pt',OLD/'initial.pt',NEW/'batches.jsonl.gz',OLD/'batches.jsonl.gz']},steps=steps,all61_initial_tensors_identical=True,actual_descriptors_equal_locked_sampler=True,all_source_class_pair_order_background_ambient_and_label_hashes_preserved=True,changed_fullphrase_windows=changed,all_fullphrase_ends_on480grid=True,policy='Post-completion actual execution audit, not recognition or qualification. Tensor equality concerns initial checkpoint bytes only, no CPU/GPU neural equality gate.')
 with (R/'4090-matchbox-grid-execution-audit-v1.json').open('x') as f:json.dump(out,f,indent=2);f.write('\n')
 print(json.dumps(out))
if __name__=='__main__':main()
