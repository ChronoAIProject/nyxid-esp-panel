"""Audit fixed channel-mask training provenance against deterministic masks."""
import hashlib,json,sys
from pathlib import Path
import numpy as np,torch
sys.path.insert(0,str(Path('research/wakeword_p4/training').resolve()))
from causal_channel_mask_augmentation_v1 import apply_channel_mask
from causal_prospective_sampler_v1 import Sampler

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 root=Path('.local/wakeword/causal-channel-mask-v1');report=json.loads((root/'result.json').read_text());assert report['steps']==5000
 sampler=Sampler(726);h=hashlib.sha256();count=0
 with (root/'training.jsonl').open() as f:
  for line in f:
   row=json.loads(line);assert row['step']==count+1;desc=sampler.descriptors();assert all({k:a[k] for k in b}==b for a,b in zip(row['descriptors'],desc,strict=True))
   x=torch.zeros(24,64,1,device='cuda');rng=torch.Generator(device='cuda').manual_seed(726+row['step']);_,m=apply_channel_mask(x,rng,.125);mh=hashlib.sha256(m.cpu().numpy().tobytes()).hexdigest();assert mh==row['channel_mask_sha256'];h.update((mh+'\n').encode());count+=1
 assert count==5000
 control=torch.load('.local/wakeword/causal-generator-prospective-contextual-v1/initial.pt',map_location='cpu',weights_only=False);masked=torch.load(root/'initial.pt',map_location='cpu',weights_only=False);assert all(torch.equal(v,masked['model'][k]) for k,v in control['model'].items())
 out=dict(schema='channel-mask-training-audit-v1',steps=count,all_descriptors_reconstructed=True,all_mask_hashes_reconstructed=True,mask_hash_chain=h.hexdigest(),identical_initial_model=True,inputs={str(root/'result.json'):sha(root/'result.json'),str(root/'training.jsonl'):sha(root/'training.jsonl'),__file__:sha(__file__)},limits='Training provenance only; no inference/development/final access.')
 p=Path('research/wakeword_p4/results/4090-channel-mask-training-audit-v1.json');p.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
if __name__=='__main__':main()
