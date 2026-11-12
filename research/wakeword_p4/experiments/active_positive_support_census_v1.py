"""Inspect actual surviving positive packs, not overwritten inherited draws."""
import hashlib
import json
from pathlib import Path
import numpy as np

sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
base=Path('.local/wakeword');out={}
for tag,slots,temporal in [('british-training-features-v1',16,False),('us-training-features-v1',8,True),('factorized-training-features-v1',8,True)]:
 root=base/tag;p=root/'report.json';report=json.loads(p.read_text());assert not report['final_data_used']
 assert sha(root/'training.npz')==report['pack_sha256']
 with np.load(root/'training.npz',allow_pickle=False) as z:x=z['x'];y=z['y']
 details=[]
 for r in report['records']:
  i=r.get('pack_index')
  if i is None or r['target']!='hey_nyx':continue
  assert r['split']=='training' and r['entire_signal_fits'] and y[i]==1 and x[i].shape==(150,40)
  end=r['feature_crop_end'];signal=r['signal_samples']
  assert (end-150)*160<=32000 and (end+1)*160>=32000+signal
  ends=list(range(end,min(350,r['raw_shape'][0])+1)) if temporal else [end]
  def five(e):
   return all((k-150)*160<=32000 and (k+1)*160>=32000+signal for k in range(e-12,e+1,3))
  hits=sum(five(e) for e in ends)
  details.append({'pack_index':i,'source':r['source'],'voice':r['voice'],'feature_crop_end':end,'signal_samples':signal,'candidate_endpoints':len(ends),'five_complete_endpoints':hits,'five_complete_at_base_crop':five(end)})
 out[tag]={'surviving_positive_slots':slots,'temporal_sampling':temporal,'pack_sha256':report['pack_sha256'],'report_sha256':sha(p),'positive_variants':len(details),'single_complete_variants':len(details),'base_crop_five_complete_variants':sum(r['five_complete_at_base_crop'] for r in details),'variants_without_five_complete_candidate':sum(r['five_complete_endpoints']==0 for r in details),'candidate_endpoints':sum(r['candidate_endpoints'] for r in details),'five_complete_endpoints':sum(r['five_complete_endpoints'] for r in details),'details':details}
report={'scope':'All actual surviving main positive pack variants; old conservative complete-container ledger only.','packs':out,'limits':'Counts are variant/endpoint censuses, not source-weighted sampler probability, physical receptive fields, phonetic timing or quality. Five-score test is five preceding stride3 contexts at current endpoint. Fixed British crop has no raw timeline in this pack; does not prove a later feasible crop absent elsewhere. No model inference, optimizer, dev/final access or promotion.'}
Path('research/wakeword_p4/results/4090-active-positive-support-census-v1.json').write_text(json.dumps(report,indent=2)+'\n')
for tag,r in out.items():print(tag,json.dumps({k:v for k,v in r.items() if k!='details'}))
