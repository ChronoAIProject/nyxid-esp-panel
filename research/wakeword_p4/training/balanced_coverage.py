"""v46 balanced training-only replacement of remaining inherited phrase slots."""
import hashlib,json
from pathlib import Path
import numpy as np
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
class BalancedCoverage:
 def __init__(self,root,temporal=True):
  root=Path(root);pack='37ea6ff5ffc8c85973afec3a63f231527e493ddeb6cee77c08d594e96bc15546';report='93733755fec50b41cf71fcc8e2efa9fdfc14956a61e8b50e98e4bd110d5248b3'
  assert sha(root/'training.npz')==pack and sha(root/'report.json')==report
  metadata=json.loads((root/'report.json').read_text());assert metadata['included']==metadata['attempted_variants']==24192 and not metadata['wake_models_scored'] and not metadata['final_data_used']
  with np.load(root/'training.npz',allow_pickle=False) as z:self.x=z['x'];self.y=z['y']
  assert self.x.shape==(24192,150,40) and self.x.dtype==np.uint16 and self.y.dtype==np.float32
  self.groups={}
  for r in metadata['records']:
   i=r['pack_index'];assert r['split']=='training' and r['entire_signal_fits'] and r['mechanical_pass'] and self.y[i]==int(r['target']=='hey_nyx')
   self.groups.setdefault((r['target'],r['voice']),[]).append(i)
  self.voices=sorted({int(r['voice']) for r in metadata['records']});assert self.voices==sorted(np.random.default_rng(726).permutation(904)[:672].tolist())
  for phrase in ['hey_nyx','hey_nick','hey_next','hey_mix']:
   for voice in self.voices:assert len(self.groups[phrase,str(voice)])==9
  self.rng=np.random.default_rng(207726)
  self.report=dict(seed=207726,pack_sha256=pack,report_sha256=report,source_sha256=sha(Path(__file__)),positive_slots=list(range(8,16)),negative_slots=list(range(44,56)),policy='After all v45 draws, equal negative phrase/numeric allowedID/uniform9variants; other slots/RNG unchanged',groups={p+'/'+v:len(ids) for (p,v),ids in self.groups.items()})
  self.temporal=temporal;self.temporal_rng=np.random.default_rng(217726);self.records=metadata['records'];self.raw={}
  if temporal:
   for r in self.records:
    raw=np.load(root/r['raw'],allow_pickle=False);i=r['pack_index'];low=r['feature_crop_end'];high=min(350,len(raw))
    assert sha(root/r['raw'])==r['raw_file_sha256'] and hashlib.sha256(raw.tobytes()).hexdigest()==r['raw_sha256']
    assert raw.dtype==np.uint16 and list(raw.shape)==r['raw_shape'] and 150<=low<=high
    assert (high-150)*160<=32000 and (low+1)*160>=32000+r['signal_samples']
    np.testing.assert_array_equal(raw[low-150:low],self.x[i]);self.raw[i]=raw
   self.report['temporal']=dict(seed=217726,positive_slots=list(range(8,16)),complete_negative_slots=list(range(44,50)),timeline_negative_slots=list(range(50,56)),policy='One independent end per chosenbalancedexample;8positive+6negative uniform complete ends,6negative uniform fulltimeline;source draws unchanged')
 def draw_indices(self):
  result=[]
  for j in range(20):
   phrase='hey_nyx' if j<8 else str(self.rng.choice(['hey_nick','hey_next','hey_mix']))
   voice=str(self.rng.choice(self.voices));result.append(int(self.rng.choice(sorted(self.groups[phrase,voice]))))
  np.testing.assert_array_equal(self.y[result],np.array([1]*8+[0]*12,np.float32));return result
 def replace(self,batch):
  ids=self.draw_indices();result=batch.copy();scaled=np.clip(self.x[ids].astype(np.float32)/25.6,0,26)
  if self.temporal:
   for j,i in enumerate(ids):
    raw=self.raw[i];low=self.records[i]['feature_crop_end'] if j<14 else 150;high=min(350,len(raw)) if j<14 else len(raw)
    end=int(self.temporal_rng.integers(low,high+1));scaled[j]=np.clip(raw[end-150:end].astype(np.float32)/25.6,0,26)
  result[8:16]=scaled[:8];result[44:56]=scaled[8:];return result
