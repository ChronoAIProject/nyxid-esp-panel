"""v44 fixed independently seeded training-only US coverage replacement."""
import hashlib,json
from pathlib import Path
import numpy as np
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
class USCoverage:
 def __init__(self,root,temporal=False):
  root=Path(root);pack='4a0d6febb8c0cd72b6bc58d26123d71b3cf3aa06736e013fd6ccae27dc58e6da';report='e73f3c3cb95b2c97a079a5cf4b4fe75d7289f588b7febb42947b37631f355f0c'
  assert sha(root/'training.npz')==pack and sha(root/'report.json')==report
  metadata=json.loads((root/'report.json').read_text());assert metadata['included']==metadata['attempted_variants']==324 and not metadata['wake_models_scored'] and not metadata['final_data_used']
  with np.load(root/'training.npz',allow_pickle=False) as z:self.x=z['x'];self.y=z['y']
  assert self.x.shape==(324,150,40) and self.x.dtype==np.uint16 and self.y.dtype==np.float32
  self.groups={}
  for r in metadata['records']:
   i=r['pack_index'];assert r['split']=='training' and r['entire_signal_fits'] and r['mechanical_pass'] and self.y[i]==int(r['target']=='hey_nyx')
   self.groups.setdefault((r['target'],r['voice']),[]).append(i)
  for phrase in ['hey_nyx','hey_nick','hey_next','hey_mix']:
   for voice in ['lessac','ryan','ljspeech']:assert len(self.groups[phrase,voice])==27
  self.rng=np.random.default_rng(167726)
  self.report=dict(seed=167726,pack_sha256=pack,report_sha256=report,source_sha256=sha(Path(__file__)),positive_slots=list(range(8)),negative_slots=list(range(32,44)),policy='After all v43 draws, equal negative phrase/preset/uniform sorted eligible variant; all other slots/RNG unchanged',groups={p+'/'+v:len(ids) for (p,v),ids in self.groups.items()})
  self.temporal=temporal;self.temporal_rng=np.random.default_rng(177726);self.records=metadata['records'];self.raw={}
  if temporal:
   for r in self.records:
    raw=np.load(root/r['raw'],allow_pickle=False);i=r['pack_index'];low=r['feature_crop_end'];high=min(350,len(raw))
    assert sha(root/r['raw'])==r['raw_file_sha256'] and hashlib.sha256(raw.tobytes()).hexdigest()==r['raw_sha256']
    assert raw.dtype==np.uint16 and list(raw.shape)==r['raw_shape'] and 150<=low<=high
    assert (high-150)*160<=32000 and (low+1)*160>=32000+r['signal_samples']
    np.testing.assert_array_equal(raw[low-150:low],self.x[i]);self.raw[i]=raw
   self.report['temporal']=dict(seed=177726,positive_slots=list(range(8)),complete_negative_slots=list(range(32,38)),timeline_negative_slots=list(range(38,44)),policy='One independent end per chosenUSexample;8positive+6negative uniform complete ends,6negative uniform fulltimeline;source draws unchanged')
 def draw_indices(self):
  result=[]
  for j in range(20):
   phrase='hey_nyx' if j<8 else str(self.rng.choice(['hey_nick','hey_next','hey_mix']))
   voice=str(self.rng.choice(['lessac','ryan','ljspeech']));result.append(int(self.rng.choice(sorted(self.groups[phrase,voice]))))
  np.testing.assert_array_equal(self.y[result],np.array([1]*8+[0]*12,np.float32));return result
 def replace(self,batch):
  ids=self.draw_indices();result=batch.copy();scaled=np.clip(self.x[ids].astype(np.float32)/25.6,0,26)
  if self.temporal:
   for j,i in enumerate(ids):
    raw=self.raw[i];low=self.records[i]['feature_crop_end'] if j<14 else 150;high=min(350,len(raw)) if j<14 else len(raw)
    end=int(self.temporal_rng.integers(low,high+1));scaled[j]=np.clip(raw[end-150:end].astype(np.float32)/25.6,0,26)
  result[:8]=scaled[:8];result[32:44]=scaled[8:];return result
