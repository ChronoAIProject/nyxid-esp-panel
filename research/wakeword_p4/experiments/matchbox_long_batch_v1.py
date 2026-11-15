"""Same-window source-balanced two-arm sampler; bounded host PCM cache."""
from collections import OrderedDict
import hashlib,json
from pathlib import Path
import numpy as np
import soundfile as sf
MATERIAL=Path('research/wakeword_p4/results/4090-matchbox-long-training-material-v1.json')
ROWS=Path('.local/wakeword/matchbox-long-material-v1/records.json')
WINDOW=44800
ENGINES=('speecht5','piper','kokoro','qwen')
def digest(a):return hashlib.sha256(a.tobytes()).hexdigest()
class Sampler:
 def __init__(self,arm,seed=726):
  assert arm in ('control','pooled');self.arm=arm;self.rows=json.loads(ROWS.read_text());groups={e:{} for e in ENGINES};self.background=[];self.cache=OrderedDict();self.decoded={}
  for i,r in enumerate(self.rows):
   assert r['role']=='training'
   if r['kind']=='synthetic-full-bare':
    key=(r['conditioning'],r['scale'],r['variant']);g=groups[r['engine']].setdefault(key,{})
    g.setdefault(r['class_id'],[]).append(i)
   else:
    assert r['kind']=='libri-training-background' and r['class_id']==0;self.background.append(i)
  self.groups={e:[g for _,g in sorted(groups[e].items())] for e in ENGINES}
  assert [len(self.groups[e]) for e in ENGINES]==[63,6048,180,72] and len(self.background)==1004
  assert all(set(g)==set(range(1,7)) for gs in self.groups.values() for g in gs)
  self.sources={e:np.random.default_rng(seed+100+k) for k,e in enumerate(ENGINES)}
  self.offsets=np.random.default_rng(seed+200);self.bg_rng=np.random.default_rng(seed+300);self.ambient_rng=np.random.default_rng(seed+400);self.permutation=np.random.default_rng(seed+500)
 def descriptors(self):
  result=[]
  for pair in range(16):
   engine='speecht5' if self.arm=='control' else ENGINES[pair%4];rng=self.sources[engine]
   group=int(rng.integers(len(self.groups[engine])));after=(256,512,1024)[int(self.offsets.integers(3))]
   for label in range(1,7):
    options=self.groups[engine][group][label];index=options[int(rng.integers(len(options)))];r=self.rows[index]
    start=r['source_lead']+r['complete_source_and_rir_samples']+after-WINDOW
    assert 0<=start<=r['source_lead'] and start+WINDOW>=r['source_lead']+r['complete_source_and_rir_samples']
    result.append(dict(kind='full-bare',index=index,start=start,label=label,pair=pair,engine=engine,group=group,after=after))
  for _ in range(24):
   index=self.background[int(self.bg_rng.integers(1004))];start=int(self.bg_rng.integers(max(1,self.rows[index]['samples']-WINDOW+1)))
   result.append(dict(kind='background',index=index,start=start,label=0))
  for _ in range(8):
   group=int(self.ambient_rng.integers(63));label=int(self.ambient_rng.integers(1,7));options=self.groups['speecht5'][group][label];index=options[int(self.ambient_rng.integers(len(options)))];r=self.rows[index]
   start=int(self.ambient_rng.integers(r['source_lead']-WINDOW+1));result.append(dict(kind='source-leading-ambient',index=index,start=start,label=0))
  result=[result[int(i)] for i in self.permutation.permutation(128)]
  assert [sum(d['label']==c for d in result) for c in range(7)]==[32,16,16,16,16,16,16]
  return result
 def pcm(self,index):
  if index in self.cache:
   self.cache.move_to_end(index);return self.cache[index]
  r=self.rows[index];wave,rate=sf.read(r['path'],dtype='int16');assert rate==16000 and wave.ndim==1 and len(wave)==r['samples']
  h=digest(wave.astype('<i2',copy=False))
  if 'pcm_sha256' in r:assert h==r['pcm_sha256']
  if index in self.decoded:assert h==self.decoded[index]['pcm_sha256']
  else:self.decoded[index]=dict(index=index,samples=len(wave),pcm_sha256=h)
  self.cache[index]=wave
  if len(self.cache)>256:self.cache.popitem(last=False)
  return wave
 def batch(self):
  d=self.descriptors();pcm=np.zeros((128,WINDOW),np.int16)
  for i,x in enumerate(d):
   raw=self.pcm(x['index'])[x['start']:x['start']+WINDOW];pcm[i,:len(raw)]=raw
  return pcm,np.array([x['label'] for x in d],np.int64),d
