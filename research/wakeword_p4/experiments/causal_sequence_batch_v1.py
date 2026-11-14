"""Fixed whole-sequence sampler; no TensorFlow and no evaluation sources."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
from causal_sequence_supervision_v1 import masks
sys.path.insert(0,str(Path('research/wakeword_p4/training').resolve()))
from ambient_sampling import stratified_probabilities

R=Path('research/wakeword_p4/results')
ROOT=Path('.local/wakeword/causal-onset-material-v1')
AMBIENT=Path('.local/wakeword/multispeaker-coverage-features-v1/broad')
def digest(a):return hashlib.sha256(a.tobytes()).hexdigest()
class Sampler:
    def __init__(self,seed=726):
        self.rng=np.random.default_rng(seed)
        self.plan=json.loads((R/'4090-causal-onset-material-plan-v1.json').read_text())
        self.frames=self.plan['batch_padded_frames'];assert self.frames==750
        self.rows=[json.loads(x) for x in (ROOT/'records.jsonl').read_text().splitlines()]
        assert len(self.rows)==51840 and all(r['source_role']=='training' and r['source']['role']=='training' for r in self.rows)
        self.by_index={r['order']:r for r in self.rows};self.binaries={}
        for path in sorted({r['feature_binary'] for r in self.rows}):
            self.binaries[path]=np.memmap(path,dtype='<u2',mode='r').reshape(-1,40)
        self.pools={}
        for label in [0,1]:
            engines=sorted({r['source']['engine'] for r in self.rows if r['source']['label']==label})
            assert len(engines)==2
            self.pools[label]=[]
            for engine in engines:
                rows=[r for r in self.rows if r['source']['label']==label and r['source']['engine']==engine]
                rows.sort(key=lambda r:(r['source']['text'],r['source']['length_scale'],r['variant'],r['lead_samples'],r['source']['speaker_id'],r['source']['draw'],r['source']['id']))
                self.pools[label].append(rows)
        self.ambient=sum([json.loads((AMBIENT/n).read_text()) for n in ['human.json','ambient.json']],[])
        assert all(r['split']=='training' for r in self.ambient)
        self.ambient_arrays=[np.load(AMBIENT/r['path'],mmap_mode='r',allow_pickle=False) for r in self.ambient]
        assert all(len(a)>=309 and a.shape[1]==40 for a in self.ambient_arrays)
        self.probabilities=stratified_probabilities(self.ambient)
    def batch(self):
        descriptors=[]
        for label,count in [(1,32),(0,48)]:
            for _ in range(count):
                pool=self.pools[label][int(self.rng.integers(2))]
                r=pool[int(self.rng.integers(len(pool)))]
                descriptors.append(dict(kind='synthetic',index=r['order'],label=label))
        for _ in range(48):
            index=int(self.rng.choice(len(self.ambient),p=self.probabilities));length=min(600,len(self.ambient_arrays[index]))
            start=int(self.rng.integers(len(self.ambient_arrays[index])-length+1))
            descriptors.append(dict(kind='ambient',index=index,start=start,frames=length,label=0))
        descriptors=[descriptors[int(i)] for i in self.rng.permutation(128)]
        x=np.zeros((128,self.frames,40),np.float32)
        p=np.zeros((128,self.frames//3),np.float32);n=np.zeros_like(p)
        for i,d in enumerate(descriptors):
            if d['kind']=='synthetic':
                row=self.by_index[d['index']];assert row['source']['label']==d['label']
                frames=row['frames'];offset=row['offset'];raw=self.binaries[row['feature_binary']][offset:offset+frames]
                _,pw,nw=masks(int(frames),d['label'],int(row['lead_samples']),int(row['container_end_samples']))
            else:
                frames=d['frames'];raw=self.ambient_arrays[d['index']][d['start']:d['start']+frames]
                _,pw,nw=masks(int(frames),0,None,None)
            assert raw.shape==(frames,40) and frames<=self.frames
            x[i,:frames]=np.clip(raw.astype(np.float32)/np.float32(25.6),0,26)
            p[i,:len(pw)]=pw;n[i,:len(nw)]=nw
            assert np.isclose(n[i].sum(),1) and np.isclose(p[i].sum(),d['label'])
            assert not np.any((p[i]>0)&(n[i]>0))
        assert sum(d['label'] for d in descriptors)==32 and np.isfinite(x).all()
        return (x,p,n),descriptors
