"""Training-only matched six-phrase windows and original Libri negatives."""
import hashlib
import json
from pathlib import Path
import numpy as np
import soundfile as sf

MATERIAL=Path('research/wakeword_p4/results/4090-matchboxnet-training-material-plan-v1.json')
SOURCE=Path('.local/wakeword/speecht5-training-rendered-v1/records.jsonl')
WINDOW=20480
def digest(a):return hashlib.sha256(a.tobytes()).hexdigest()

class Sampler:
    def __init__(self,seed=726,load_pcm=True):
        self.rng=np.random.default_rng(seed);self.plan=json.loads(MATERIAL.read_text());self.rows=self.plan['records']
        original={r['path']:r for r in map(json.loads,SOURCE.read_text().splitlines())}
        groups={};self.background=[];self.cache={};self.decoded=[]
        for index,row in enumerate(self.rows):
            assert row['role']=='training'
            if row['kind']=='synthetic-full-bare':
                r=original[row['path']];assert r['carrier']=='bare' and r['role']=='training'
                key=(r['voice_id'],r['speed_factor'],r['variant'])
                group=groups.setdefault(key,{})
                assert row['class_id'] not in group;group[row['class_id']]=index
            else:
                assert row['kind']=='libri-training-background' and row['class_id']==0
                self.background.append(index)
            if load_pcm:
                pcm,rate=sf.read(row['path'],dtype='int16');assert rate==16000 and pcm.ndim==1 and len(pcm)==row['samples']
                h=digest(pcm.astype('<i2',copy=False))
                if 'pcm_sha256' in row:assert h==row['pcm_sha256']
                self.cache[index]=pcm
                self.decoded.append(dict(index=index,pcm_sha256=h,samples=len(pcm)))
        self.groups=[groups[key] for key in sorted(groups)]
        assert len(self.groups)==63 and all(set(g)==set(range(1,7)) for g in self.groups) and len(self.background)==1004

    def descriptors(self):
        result=[]
        for pair in range(16):
            group=int(self.rng.integers(63));after=[256,512,1024][int(self.rng.integers(3))]
            for label in range(1,7):
                index=self.groups[group][label];row=self.rows[index]
                start=row['source_lead']+row['complete_source_and_rir_samples']+after-WINDOW
                assert 0<=start<=row['source_lead'] and start+WINDOW>=row['source_lead']+row['complete_source_and_rir_samples']
                result.append(dict(kind='full-bare',index=index,start=start,label=label,pair=pair,group=group,after=after))
        for _ in range(24):
            index=self.background[int(self.rng.integers(1004))];samples=self.rows[index]['samples']
            start=int(self.rng.integers(max(1,samples-WINDOW+1)))
            result.append(dict(kind='background',index=index,start=start,label=0))
        for _ in range(8):
            group=int(self.rng.integers(63));label=int(self.rng.integers(1,7));index=self.groups[group][label];row=self.rows[index]
            start=int(self.rng.integers(row['source_lead']-WINDOW+1))
            assert start+WINDOW<=row['source_lead']
            result.append(dict(kind='source-leading-ambient',index=index,start=start,label=0))
        result=[result[int(i)] for i in self.rng.permutation(128)]
        assert [sum(d['label']==i for d in result) for i in range(7)]==[32,16,16,16,16,16,16]
        return result

    def batch(self):
        descriptors=self.descriptors();pcm=np.zeros((128,WINDOW),np.int16)
        for i,d in enumerate(descriptors):
            raw=self.cache[d['index']][d['start']:d['start']+WINDOW]
            pcm[i,:len(raw)]=raw
        labels=np.array([d['label'] for d in descriptors],np.int64)
        return pcm,labels,descriptors
