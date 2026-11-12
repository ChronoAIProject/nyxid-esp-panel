"""Independent fixed-RNG training-only paired five-window sampling."""
import hashlib,json
from pathlib import Path
import numpy as np


class PairedTemporalSampler:
    phrases=('hey_nyx','hey_nick','hey_next','hey_mix')
    arms=('quiet','noise25','combined','noise15','rir')

    def __init__(self,root,authenticated_raw=None):
        self.root=Path(root)
        self.report_sha256=hashlib.sha256((self.root/'report.json').read_bytes()).hexdigest()
        assert self.report_sha256=='7b798653f73e2dcd3b01fd5b7092c947a26b9c8ec2525ff295de2e6478042e44'
        self.records=json.loads((self.root/'report.json').read_text())['records']
        self.lookup={(int(r['voice']),r['length_multiplier'],r['target'],r['arm']):r for r in self.records}
        assert len(self.lookup)==len(self.records)==40320
        self.voices=sorted({int(r['voice']) for r in self.records})
        assert self.voices==sorted(np.random.default_rng(726).permutation(904)[:672].tolist())
        self.raw={} if authenticated_raw is None else authenticated_raw
        self.verified=set();self.rng=np.random.default_rng(237726);self.draws=0;self.last_trace=None

    def get_raw(self,record):
        index=record['pack_index']
        if index not in self.verified:
            if index not in self.raw:
                path=self.root/record['raw']
                assert hashlib.sha256(path.read_bytes()).hexdigest()==record['raw_file_sha256']
                self.raw[index]=np.load(path,allow_pickle=False)
            value=self.raw[index]
            assert value.dtype==np.uint16 and list(value.shape)==record['raw_shape']
            assert hashlib.sha256(value.tobytes()).hexdigest()==record['raw_sha256']
            assert record['split']=='training' and record['entire_signal_fits'] and record['mechanical_pass']
            self.verified.add(index)
        return self.raw[index]

    def batch(self):
        speed=float(self.rng.choice([.9,1.,1.1]))
        voices=[int(v) for v in self.rng.choice(self.voices,size=2,replace=False)]
        rows=[self.lookup[v,speed,p,a] for v in voices for p in self.phrases for a in self.arms]
        low=max(r['feature_crop_end'] for r in rows)+12
        high=min(350,*(r['raw_shape'][0]-1 for r in rows))
        legal=np.arange(((low+2)//3)*3,high+1,3,dtype=np.int64)
        if not len(legal):raise ValueError('Infeasible group; never discard/resample')
        end=int(self.rng.choice(legal));ends=list(range(end-12,end+1,3))
        windows=[]
        for record in rows:
            raw=self.get_raw(record)
            assert all((e-150)*160<=32000 and (e+1)*160>=32000+record['signal_samples'] for e in ends)
            windows.append(np.stack([raw[e-149:e+1] for e in ends]))
        windows=np.stack(windows);assert windows.shape==(40,5,150,40)
        labels=np.asarray([r['target']=='hey_nyx' for r in rows],np.float32)
        assert labels.sum()==10
        self.last_records=rows
        self.last_trace={'block':self.draws,'voices':voices,'speed':speed,'endpoint_inclusive':end,
          'endpoints_inclusive':ends,'pack_indices':[r['pack_index'] for r in rows],
          'window_sha256':[[hashlib.sha256(w.tobytes()).hexdigest() for w in group] for group in windows]}
        self.draws+=1
        return np.clip(windows.astype(np.float32)/25.6,0,26),labels,np.where(labels>0,1.,20.).astype(np.float32)
