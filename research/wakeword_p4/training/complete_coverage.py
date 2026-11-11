"""Preregistered v40 half-slot replacement; canonical aliases contribute once."""
import hashlib
import json
from collections import Counter
from pathlib import Path
import numpy as np

class CompleteCoverage:
    def __init__(self, root, negative_timeline=False, positive_timeline=False):
        root=Path(root)
        expected={'training.npz':'bb01f3219db10f31e1095af4912c1807ec38b0b3c640c4676186c1b77a620aa1',
          'sources.json':'0f2b008c2525ee9c093580008af1029618823f92d2718622acd513e682cbc981',
          'records.json':'757f1e6df724943c0e0bf68b9fe7de9b3402cbcebb9cf6c87a89e876ee9af090',
          'report.json':'2c2c49dd90a6c3547a78b190de420619ea25d79344d9efd9b8a69c7bc6dc0751'}
        for name,digest in expected.items():
            if hashlib.sha256((root/name).read_bytes()).hexdigest()!=digest:
                raise ValueError('Unpinned complete coverage input: '+name)
        self.sources=json.loads((root/'sources.json').read_text())
        self.records=json.loads((root/'records.json').read_text())
        with np.load(root/'training.npz',allow_pickle=False) as z:
            self.x=z['x'];self.y=z['y']
        assert self.x.shape==(7782,150,40) and self.x.dtype==np.uint16
        assert self.y.shape==(7782,) and self.y.dtype==np.float32
        self.engines=['kokoro','macos-say','openai-tts','piper','qwen']
        self.phrases=['hey nick','hey next','hey mix']
        self.groups={};excluded=Counter();seen=set()
        for i,g in enumerate(self.sources):
            row=g['aliases'][0];digest=g['source_sha256']
            if digest in seen:raise ValueError('Duplicate canonical WAV')
            seen.add(digest)
            task='positive' if row['label']==1 else row['phrase']
            if task not in ['positive',*self.phrases]:
                excluded[task]+=1;continue
            identity=json.dumps(sorted(row['speakers'],key=str),sort_keys=True)
            self.groups.setdefault((row['engine'],task),{}).setdefault(identity,[]).append(i)
            for v in range(3):
                r=self.records[3*i+v]
                assert (r['source_index'],r['variant'],r['source_sha256'])==(i,v,digest)
                assert self.y[3*i+v]==row['label']
        for engine in self.engines:
            for task in ['positive',*self.phrases]:
                if not self.groups.get((engine,task)):raise ValueError('Missing engine/task')
        self.rng=np.random.default_rng(127726)
        self.report=dict(input_sha256=expected,seed=127726,positive_slots=list(range(16)),
          negative_slots=list(range(32,56)),groups={e+'/'+t:{'identities':len(g),'canonical_sources':sum(map(len,g.values()))} for (e,t),g in self.groups.items()},
          excluded_canonical_by_phrase=dict(excluded),feature_environment='Pinned Mac saved uint16 features; measured Linux frontend differences retained',
          policy='Equal phrase for negatives, engine, conditioning identity, canonical WAV, variant; aliases once. Original British draws occur first; remaining British/ambient slots untouched.',
          source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())

        self.negative_timeline = negative_timeline
        self.temporal_rng = np.random.default_rng(137726)
        self.raw_negatives = {}
        if negative_timeline:
            for i,r in enumerate(self.records):
                row=self.sources[r['source_index']]['aliases'][0]
                if row['label'] or row['phrase'] not in self.phrases:
                    continue
                raw=np.load(root/r['raw'],allow_pickle=False)
                assert raw.dtype==np.uint16 and raw.ndim==2 and raw.shape[1]==40 and len(raw)>=150
                assert hashlib.sha256(raw.tobytes()).hexdigest()==r['raw_sha256']
                self.raw_negatives[i]=raw
            self.report['negative_timeline'] = dict(seed=137726,
                policy='Uniform integer end150..len(raw),24 candidate negatives only',
                authenticated_variants=len(self.raw_negatives))

        self.positive_timeline = positive_timeline
        self.positive_rng = np.random.default_rng(147726)
        self.raw_positives = {}
        if positive_timeline:
            if not negative_timeline:
                raise ValueError('Positive timeline requires v41 negative timeline')
            for i,r in enumerate(self.records):
                if not r['label']:
                    continue
                raw=np.load(root/r['raw'],allow_pickle=False)
                assert raw.dtype==np.uint16 and raw.shape==(r['raw_shape'][0],40)
                assert hashlib.sha256(raw.tobytes()).hexdigest()==r['raw_sha256']
                low=r['feature_crop_end'];high=min(350,len(raw))
                assert 150<=low<=high
                assert (high-150)*160<=32000 and (low+1)*160>=32000+r['signal_samples']
                np.testing.assert_array_equal(raw[low-150:low],self.x[i])
                self.raw_positives[i]=raw
            self.report['positive_timeline'] = dict(seed=147726,
                policy='Uniform end crop_end..min(350,len(raw)),16 candidate positives only; conservative original sample ledger',
                authenticated_variants=len(self.raw_positives))

    def draw_indices(self):
        ids=[]
        for j in range(40):
            task='positive' if j<16 else str(self.rng.choice(self.phrases))
            engine=str(self.rng.choice(self.engines));g=self.groups[engine,task]
            identity=str(self.rng.choice(sorted(g)))
            source=int(self.rng.choice(g[identity]));variant=int(self.rng.integers(3))
            ids.append(3*source+variant)
        assert np.array_equal(self.y[ids],np.array([1]*16+[0]*24,dtype=np.float32))
        return ids

    def replace(self, british):
        ids=self.draw_indices();result=british.copy()
        samples=np.clip(self.x[ids].astype(np.float32)/25.6,0,26)
        if self.positive_timeline:
            for j,i in enumerate(ids[:16]):
                raw=self.raw_positives[i]
                end=int(self.positive_rng.integers(self.records[i]['feature_crop_end'],min(350,len(raw))+1))
                samples[j]=np.clip(raw[end-150:end].astype(np.float32)/25.6,0,26)
        if self.negative_timeline:
            for j,i in enumerate(ids[16:],16):
                raw=self.raw_negatives[i]
                end=int(self.temporal_rng.integers(150,len(raw)+1))
                samples[j]=np.clip(raw[end-150:end].astype(np.float32)/25.6,0,26)
        result[:16]=samples[:16];result[32:56]=samples[16:]
        return result
