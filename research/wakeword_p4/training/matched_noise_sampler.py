"""Actual paired-input nuisance intervention; inherited schedule/RNG unchanged."""
import hashlib,io,json
from pathlib import Path
import numpy as np
import soundfile as sf
from scipy.signal import fftconvolve
from features import features
from paired_rendering_sampler import PairedRenderingSampler
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
class MatchedNoiseSampler(PairedRenderingSampler):
    def __init__(self, previous, condition):
        assert condition in ('independent','common')
        self.__dict__.update(previous.__dict__)
        self.condition=condition; self.cache={}
        self.source=Path('.local/wakeword/piper-balanced-training-full-v1')
        pack=Path('.local/wakeword/piper-balanced-training-features-v1')
        assert sha(self.source/'manifest.jsonl')=='e3a69b86294bf91db29777428e785252411710834d22ba92a037d595c963cdc0'
        assert sha(pack/'report.json')=='93733755fec50b41cf71fcc8e2efa9fdfc14956a61e8b50e98e4bd110d5248b3'
        self.sources={(int(r['voice']),r['length_multiplier'],r['target']):r for r in json.loads((pack/'report.json').read_text())['records'] if r['variant']==0}
    def render(self,voice,speed):
        key=voice,speed
        if key in self.cache:return self.cache[key]
        dry=[]
        for phrase in self.phrases:
            r=self.sources[voice,speed,phrase];p=self.source/r['source'];assert sha(p)==r['source_sha256']
            s,rate=sf.read(p,dtype='float64');assert rate==16000 and s.ndim==1 and len(s)==r['source_samples'];dry.append(s)
        size=32000+max(len(s)+960 for s in dry)+16000
        ir=np.zeros(961,np.float64);ir[[0,160,480,960]]=[1,.25,-.12,.06]
        noise={}
        for mode in ['common','independent']:
            for j,phrase in enumerate(self.phrases):
                domain=f'common-noise-v1:{voice}:{speed}' if mode=='common' else f'independent-noise-v1:{voice}:{speed}:{phrase}'
                seed=int(hashlib.sha256(domain.encode()).hexdigest()[:16],16)
                n=np.random.default_rng(seed).normal(0,1,size);n/=np.sqrt(np.mean(n**2));noise[mode,j]=n
        raw={}
        for arm in self.arms:
            signals=[];waves={}
            for j,s in enumerate(dry):
                signal=fftconvolve(s,ir,mode='full') if arm in ('combined','rir') else s
                signal=signal*(.08/np.sqrt(np.mean(signal**2)))
                signals.append(np.pad(signal,(32000,size-32000-len(signal))))
                for mode in ['common','independent']:
                    n=noise[mode,j]*(.08*10**(-(25 if arm=='noise25' else 15)/20)) if arm in ('noise25','combined','noise15') else np.zeros(size)
                    waves[mode,j]=signals[-1]+n
            gain=min(1.,.95/max(np.max(np.abs(w)) for w in waves.values()))
            for j in range(4):
                individual=min(1.,.95/np.max(np.abs(signals[j])))
                w=signals[j]*individual if arm=='quiet' else waves[self.condition,j]*gain
                assert np.isfinite(w).all() and np.max(np.abs(w))<1
                buf=io.BytesIO();sf.write(buf,w,16000,subtype='PCM_16',format='WAV');buf.seek(0);pcm,rate=sf.read(buf,dtype='int16')
                assert np.any(pcm) and not np.any((pcm==-32768)|(pcm==32767))
                raw[j,arm]=features(pcm)
        self.cache[key]=(raw,max((32000+len(s)+960+159)//160-1 for s in dry))
        return self.cache[key]
    def batch(self):
        _,labels,weights=super().batch()
        tr=self.last_trace;end=tr['endpoint_inclusive'];windows=[]
        for voice in tr['voices']:
            raw,low=self.render(voice,tr['speed']);assert end>=low
            for j in range(4):
                for arm in self.arms:
                    assert end<len(raw[j,arm]);windows.append(raw[j,arm][end-149:end+1])
        x=np.clip(np.stack(windows).astype(np.float32)/25.6,0,26)
        assert x.shape==(40,150,40)
        tr['original_window_sha256']=tr.pop('window_sha256')
        tr['window_sha256']=[hashlib.sha256(w.tobytes()).hexdigest() for w in windows]
        tr['network_tensor_sha256']=hashlib.sha256(x.tobytes()).hexdigest()
        tr['noise_condition']=self.condition
        return x,labels,weights
