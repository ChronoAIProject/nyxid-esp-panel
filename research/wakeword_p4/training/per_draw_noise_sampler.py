"""Paired fixed/per-draw independent noise, preserving original source RNG."""
import hashlib,io
import numpy as np
import soundfile as sf
from scipy.signal import fftconvolve
from features import features
from matched_noise_sampler import MatchedNoiseSampler,sha

class PerDrawNoiseSampler(MatchedNoiseSampler):
    def __init__(self, previous, condition):
        assert condition in ('fixed','per_draw')
        super().__init__(previous, 'independent')
        self.condition=condition

    def render(self,voice,speed):
        key=self.last_trace['block'],voice,speed
        if key in self.cache:return self.cache[key]
        dry=[]
        for phrase in self.phrases:
            r=self.sources[voice,speed,phrase];p=self.source/r['source'];assert sha(p)==r['source_sha256']
            s,rate=sf.read(p,dtype='float64');assert rate==16000 and s.ndim==1 and len(s)==r['source_samples'];dry.append(s)
        size=32000+max(len(s)+960 for s in dry)+16000
        ir=np.zeros(961,np.float64);ir[[0,160,480,960]]=[1,.25,-.12,.06]
        noise={}
        for mode in ['fixed','per_draw']:
            for j,phrase in enumerate(self.phrases):
                domain=(f'independent-noise-v1:{voice}:{speed}:{phrase}' if mode=='fixed'
                        else f'per-draw-noise-v1:{self.last_trace["block"]}:{voice}:{speed}:{phrase}')
                seed=int(hashlib.sha256(domain.encode()).hexdigest()[:16],16)
                n=np.random.default_rng(seed).normal(0,1,size);n/=np.sqrt(np.mean(n**2));noise[mode,j]=n
        raw={}
        for arm in self.arms:
            signals=[];waves={}
            for j,s in enumerate(dry):
                signal=fftconvolve(s,ir,mode='full') if arm in ('combined','rir') else s
                signal=signal*(.08/np.sqrt(np.mean(signal**2)))
                signals.append(np.pad(signal,(32000,size-32000-len(signal))))
                for mode in ['fixed','per_draw']:
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
        # Bound acoustic cache lifetime to one draw: no per-step waveform growth.
        self.cache.clear()
        result=super().batch()
        self.last_trace['draw_noise_condition']=self.condition
        self.last_trace['noise_seed_block']=self.last_trace['block']
        return result
