"""Training-only spectral-envelope mechanism pilot, not a VTLP reproduction."""
import hashlib,json,sys
from pathlib import Path
import numpy as np
R=Path('research/wakeword_p4/results');A=R/'4090-chatterbox-bare-training-admission-v1.json'
P=R/'4090-bare-envelope-warp-pilot-plan-v1.json';O=Path('.local/wakeword/bare-envelope-warp-pilot-v1')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def warp(pcm,alpha):
    from scipy.signal import stft,istft
    from scipy.ndimage import uniform_filter1d
    _,_,z=stft(pcm,fs=16000,nperseg=512,noverlap=384,boundary='zeros',padded=True)
    env=uniform_filter1d(np.log(np.maximum(np.abs(z),1e-8)),size=21,axis=0,mode='nearest')
    bins=np.arange(len(env));mapped=np.minimum(bins/alpha,bins[-1])
    shifted=np.stack([np.interp(mapped,bins,env[:,i]) for i in range(env.shape[1])],axis=1)
    ratio=np.exp(np.clip(shifted-env,-np.log(10),np.log(10)))
    _,out=istft(z*ratio,fs=16000,nperseg=512,noverlap=384,boundary=True)
    out=out[:len(pcm)]
    gain=np.sqrt(np.mean(pcm**2))/max(1e-12,np.sqrt(np.mean(out**2)))
    out*=gain;peak=float(np.max(np.abs(out)));limit=min(1,.999/max(peak,1e-12));out*=limit
    return out.astype(np.float32),dict(rms_gain=float(gain),peak_before_limit=peak,peak_limit_gain=limit,max_ratio=float(ratio.max()),min_ratio=float(ratio.min()))
def main():
    a=json.loads(A.read_text());rows=sorted([dict(r,status=s) for s in ['admitted','quarantined'] for r in a[s]],key=lambda r:r['index'])
    assert len(rows)==48
    files={str(A):sha(A),str(Path(__file__)):sha(__file__)}
    for r in rows:
        assert r['request']['role']=='training-only' and sha(r['wav']['path'])==r['wav']['sha256']
        files[r['wav']['path']]=r['wav']['sha256']
    if sys.argv[1]=='lock':
        with P.open('x') as f:json.dump(dict(files=files,alphas=[.9,1.1],policy='All48 once including25quarantined, diagnostic only. CPU preprocessing2threads, no wake scoring/ASR/newTTS/model training/admission changes. STFT512/hop128,21bin log-envelope mean,frequency-map bin/alpha,ratio clipped[.1,10],RMSmatch then .999peaklimit. Preserve all derived floatPCM/hashes. Not canonical VTLP or phonetic preservation proof.'),f,indent=2)
        return
    assert sys.argv[1]=='run' and json.loads(P.read_text())['files']==files and not O.exists()
    import soundfile as sf
    O.mkdir();records=[];arrays={}
    for r in rows:
        pcm,sr=sf.read(r['wav']['path'],dtype='float32');assert sr==16000 and pcm.ndim==1
        identity,_=warp(pcm,1);error=float(np.max(np.abs(identity-pcm)));assert error<1e-5
        for alpha in [.9,1.1]:
            y,m=warp(pcm,alpha);assert len(y)==len(pcm) and np.isfinite(y).all() and np.max(np.abs(y))<1
            key=f"{r['index']:03d}-{alpha}";arrays[key]=y
            records.append(dict(index=r['index'],status=r['status'],alpha=alpha,samples=len(y),source_sha256=r['wav']['sha256'],derived_float_sha256=hashlib.sha256(y.tobytes()).hexdigest(),identity_max_error=error,rms_difference=float(np.sqrt(np.mean((y-pcm)**2))),**m))
    np.savez_compressed(O/'derived.npz',**arrays)
    (O/'result.json').write_text(json.dumps(dict(plan_sha256=sha(P),records=records,derived_sha256=sha(O/'derived.npz'),limits='Mechanical transformation only; no measured F0/formant or phonetic correctness, no independent voices. All original labels unchanged. Gain/limiter and fullwave changes are disclosed confounds.'),indent=2)+'\n')
    print(json.dumps(dict(completed=len(records),bytes=(O/'derived.npz').stat().st_size,identity_max_error=max(r['identity_max_error'] for r in records),limited=sum(r['peak_limit_gain']<1 for r in records))))
if __name__=='__main__':main()
