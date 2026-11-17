"""All96 training-only sources; descriptive autocorrelation, never phone truth."""
import hashlib,json,sys
from pathlib import Path
import numpy as np
from pilot_bare_envelope_warp_v1 import warp
R=Path('research/wakeword_p4/results');P=R/'4090-envelope-harmonics-plan-v1.json';O=Path('.local/wakeword/envelope-harmonics-v1')
A=R/'4090-chatterbox-bare-training-admission-v1.json';B=R/'4090-chatterbox-training-admission-v1.json'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def cohort():
    rows=[]
    for owner,label in [(A,1),(B,0)]:
        data=json.loads(owner.read_text())
        for status in ['admitted','quarantined']:
            for r in data[status]:
                if r['request']['requested_label']!=label:continue
                assert r['request']['role']=='training-only'
                rows.append(dict(r,status=status,label=label))
    assert len(rows)==96 and sum(r['label'] for r in rows)==48
    return sorted(rows,key=lambda r:(r['label'],r['index']))
def estimate(x):
    from scipy.signal import find_peaks
    frames=np.lib.stride_tricks.sliding_window_view(x,1024)[::160].astype(np.float64)
    rms=np.sqrt(np.mean(frames**2,axis=1));frames-=frames.mean(axis=1,keepdims=True)
    frames*=np.hanning(1024)
    spectrum=np.fft.rfft(frames,n=2048,axis=1)
    corr=np.fft.irfft(spectrum*spectrum.conj(),n=2048,axis=1)[:,:1024]
    energy=np.pad(np.cumsum(frames**2,axis=1),((0,0),(1,0)))
    lags=np.arange(40,201)
    denominator=np.sqrt(np.maximum(energy[:,1024-lags]*(energy[:,-1,None]-energy[:,lags]),1e-30))
    normalized=corr[:,lags]/denominator
    f0=np.zeros(len(frames));confidence=np.zeros(len(frames))
    for i,values in enumerate(normalized):
        peaks,_=find_peaks(values)
        if not len(peaks):continue
        k=peaks[np.argmax(values[peaks])];a,b,c=values[k-1:k+2]
        offset=.5*(a-c)/(a-2*b+c) if abs(a-2*b+c)>1e-12 else 0
        f0[i]=16000/(40+k+offset);confidence[i]=values[k]
    voiced=(confidence>=.6)&(rms>=.001)&(f0>0)
    return f0,confidence,voiced
def main():
    rows=cohort();paths=[A,B,Path(__file__),Path(__file__).with_name('pilot_bare_envelope_warp_v1.py')]
    files={str(p):sha(p) for p in paths}
    for r in rows:
        path=Path(r['wav']['path']);assert path.stat().st_size==r['wav']['bytes'] and sha(path)==r['wav']['sha256'];files[str(path)]=r['wav']['sha256']
    if sys.argv[1]=='lock':
        with P.open('x') as f:json.dump(dict(files=files,count=96,alphas=[.9,1.1],estimator='1024Hann/160hop, overlap-energy-normalized autocorrelation, local peak maximum in80..400Hz, parabolic lag refinement. Original voiced confidence>=.6 RMS>=.001. Preserve original-frame selection across transforms; no tuning or pass gate.',policy='48bare originals including25quarantined +48originaltraining negatives; same4reference identities, not matched bare lexical contrasts. Descriptive estimator only; no phonetic/F0truth, ASR, wake scoring, training or label change. CPU2threads.'),f,indent=2)
        return
    assert sys.argv[1]=='run' and json.loads(P.read_text())['files']==files and not O.exists()
    import soundfile as sf
    O.mkdir();records=[];arrays={}
    for r in rows:
        x,sr=sf.read(r['wav']['path'],dtype='float32');assert sr==16000
        f,c,v=estimate(x);key=f"{r['label']}-{r['index']:03d}";arrays[key+'-original']=np.stack([f,c,v])
        for alpha in [.9,1.1]:
            y,meta=warp(x,alpha);fy,cy,vy=estimate(y);assert len(fy)==len(f)
            arrays[key+f'-{alpha}']=np.stack([fy,cy,vy])
            comparable=v&(fy>0);cents=1200*np.log2(fy[comparable]/f[comparable]);both=v&vy
            row=dict(index=r['index'],label=r['label'],status=r['status'],voice=r['request']['voice'],reference_sha256=r['request']['reference_sha256'],text=r['request']['text'],alpha=alpha,original_voiced_frames=int(v.sum()),transformed_voiced_at_original=int(both.sum()),comparable=int(comparable.sum()),absolute_cents=[float(z) for z in np.abs(cents)],over600cents=int(np.sum(np.abs(cents)>600)),source_sha256=r['wav']['sha256'],derived_sha256=hashlib.sha256(y.tobytes()).hexdigest())
            records.append(row)
    assert len(records)==192
    np.savez_compressed(O/'tracks.npz',**arrays)
    summary=[]
    for label in [0,1]:
        for alpha in [.9,1.1]:
            rs=[r for r in records if r['label']==label and r['alpha']==alpha];cents=[z for r in rs for z in r['absolute_cents']]
            summary.append(dict(label=label,alpha=alpha,clips=len(rs),original_voiced=sum(r['original_voiced_frames'] for r in rs),retained_voiced=sum(r['transformed_voiced_at_original'] for r in rs),comparable=sum(r['comparable'] for r in rs),median_absolute_cents=float(np.median(cents)),p95_absolute_cents=float(np.percentile(cents,95)),over600cents=sum(r['over600cents'] for r in rs)))
    report=dict(plan_sha256=sha(P),records=records,summary=summary,tracks_sha256=sha(O/'tracks.npz'),limits='Estimator outputs are not measured F0truth/phonetic preservation. No acceptance gate. Wholewave/RMSchanges remain. Negative sentences share reference identities but are not bare five-phrase contrasts.')
    (O/'result.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(summary),flush=True)
if __name__=='__main__':main()
