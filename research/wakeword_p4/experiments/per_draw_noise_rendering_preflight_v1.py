"""Matched fixed/per-draw noise construction; no network or optimizer calls.

Seeds are fixed before inference. Each group/arm shares headroom across BOTH
conditions and four phrases. Only stochastic nuisance identity differs.
"""
import hashlib, io, json, sys
from pathlib import Path
import numpy as np
import soundfile as sf
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'training'))
from features import features
sha=lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
base=Path('.local/wakeword')
src=base/'common-noise-rendering-preflight-v1'
out=base/'per-draw-noise-rendering-preflight-v1'
assert not out.exists()
report=json.loads((src/'report.json').read_text())
assert report['groups']==100 and report['scheduled_supported_groups']==100
assert sha(src/'windows.npz')==report['windows_sha256']
out.mkdir()
receipts=[]; batches={'fixed':[], 'per_draw':[]}
phrases=['hey_nyx','hey_nick','hey_next','hey_mix']
arms=['quiet','noise25','combined','noise15','rir']
for b in range(50):
    windows={m:[] for m in batches}
    for r in report['groups_receipts'][b*2:b*2+2]:
        p=src/r['arrays']; assert sha(p)==r['arrays_sha256']
        arrays=np.load(p,allow_pickle=False); size=r['buffer_samples']
        noise_by_mode={}; seeds={}
        for mode in batches:
            seeds[mode]=[]; noise_by_mode[mode]=[]
            for phrase in phrases:
                domain=(f'independent-noise-v1:{r["voice"]}:{r["speed"]}:{phrase}' if mode=='fixed'
                        else f'per-draw-noise-v1:{b}:{r["voice"]}:{r["speed"]}:{phrase}')
                seed=int(hashlib.sha256(domain.encode()).hexdigest()[:16],16)
                n=np.random.default_rng(seed).normal(0,1,size)
                n/=np.sqrt(np.mean(n**2));noise_by_mode[mode].append(n);seeds[mode].append(seed)
            assert len(set(seeds[mode]))==4
        stored={'rir':arrays['rir']}; gains={}; raws={}
        for arm in arms:
            scale=.08*10**(-(25 if arm=='noise25' else 15)/20)
            noisy=arm in ['noise25','combined','noise15']
            waves={}
            for j in range(4):
                signal=arrays[f'signal_{j}_{arm}']
                stored[f'signal_{j}_{arm}']=signal
                for mode in batches:
                    n=(noise_by_mode[mode][j]*scale) if noisy else np.zeros(size)
                    stored[f'noise_{mode}_{j}_{arm}']=n
                    waves[mode,j]=signal+n
            gain=min(1., .95/max(np.max(np.abs(w)) for w in waves.values()))
            gains[arm]=gain
            for mode in batches:
                for j in range(4):
                    # Quiet stays byte-identical to the authenticated baseline.
                    w=arrays[f'wave_{j}_{arm}'] if arm=='quiet' else waves[mode,j]*gain
                    buf=io.BytesIO(); sf.write(buf,w,16000,subtype='PCM_16',format='WAV'); buf.seek(0)
                    pcm,rate=sf.read(buf,dtype='int16')
                    assert rate==16000 and len(pcm)==size and np.any(pcm)
                    assert not np.any((pcm==-32768)|(pcm==32767)) and np.isfinite(w).all()
                    if arm=='quiet': np.testing.assert_array_equal(pcm,arrays[f'pcm_{j}_{arm}'])
                    raw=features(pcm); endpoint=r['scheduled_endpoint_inclusive']
                    assert endpoint < len(raw) and endpoint>=149
                    raws[mode,j,arm]=raw
                    stored[f'pcm_{mode}_{j}_{arm}']=pcm
            if not noisy:
                for j in range(4): np.testing.assert_array_equal(raws['fixed',j,arm],raws['per_draw',j,arm])
        for mode in batches:
            for j in range(4):
                for arm in arms:
                    raw=raws[mode,j,arm]; end=r['scheduled_endpoint_inclusive']
                    windows[mode].append(np.clip(raw[end-149:end+1].astype(np.float32)/25.6,0,26))
        path=out/r['arrays']; np.savez_compressed(path,**stored)
        receipts.append({**r,'source_arrays_sha256':r['arrays_sha256'],'arrays_sha256':sha(path),'noise_seeds':seeds, 'block':b,'matched_headroom':gains})
    for mode in batches: batches[mode].append(np.stack(windows[mode]))
    if b%10==0: print(json.dumps({'batch':b}),flush=True)
paths={}
for mode in batches:
    x=np.stack(batches[mode]); assert x.shape==(50,40,150,40) and np.isfinite(x).all()
    path=out/f'{mode}-windows.npz'; np.savez_compressed(path,x=x); paths[mode]={'path':str(path),'sha256':sha(path)}
result={'script_sha256':sha(Path(__file__)),'source_report_sha256':sha(src/'report.json'),'conditions':paths,'batches':50,'groups':100,'windows_per_condition':2000,'source_schedule_equal':True,'headroom_equal_across_conditions':True,'quiet_pcm_equal_to_original':True,'nonnoisy_features_equal_across_conditions':True,'all_scheduled_groups_supported':True,'optimizer_updates':0,'groups_receipts':receipts,'limits':'Construction only, no inference or acceptance. Same complete source signals, buffer, endpoints, RIR and joint headroom across conditions. Fixed and block-specific independent noise domains specified before scoring. Fixed control headroom is recomputed jointly with treatment; not asserted byte-identical to prior independent noisy inputs. Quiet per-source original headroom preserved. Generation boundaries are not phonetic ground truth.'}
(out/'report.json').write_text(json.dumps(result,indent=2)+'\n')
Path('research/wakeword_p4/results/4090-per-draw-noise-rendering-preflight-v1.json').write_bytes((out/'report.json').read_bytes())
print(json.dumps({k:v for k,v in result.items() if k!='groups_receipts'}),flush=True)
