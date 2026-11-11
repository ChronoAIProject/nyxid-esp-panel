"""Build pinned Kokoro/Alba/Cori training-only complete-waveform feature pack."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.signal import fftconvolve
import soundfile as sf
from features import features


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--alba',type=Path,required=True)
    p.add_argument('--cori',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    expected=[(a.dataset,'kokoro','405dba05d39f2cf9232632380a81cfaec9dc9d0cb8f1fa3227cb3ab696b00dea',400),
              (a.alba,'alba','2bd3c99aadb913df3dd5c6087a7929c4accc5af41d92cf02bd4d63ce1435b738',12),
              (a.cori,'cori','bad9790b00051aabf22260a6bb6b4ee54aaed7d9b3a59c66b9fbb2fb982c490c',12)]
    rows=[]; manifests={}
    for root,group,digest,count in expected:
        manifest=root/'manifest.jsonl';assert sha(manifest)==digest
        part=[json.loads(x) for x in manifest.read_text().splitlines()]
        assert len(part)==count
        for r in part:
            assert r['split'] in ('training','train') and r['quality_pass'] and not r['trimmed']
            assert sha(root/r['path'])==r['sha256']
            r['_root']=root;r['source_group']=group
            r['speed']=r.get('speed',r.get('length_multiplier'))
            r['voice']=group+'/'+r['voice'];rows.append(r)
        manifests[group]=digest
    xs=[];labels=[];records=[];counts=Counter()
    ir=np.zeros(961,dtype=np.float64);ir[[0,160,480,960]]=[1,.25,-.12,.06]
    for row in rows:
        pcm,rate=sf.read(row['_root']/row['path'],dtype='float64');assert rate==16000 and pcm.ndim==1
        for variant in range(3):
            seed=int(hashlib.sha256((row['path']+f':{variant}:phone-v1').encode()).hexdigest()[:16],16)
            rng=np.random.default_rng(seed)
            speech=fftconvolve(pcm,ir,mode='full') if variant==2 else pcm.copy()
            signal=speech*(.08/max(1e-8,float(np.sqrt(np.mean(speech**2)))))
            lead=32000;end_sample=lead+len(signal)
            wave=np.pad(signal,(lead,16000))
            snr=None
            if variant:
                snr=25 if variant==1 else 15
                noise=rng.normal(0,1,len(wave));noise*=.08*10**(-snr/20)/np.sqrt(np.mean(noise**2))
                wave=wave+noise
            peak=float(np.max(np.abs(wave)))
            gain=min(1.,.95/max(peak,1e-8));wave*=gain
            assert np.isfinite(wave).all() and np.max(np.abs(wave))<1
            # Exclusive end e: last used feature e-2 covers through160*(e+1).
            e=(end_sample+159)//160-1
            begin=(e-150)*160;stop=(e+1)*160
            fits=begin<=lead and stop>=end_sample
            item=dict(source=row['path'],source_sha256=row['sha256'],source_group=row['source_group'],voice=row['voice'],speed=row['speed'],
                target=row['target'],split='training',variant=variant,seed=seed,snr_db=snr,
                synthetic_rir=variant==2,rir_tail_samples=960 if variant==2 else 0,
                source_samples=len(pcm),signal_samples=len(signal),gain_after_mix=gain,
                entire_signal_fits=fits,crop_start_sample=begin,crop_stop_sample=stop,
                feature_crop_end=e,pack_index=None)
            if fits:
                raw=features(wave*32768)
                crop=raw[e-150:e]
                assert crop.shape==(150,40)
                item['pack_index']=len(xs);xs.append(crop);labels.append(int(row['target']=='hey_nyx'))
                counts[f"{row['target']}/variant{variant}/included"]+=1
            else: counts[f"{row['target']}/variant{variant}/excluded_context"]+=1
            records.append(item)
    pack=a.output/'training.npz'
    np.savez_compressed(pack,x=np.asarray(xs,dtype=np.uint16),y=np.asarray(labels,dtype=np.float32))
    report=dict(role='training-only-phone-binary-pack',sources=len(rows),attempted_variants=len(records),
        included=len(xs),excluded_context=len(records)-len(xs),counts=dict(counts),
        source_manifest_sha256=manifests,pack_sha256=sha(pack),script_sha256=sha(Path(__file__)),
        frontend_source_sha256=sha(Path(__file__).with_name('features.py')),
        rir=dict(kind='synthetic discrete impulse response',delays_samples=[0,160,480,960],amplitudes=[1,.25,-.12,.06]),
        context_frames=150,model_effective_support_samples=24160,records=records,
        final_data_used=False,wake_models_scored=False,
        limits='Training voices only. Excluded long sources retained in source dataset/report, never truncated or relabeled. Entire signal includes native silence and full RIR tail; this is not phonetic truth. Fixed synthetic room is augmentation, not measured-room generalization.')
    (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='records'}),flush=True)


if __name__=='__main__': main()
