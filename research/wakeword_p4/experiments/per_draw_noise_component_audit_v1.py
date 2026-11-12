"""Independent saved-component reconstruction audit; no model or feature inference."""
import hashlib, io, json
from pathlib import Path
import numpy as np
import soundfile as sf
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
base=Path('.local/wakeword'); root=base/'per-draw-noise-rendering-preflight-v1'; original=base/'common-noise-rendering-preflight-v1'
r=json.loads((root/'report.json').read_text()); old=json.loads((original/'report.json').read_text())
assert sha(original/'report.json')==r['source_report_sha256']
assert len(r['groups_receipts'])==100
count=0; gains_changed=0
for receipt,prior in zip(r['groups_receipts'],old['groups_receipts']):
    p=root/receipt['arrays']; op=original/prior['arrays']
    assert sha(p)==receipt['arrays_sha256'] and sha(op)==receipt['source_arrays_sha256']==prior['arrays_sha256']
    with np.load(p,allow_pickle=False) as a, np.load(op,allow_pickle=False) as o:
        np.testing.assert_array_equal(a['rir'],o['rir'])
        size=receipt['buffer_samples']; noises={}
        for mode in ['fixed','per_draw']:
            noises[mode]=[]
            for phrase,seed in zip(['hey_nyx','hey_nick','hey_next','hey_mix'],receipt['noise_seeds'][mode]):
                domain=(f'independent-noise-v1:{receipt["voice"]}:{receipt["speed"]}:{phrase}' if mode=='fixed'
                        else f'per-draw-noise-v1:{receipt["block"]}:{receipt["voice"]}:{receipt["speed"]}:{phrase}')
                expected=int(hashlib.sha256(domain.encode()).hexdigest()[:16],16)
                assert seed==expected
                n=np.random.default_rng(seed).normal(0,1,size);n/=np.sqrt(np.mean(n**2));noises[mode].append(n)
        for arm in ['quiet','noise25','combined','noise15','rir']:
            noisy=arm in ['noise25','combined','noise15']; peak=0
            for j in range(4):
                s=a[f'signal_{j}_{arm}'];np.testing.assert_array_equal(s,o[f'signal_{j}_{arm}'])
                for mode in ['fixed','per_draw']:
                    n=a[f'noise_{mode}_{j}_{arm}']
                    expected=(noises[mode][j]*(.08*10**(-(25 if arm=='noise25' else 15)/20))) if noisy else np.zeros(size)
                    np.testing.assert_array_equal(n,expected)
                    peak=max(peak,float(np.max(np.abs(s+n))))
            gain=min(1.,.95/peak);assert gain==receipt['matched_headroom'][arm]
            if arm!='quiet' and gain!=prior['group_headroom'][arm]:gains_changed+=1
            for j in range(4):
                for mode in ['fixed','per_draw']:
                    w=o[f'wave_{j}_{arm}'] if arm=='quiet' else (a[f'signal_{j}_{arm}']+a[f'noise_{mode}_{j}_{arm}'])*gain
                    buf=io.BytesIO();sf.write(buf,w,16000,subtype='PCM_16',format='WAV');buf.seek(0)
                    pcm,rate=sf.read(buf,dtype='int16');np.testing.assert_array_equal(pcm,a[f'pcm_{mode}_{j}_{arm}'])
                    assert rate==16000 and len(pcm)==size and np.any(pcm) and np.max(np.abs(w))<=.950000000000001
                    assert not np.any((pcm==-32768)|(pcm==32767));count+=1
    if receipt['batch']%10==0:print(json.dumps({'groups_audited':count//40}),flush=True)
for mode,item in r['conditions'].items():
    p=Path(item['path']); assert sha(p)==item['sha256']
    with np.load(p,allow_pickle=False) as a:assert a['x'].shape==(50,40,150,40) and np.isfinite(a['x']).all()
result={'script_sha256':sha(Path(__file__)),'matched_report_sha256':sha(root/'report.json'),'groups':100,'pcm_arrays_exactly_reconstructed':count,'all_saved_signals_equal_authenticated_source':True,'all_fixed_and_per_draw_seeds_reproduced':True,'all_noise_components_exact':True,'rir_exact':True,'joint_headroom_exact':True,'nonquiet_group_gains_changed_from_original_common_render':gains_changed,'windows_file_hashes_verified':True,'optimizer_updates':0,'limits':'Saved component and PCM construction audit only. Feature regeneration, network input integration, gradients and quality acceptance are separate. No newdev/final data.'}
assert not Path('research/wakeword_p4/results/4090-per-draw-noise-component-audit-v1.json').exists()
Path('research/wakeword_p4/results/4090-per-draw-noise-component-audit-v1.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
