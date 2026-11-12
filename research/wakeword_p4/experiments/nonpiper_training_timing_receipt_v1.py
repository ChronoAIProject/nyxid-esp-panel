"""Authenticate standalone training sources and distinguish energy proxy from generation end."""
import hashlib
import json
from pathlib import Path
from collections import Counter
import numpy as np
import soundfile as sf

sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
root=Path('.local/wakeword/features-gpu-ami-extra-v1');index=root/'features.json'
rows=[];sources={};counts=Counter();manifests={}
for i,r in enumerate(json.loads(index.read_text())):
    if r['split']!='training' or r['label']!=1 or r.get('engine') not in ['kokoro','qwen-cuda']:continue
    name=r['path'].split('/')[0]
    if name not in manifests:
        p=root/'dataset'/name/'manifest.jsonl'
        manifests[name]={'path':str(p),'sha256':sha(p),'rows':{x['path']:x for x in map(json.loads,p.read_text().splitlines()) if x['split']=='training'}}
    original=manifests[name]['rows'][r['path'].split('/',1)[1]]
    assert original['label']==1 and original['sha256']==r['sha256'] and original['samples']==r['samples'] and original['text']==r['text']
    assert r['text'].strip().lower().rstrip('.!?') in ['hey nicks','hey nix','hey nyx']
    if r['path'] not in sources:
        p=root/'dataset'/r['path'];assert sha(p)==r['sha256']
        pcm,rate=sf.read(p,dtype='float32');assert rate==16000 and pcm.ndim==1 and len(pcm)==r['samples'] and np.isfinite(pcm).all()
        count=len(pcm)//160;rms=np.sqrt(np.mean(pcm[:count*160].reshape(-1,160)**2,axis=1));active=np.flatnonzero(rms>max(1e-6,float(rms.max())*.005));assert len(active)
        proxy=min(len(pcm),(int(active[-1])+1)*160+960)
        sources[r['path']]={'sha256':r['sha256'],'samples':len(pcm),'original_keyword_end':original['keyword_end'],'recomputed_energy_proxy':proxy,'generation_reference_end':len(pcm),'text':r['text'],'engine':r['engine']}
    s=sources[r['path']];assert r['keyword_end']==s['recomputed_energy_proxy'] and r['synthesis_end']==s['samples'] and original['keyword_end']==s['samples']
    counts[r['engine']]+=1
    rows.append({'feature_index':i,'source':r['path'],'engine':r['engine'],'energy_proxy':r['keyword_end'],'generation_reference_end':s['samples'],'variant':r['variant']})
assert len(rows)==3300 and len(sources)==330
out={'scope':'All current Kokoro/Qwen main positive training rows, no evaluation waveform access.',
     'inputs':{str(index):sha(index),**{v['path']:v['sha256'] for v in manifests.values()}},'counts':dict(counts),'unique_sources':len(sources),'sources':sources,'rows':rows,
     'interpretation':'Stored keyword_end is an energy proxy exactly reproduced from original unchanged PCM. Original standalone generation end is samples/synthesis_end. Use full untrimmed generation interval as a conservative synthetic reference, never claim proxy is phone timing.',
     'limits':'330 original training WAV hashes authenticated; original manifests and intended standalone text verified. Neither input text nor generation end establishes human phonetic truth. No source dropping, speed/PCM change, inference, training or newdev/final use.'}
Path('research/wakeword_p4/results/4090-nonpiper-training-timing-receipt-v1.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({'rows':len(rows),'sources':len(sources),'counts':dict(counts)}))
