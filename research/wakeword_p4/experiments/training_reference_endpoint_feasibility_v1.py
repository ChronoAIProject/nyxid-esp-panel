"""Enumerate all main training positive generation-reference and full-source endpoints."""
import hashlib
import json
from collections import Counter
from pathlib import Path
import numpy as np
import soundfile as sf

sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
root=Path('.local/wakeword/features-gpu-ami-extra-v1')
index=root/'features.json'
previous=Path('research/wakeword_p4/results/4090-piper-feature-timing-mapping-v1.json')
assert sha(index)==json.loads(previous.read_text())['inputs'][str(index)]
nonpiper=Path('research/wakeword_p4/results/4090-nonpiper-training-timing-receipt-v1.json')
assert json.loads(nonpiper.read_text())['inputs'][str(index)]==sha(index)
renderer=Path('research/wakeword_p4/training/stream_data.py')
assert sha(renderer)=='754c9f2b7644a71643e2da9b7cae936d9791c1d1edb013e1fb1609b2a674d18b'
rirs={};rows=[];summary=Counter();byvariant={};byengine={}
for i,r in enumerate(json.loads(index.read_text())):
    if r['split']!='training' or r['label']!=1 or r.get('engine','piper') not in ['piper','kokoro','qwen-cuda']:continue
    v=r['variant'];tail=0
    if v in [2,3]:
        # The original synthetic renderer allocates6400 taps, delay0.
        tail=6399
    elif v in [4,5]:
        key=r['rir_path'];p=Path('.local/wakeword/mit-rirs')/key
        if key not in rirs:
            assert sha(p)==r['rir_sha256']
            ir,rate=sf.read(p,dtype='float32');assert rate==16000 and ir.ndim==1 and np.isfinite(ir).all()
            onset=np.flatnonzero(np.abs(ir)>=float(np.abs(ir).max())*.1);assert len(onset)
            rirs[key]={'sha256':sha(p),'samples':len(ir),'trimmed_onset':int(onset[0]),'tail_samples':len(ir)-int(onset[0])-1}
        assert rirs[key]['sha256']==r['rir_sha256'];tail=rirs[key]['tail_samples']
    else:assert v in [0,1,6,7,8,9]
    lead=r['lead_frame']*160
    def endpoints(samples):
        lo=max(150,(lead+159+samples+tail+159)//160-1)
        hi=min(r['frames'],r['lead_frame']+150)
        # Full finite RIR extent, even when renderer truncates output; no
        # silent shortening to make a source fit this conservative ledger.
        return {'first':lo if lo<=hi else None,'last':hi if lo<=hi else None,'count':max(0,hi-lo+1)}
    reference=r['keyword_end'] if r.get('engine','piper')=='piper' else r['samples']
    target=endpoints(reference);full=endpoints(r['samples'])
    assert full['count']<=target['count']
    flags={'target_reference_supported':target['count']>0,'full_source_supported':full['count']>0,'no_legal_endpoint':target['count']==0}
    summary.update(k for k,b in flags.items() if b)
    c=byvariant.setdefault(str(v),Counter());c.update(k for k,b in flags.items() if b)
    engine=r.get('engine','piper');c=byengine.setdefault(engine,Counter());c['rows']+=1;c.update(k for k,b in flags.items() if b)
    original=[min(r['frames'],max(150,r['keyword_frame']+j)) for j in range(3,28)]
    supported=sum(target['count']>0 and target['first']<=e<=target['last'] for e in original)
    c['original_candidate_draws']+=25;c['original_supported_draws']+=supported
    rows.append({'feature_index':i,'engine':engine,'original_supported_draws':supported,'source_sha256':r['sha256'],'variant':v,'tail_samples':tail,**flags,'target_endpoints':target,'full_source_endpoints':full,'unsupported_fallback':target['count']==0})
assert len(rows)==63300
out={'scope':'All63300 main Piper/Kokoro/Qwen positive training rows; conservative150frame sample-container ledger with lead uncertainty159samples and full finite RIR extent.',
     'inputs':{str(p):sha(p) for p in [index,previous,nonpiper,renderer]},'rirs':rirs,'summary':dict(summary),'by_variant':{k:dict(v) for k,v in byvariant.items()},'by_engine':{k:dict(v) for k,v in byengine.items()},'rows':rows,
     'predicate':'[e-150:e], start=(e-150)*160<=floor(lead/160)*160; end=(e+1)*160>=floor(lead/160)*160+159+reference_samples+tail_samples;150<=e<=frames.',
     'limits':'Generation reference not phonetic or physical receptive-field truth. Finite extent is conservative and includes quiet RIR tail; no energy-tail cutoff. Does not prove actual tail audibility or label error. Full-source and target-reference predicates separate. Feature binary and synthetic random IR bytes not independently reconstructed here; renderer bounds only. Unsupported fallback retains original draw; no training, newdev/final or promotion.'}
Path('research/wakeword_p4/results/4090-training-reference-endpoint-feasibility-v1.json').write_text(json.dumps(out,separators=(',',':'))+'\n');print(json.dumps(out['summary']));print(json.dumps(out['by_engine']))
