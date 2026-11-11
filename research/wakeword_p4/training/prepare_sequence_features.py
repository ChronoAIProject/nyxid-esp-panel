"""Build five-score supervision segments from existing complete training streams."""
import hashlib
import json
from pathlib import Path
import numpy as np
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
base=Path('.local/wakeword/british-training-features-v1')
history=Path('.local/wakeword/v37-full-history-probe-v1')
assert sha(base/'training.npz')=='c0c389fc4a22e1fb9807d99e8a7a98e0e028696d2047dd27dce81f1de8fef096'
assert sha(history/'arrays.npz')=='bfc595c6b2fab777727017bcfc495ffc84cf13907f0fcb2ccb6dfdf008b15ec6'
old=np.load(base/'training.npz',allow_pickle=False);raws=np.load(history/'arrays.npz',allow_pickle=False)
rows=json.loads((base/'report.json').read_text())['records'];xs=[];labels=[];records=[];groups={}
for row in rows:
    if row['pack_index'] is None:continue
    i=row['pack_index'];raw=raws[f'raw_{i}'];e=row['feature_crop_end']
    np.testing.assert_array_equal(raw[e-150:e],old['x'][i])
    valid=all((e+d-150)*160<=32000 and (e+d+1)*160>=32000+row['signal_samples'] for d in (0,12)) and e+12<=len(raw)
    r={**row,'base_index':i,'sequence_index':None,'all_five_whole_signal_valid':bool(valid)}
    if valid:
        crop=raw[e-150:e+12];assert crop.shape==(162,40)
        r['sequence_index']=len(xs);xs.append(crop);labels.append(int(row['target']=='hey_nyx'))
        groups.setdefault(row['target'],{}).setdefault(row['source_group'],set()).add(row['voice'])
    records.append(r)
assert set(groups)=={'hey_nyx','hey_nick','hey_next','hey_mix'}
assert all(set(g)=={'kokoro','alba','cori'} for g in groups.values())
out=Path('.local/wakeword/british-sequence-training-features-v1');out.mkdir(exist_ok=False)
np.savez_compressed(out/'training.npz',x=np.asarray(xs,dtype=np.uint16),y=np.asarray(labels,dtype=np.float32))
r={'role':'training-only-five-score-segments','base_pack_sha256':sha(base/'training.npz'),
    'history_arrays_sha256':sha(history/'arrays.npz'),'pack_sha256':sha(out/'training.npz'),
    'source_sha256':sha(Path(__file__)),'frames':162,'score_count':5,'attempted':len(records),
    'included':len(xs),'excluded':len(records)-len(xs),'final_data_used':False,
    'wake_models_scored_for_inclusion':False,'groups':{t:{g:sorted(v) for g,v in groups[t].items()} for t in groups},
    'records':records,'limits':'Generated signal/RIR completeness, not phonetic truth. Unsupported supplement samples remain available to baseline v37 main BCE. No score-based inclusion.'}
(out/'report.json').write_text(json.dumps(r,indent=2)+'\n')
print(json.dumps({k:v for k,v in r.items() if k!='records'}))
