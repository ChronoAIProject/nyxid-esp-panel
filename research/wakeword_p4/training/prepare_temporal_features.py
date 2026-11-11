"""Training-only end-offset crops; source balance must be at base-example level."""
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
report=json.loads((base/'report.json').read_text());xs=[];parents=[];offsets=[];records=[]
for row in report['records']:
    i=row['pack_index']
    if i is None:continue
    raw=raws[f'raw_{i}'];e=row['feature_crop_end']
    np.testing.assert_array_equal(raw[e-150:e],old['x'][i])
    for offset in range(15):
        end=e+offset;begin=(end-150)*160;stop=(end+1)*160
        valid=begin<=32000 and stop>=32000+row['signal_samples'] and end<=len(raw)
        entry={'base_index':i,'offset_frames':offset,'entire_signal_fits':bool(valid),'pack_index':None}
        if valid:
            crop=raw[end-150:end];assert crop.shape==(150,40)
            entry['pack_index']=len(xs);xs.append(crop);parents.append(i);offsets.append(offset)
        records.append(entry)
out=Path('.local/wakeword/british-temporal-training-features-v1');out.mkdir(exist_ok=False)
np.savez_compressed(out/'training.npz',x=np.asarray(xs,dtype=np.uint16),base_index=np.asarray(parents,dtype=np.int32),offset=np.asarray(offsets,dtype=np.int8))
r={'role':'training-only-temporal-offset-pack','base_pack_sha256':sha(base/'training.npz'),
    'history_arrays_sha256':sha(history/'arrays.npz'),'pack_sha256':sha(out/'training.npz'),
    'source_sha256':sha(Path(__file__)),'base_examples':len(old['x']),
    'attempted':len(records),'included':len(xs),'excluded':sum(x['pack_index'] is None for x in records),
    'offset_range_frames':[0,14],'final_data_used':False,'wake_models_scored_for_inclusion':False,
    'records':records,'limits':'Whole generated signal/RIR support, not phonetic truth. No score-based filters. Preserve equal base sampling then uniform valid offsets; never uniform expanded rows.'}
(out/'report.json').write_text(json.dumps(r,indent=2)+'\n')
print(json.dumps({k:v for k,v in r.items() if k!='records'}))
