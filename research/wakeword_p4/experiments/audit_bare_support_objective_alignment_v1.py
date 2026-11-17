"""Describe saved positive-mask and five-call support, no loss/threshold tuning."""
import json,hashlib
from pathlib import Path
from collections import Counter
import numpy as np
R=Path('research/wakeword_p4/results')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    src=Path('.local/wakeword/bare-support-cached-context-v1/result.json');p=json.loads(src.read_text());files={str(src):sha(src),str(Path(__file__)):sha(__file__)};rows=[];counts=Counter()
    for r in p['records']:
        trace=Path(r['trace']['path']);assert sha(trace)==r['trace']['sha256'];files[str(trace)]=sha(trace)
        with np.load(trace) as z:
            codes=z['uint8'].astype(np.int64);mask=z['positive_weights']>0;pos=np.flatnonzero(mask);assert len(pos)>=5
            sums=np.convolve(codes,np.ones(5,dtype=np.int64),'valid');inside=np.convolve(mask.astype(int),np.ones(5,dtype=int),'valid')==5
            m=int(sums[inside].max());whole=int(sums[z['relative_block_return'][4:]>0].max())
            row=dict(index=r['index'],pair=r['pair'],condition=r['condition'],gap_samples=r['gap_samples'],target_hit=bool(r['target_events']),positive_mask_calls=len(pos),positive_mask_codes=codes[mask].tolist(),max_sum5_fully_inside_mask=m,max_target_sum5=whole,any_fixed_crossing_inside_mask=m>=1175,first_target_event_inside_mask=bool(r['target_events']) and bool(mask[r['target_events'][0]['step']-1]),masked_probability_mean=float(codes[mask].mean()/256))
        rows.append(row);counts['presentations']+=1;counts['target_hits']+=row['target_hit'];counts['crossings_fully_inside_positive_mask']+=row['any_fixed_crossing_inside_mask'];counts['hits_without_full_mask_crossing']+=row['target_hit'] and not row['any_fixed_crossing_inside_mask'];counts['first_target_event_outside_mask']+=row['target_hit'] and not row['first_target_event_inside_mask']
    assert len(rows)==92
    loss=Path('research/wakeword_p4/experiments/causal_fullsequence_graph_v1.py');files[str(loss)]=sha(loss)
    result=dict(schema='bare-support-objective-alignment-v1',files=files,counts=dict(counts),records=rows,policy='All92savedcachedtrainingpresentations. Fixed threshold1175/1280, five-step sums wholly inside existing positive-mask calls versus any target event. Scalar mean positiveBCE+20negativeBCE is implementation fact, not proof negativeweight causes misses. No new loss/inference/training/threshold search or acceptance. INT8 mask metrics do not equal original float training loss; no causal or phonetic truth claim.')
    with (R/'4090-bare-support-objective-alignment-v1.json').open('x') as f:f.write(json.dumps(result,indent=2)+'\n')
    print(json.dumps(counts))
if __name__=='__main__':main()
