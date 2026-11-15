"""Original source-balanced draws with metadata-only categorical labels."""
import json
import numpy as np
from causal_source_balanced_batch_v1 import Sampler as Parent,ROOT,digest
from causal_sequence_supervision_v1 import masks
from causal_lexical_competition_v1 import CLASSES
class Sampler(Parent):
    def __init__(self,seed=726):
        super().__init__(seed)
        rows=json.loads((ROOT/'features.json').read_text())
        self.lexical_by_offset={r['offset']:r for r in rows if r['text'] in CLASSES[2:] and r['source_pair_key']['carrier']=='bare'}
    def batch(self):
        (x,p,n),d=super().batch();labels=np.zeros((128,282),np.int32)
        labels[p>0]=1;selected=0
        for i,desc in enumerate(d):
            if desc['kind']!='source_balanced_bare_negative':continue
            row=self.lexical_by_offset[desc['index']]
            assert row['text']==desc['phrase'] and row['split']=='training'
            lead=int((row['lead_frame']+2)*160);end=int(row['generated_container_end_sample'])
            _,pw,_=masks(row['frames'],1,lead,end)
            indices=np.flatnonzero(pw>0);assert len(indices)>0
            labels[i,indices]=CLASSES.index(row['text']);selected+=1
        assert selected==8
        return (x,p,n,labels),d
