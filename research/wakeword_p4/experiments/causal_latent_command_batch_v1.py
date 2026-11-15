"""Reuse existing broad training commands, keep original lexical negative draws."""
import collections
import json
import numpy as np
from causal_source_balanced_batch_v1 import Sampler as Parent,ROOT,digest
from causal_sequence_supervision_v1 import masks

class Sampler(Parent):
    def __init__(self,seed=726):
        super().__init__(seed)
        self.command_rng=np.random.default_rng(seed+2)
        self.commands=collections.defaultdict(list)
        rows=json.loads((ROOT/'features.json').read_text())
        selected=[r for r in rows if r['label']==1 and r['source_pair_key']['carrier']!='bare']
        assert len(selected)==18684
        for row in selected:
            assert row['split']=='training' and row['target_id']=='hey_nyx' and row['frames']<=846
            self.commands[(row['engine'],row['conditioning_id'])].append(row)
    def batch(self):
        (x,p,n),descriptors=super().batch()
        eligible=np.zeros((128,278),bool);weak=np.zeros(128,bool)
        assert x.shape==(128,846,40) and p.shape==n.shape==(128,282)
        support=(3*np.arange(1,283,dtype=np.int64)+2)*160
        for i,desc in enumerate(descriptors):
            if desc['kind']!='existing_broad_bare_positive':continue
            anchor=self.positive_by_offset[desc['index']]
            pool=self.commands.get((anchor['engine'],anchor['conditioning_id']))
            if not pool:continue  # Qwen has no existing complete command cohort.
            row=pool[int(self.command_rng.integers(len(pool)))];frames=row['frames']
            lead=int((row['lead_frame']+2)*160);end=int(row['generated_container_end_sample'])
            _,_,nw=masks(frames,1,lead,end)
            raw=self.extra_positive_values[row['offset']:row['offset']+frames]
            x[i].fill(0);p[i].fill(0);n[i].fill(0)
            x[i,:frames]=np.clip(raw.astype(np.float32)/np.float32(25.6),0,26)
            n[i,:len(nw)]=nw
            # Five-output windows entirely within available complete features,
            # after source onset and through the existing postcontainer horizon.
            eligible[i]=(support[:-4]>lead)&(support[4:]<=end+27*160)&(np.arange(278)+4<len(nw))
            assert eligible[i].any();weak[i]=True
            descriptors[i]=dict(kind='existing_broad_latent_command',index=row['offset'],source_id=row['source_id'],engine=row['engine'],conditioning_id=row['conditioning_id'],carrier=row['source_pair_key']['carrier'],variant=row['variant'],label=1,lead_samples=lead,container_end_samples=end,eligible_windows=int(eligible[i].sum()),original_bare_offset=anchor['offset'])
        return (x,p,n,eligible,weak),descriptors
