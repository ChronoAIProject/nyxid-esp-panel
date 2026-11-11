"""Append balanced training-only complete-PCM phone examples to binary BCE."""
import hashlib
import json
from pathlib import Path
import numpy as np


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


class PhoneBinary:
    def __init__(self, root, source_balanced=False):
        root=Path(root);report=json.loads((root/'report.json').read_text())
        if (report['role']!='training-only-phone-binary-pack' or report['context_frames']!=150
                or report['final_data_used'] or report['wake_models_scored']
                or sha(root/'training.npz')!=report['pack_sha256']):
            raise ValueError('Require audited training-only phone pack')
        z=np.load(root/'training.npz',allow_pickle=False)
        self.x=z['x'];self.y=z['y'];self.groups={}
        if self.x.dtype!=np.uint16 or self.x.shape!=(report['included'],150,40):
            raise ValueError('Phone pack shape/dtype mismatch')
        included=[r for r in report['records'] if r['pack_index'] is not None]
        if [r['pack_index'] for r in included]!=list(range(len(self.x))):
            raise ValueError('Non-contiguous pack records')
        for r in included:
            i=r['pack_index'];target=r['target']
            if (r['split']!='training' or not r['entire_signal_fits']
                    or self.y[i]!=int(target=='hey_nyx')):
                raise ValueError('Phone pack scope/label mismatch')
            self.groups.setdefault(target,{}).setdefault(r['voice'],[]).append(i)
        if set(self.groups)!={'hey_nyx','hey_nick','hey_next','hey_mix'}:
            raise ValueError('Missing phrase class')
        self.source_balanced=source_balanced
        if source_balanced:
            if report.get('source_manifest_sha256') != {
                'kokoro':'405dba05d39f2cf9232632380a81cfaec9dc9d0cb8f1fa3227cb3ab696b00dea',
                'alba':'2bd3c99aadb913df3dd5c6087a7929c4accc5af41d92cf02bd4d63ce1435b738',
                'cori':'bad9790b00051aabf22260a6bb6b4ee54aaed7d9b3a59c66b9fbb2fb982c490c'}:
                raise ValueError('Require pinned three-source training manifests')
            for target, voices in self.groups.items():
                if {v.split('/')[0] for v in voices}!={'kokoro','alba','cori'}:
                    raise ValueError('Each phrase needs all three sources')
        self.rng=np.random.default_rng(87726)
        self.report=dict(source_balanced=source_balanced,pack_sha256=report['pack_sha256'],report_sha256=sha(root/'report.json'),
            seed=87726,extra_batch_size=16,positive_slots=8,negative_slots=8,loss_weight=.1,
            policy='Original128 BCE + .1*extra16 weighted BCE; extra negatives weight20. Independent RNG; equal voices within phrase, equal negative phrase choice; no auxiliary head; frozen BN. Original batch and representative generator unchanged.',
            groups={k:{v:len(ids) for v,ids in voices.items()} for k,voices in self.groups.items()})

        if source_balanced:
            self.report['policy']+=' Source-balanced variant: equal kokoro/alba/cori checkpoint groups before equal voices; source groups do not certify physical speaker independence.'

    def append(self,x,y,weights):
        if x.shape!=(128,150,40) or y.shape!=(128,1) or weights.shape!=(128,):
            raise ValueError('Require baseline128 batch')
        ids=[]
        for i in range(16):
            target='hey_nyx' if i<8 else str(self.rng.choice(['hey_nick','hey_next','hey_mix']))
            voices=self.groups[target]
            choices=sorted(voices)
            if self.source_balanced:
                group=str(self.rng.choice(['kokoro','alba','cori']))
                choices=[v for v in choices if v.startswith(group+'/')]
            voice=str(self.rng.choice(choices))
            ids.append(int(self.rng.choice(voices[voice])))
        extra_y=self.y[ids,None].astype(np.float32)
        extra_w=np.where(extra_y[:,0]>0,1.,20.).astype(np.float32)
        # Keras divides by144: preserve main128 mean and add .1*extra16 mean.
        return (np.concatenate([x,np.clip(self.x[ids].astype(np.float32)/25.6,0,26)]),
                np.concatenate([y,extra_y]),
                np.concatenate([weights*(144/128),extra_w*(.1*144/16)]))
