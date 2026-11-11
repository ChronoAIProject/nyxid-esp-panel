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


class FullSourceBatch(PhoneBinary):
    """Entire phrase batch from audited complete synthetic sources, no extra loss."""
    def __init__(self, root, temporal_root=None):
        super().__init__(root, source_balanced=True)
        if self.report['pack_sha256']!='c0c389fc4a22e1fb9807d99e8a7a98e0e028696d2047dd27dce81f1de8fef096':
            raise ValueError('Require frozen full-source pack')
        self.rng=np.random.default_rng(97726)
        self.report={k:v for k,v in self.report.items() if k in ('pack_sha256','report_sha256','groups')}
        self.report.update(seed=97726,positive_slots=32,confusing_negative_slots=48,
            policy='Replace first80 phrase slots with complete-source crops; equal checkpoint groups before voices, equal negative phrase choice. Original48 ambient slots/weights retained; no supplementary BCE or head. Groups do not certify physical-person independence.')
        self.temporal = None
        if temporal_root is not None:
            temporal_root=Path(temporal_root)
            temporal_report=json.loads((temporal_root/'report.json').read_text())
            expected='b84d6b6fb95c50c23084d3fd8c5390d7756d731d87c3f5b754fe68de565ab9bd'
            if (sha(temporal_root/'training.npz')!=expected or temporal_report['pack_sha256']!=expected
                    or temporal_report['base_pack_sha256']!=self.report['pack_sha256']
                    or temporal_report['final_data_used'] or temporal_report['wake_models_scored_for_inclusion']):
                raise ValueError('Require pinned training-only temporal pack')
            z=np.load(temporal_root/'training.npz',allow_pickle=False)
            self.temporal=z['x'];self.temporal_groups={}
            for j,parent in enumerate(z['base_index']):
                self.temporal_groups.setdefault(int(parent),[]).append(j)
            if set(self.temporal_groups)!=set(range(len(self.x))):
                raise ValueError('Temporal expansion must preserve every base example')
            for parent,ids in self.temporal_groups.items():
                zero=[j for j in ids if z['offset'][j]==0]
                if len(zero)!=1:raise ValueError('Require unique offset0')
                np.testing.assert_array_equal(self.temporal[zero[0]],self.x[parent])
            self.temporal_rng=np.random.default_rng(107726)
            self.report['temporal']={'pack_sha256':expected,'report_sha256':sha(temporal_root/'report.json'),
                'seed':107726,'offset_range_frames':[0,14],'included':len(self.temporal),
                'policy':'Original v37 base RNG unchanged; independent uniform valid offset after base draw, preserving long-source mass.'}

    def first80(self):
        indices=[]
        for i in range(80):
            target='hey_nyx' if i<32 else str(self.rng.choice(['hey_nick','hey_next','hey_mix']))
            group=str(self.rng.choice(['kokoro','alba','cori']))
            voices=self.groups[target]
            voice=str(self.rng.choice(sorted(v for v in voices if v.startswith(group+'/'))))
            indices.append(int(self.rng.choice(voices[voice])))
        expected=np.asarray([1]*32+[0]*48,dtype=np.float32)
        np.testing.assert_array_equal(self.y[indices],expected)
        if self.temporal is not None:
            ids=[int(self.temporal_rng.choice(self.temporal_groups[i])) for i in indices]
            return np.clip(self.temporal[ids].astype(np.float32)/25.6,0,26)
        return np.clip(self.x[indices].astype(np.float32)/25.6,0,26)
